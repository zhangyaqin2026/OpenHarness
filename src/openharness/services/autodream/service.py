"""Auto-dream service."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from openharness.config.settings import Settings
from openharness.memory.paths import get_project_memory_dir
from openharness.memory.usage import find_stale_memory_candidates
from openharness.services.autodream.backup import create_memory_backup, diff_memory_dirs
from openharness.services.autodream.lock import (
    list_sessions_touched_since,
    read_last_consolidated_at,
    rollback_consolidation_lock,
    try_acquire_consolidation_lock,
)
from openharness.services.autodream.prompt import build_consolidation_prompt
from openharness.services.session_storage import get_project_session_dir
from openharness.tasks.manager import get_task_manager
from openharness.tasks.types import TaskRecord

SESSION_SCAN_INTERVAL_SECONDS = 10 * 60
_CHILD_ENV = "OPENHARNESS_AUTODREAM_CHILD"
_last_session_scan_at: dict[str, float] = {}
_listener_registered = False


def _enabled(settings: Settings) -> bool:
    return bool(settings.memory.enabled and settings.memory.auto_dream_enabled)


def _has_dream_signal(session_ids: list[str], *, force: bool) -> bool:
    """Return whether recent sessions are worth consolidating."""

    if force:
        return True
    return bool(session_ids)


def _memory_files_mtime_snapshot(memory_dir: Path) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for path in memory_dir.glob("*.md"):
        try:
            snapshot[path.name] = path.stat().st_mtime
        except OSError:
            continue
    return snapshot


def _files_changed_since(memory_dir: Path, before: dict[str, float]) -> list[str]:
    changed: list[str] = []
    for path in sorted(memory_dir.glob("*.md")):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if before.get(path.name) != mtime:
            changed.append(path.name)
    return changed


def _ensure_listener_registered() -> None:
    global _listener_registered
    if _listener_registered:
        return

    async def _listener(task: TaskRecord) -> None:
        if task.type != "dream":
            return
        prior_raw = task.metadata.get("prior_mtime", "")
        memory_dir = task.metadata.get("memory_dir") or None
        if not prior_raw:
            return
        try:
            prior_mtime = float(prior_raw)
        except ValueError:
            return
        if task.status in {"failed", "killed"} or task.metadata.get("preview") == "true":
            rollback_consolidation_lock(task.cwd, prior_mtime, memory_dir=memory_dir)

    get_task_manager().register_completion_listener(_listener)
    _listener_registered = True


def _resolve_memory_dir(cwd: str | Path, memory_dir: str | Path | None) -> Path:
    return Path(memory_dir).expanduser().resolve() if memory_dir is not None else get_project_memory_dir(cwd)


def _resolve_session_dir(cwd: str | Path, session_dir: str | Path | None) -> Path:
    return Path(session_dir).expanduser().resolve() if session_dir is not None else get_project_session_dir(cwd)

"""OpenHarness 长期记忆系统的 “真正执行器”：
再次校验是否满足整理条件（时间、会话数、开关）
加锁 → 防止同时多次整理记忆
备份记忆文件 → 防止出错丢失数据
构建提示词 → 让 LLM 去阅读会话、总结、合并、写入长期记忆
创建子进程 → 后台调用 AI 模型执行记忆整理
监听任务完成 → 记录哪些记忆文件被修改
"""
async def start_dream_now(
    *,
    cwd: str | Path,
    settings: Settings,
    model: str | None = None,
    current_session_id: str | None = None,
    force: bool = False,   #强制启动，跳过时间 / 会话数检查
    memory_dir: str | Path | None = None, #长期记忆存放目录
    session_dir: str | Path | None = None, #对话会话存放目录
    app_label: str = "openharness",     #记忆标识（区分项目 / 个人）
    runner_module: str = "openharness", #运行模块（openharness /ohmo）
    preview: bool = False,
) -> TaskRecord | None:
    """Start a dream task immediately, optionally bypassing time/session gates."""

    if os.environ.get(_CHILD_ENV):
        return None
    if not settings.memory.enabled:
        return None

    cwd = Path(cwd).resolve() #把目录转为绝对路径，避免路径错误。
    resolved_memory_dir = _resolve_memory_dir(cwd, memory_dir) #确定最终记忆目录（不传则用默认 ./memory）。
    resolved_session_dir = _resolve_session_dir(cwd, session_dir) #确定最终会话目录（不传则用默认 ./sessions）。
    last_at = read_last_consolidated_at(cwd, memory_dir=resolved_memory_dir) #读取上一次记忆整理的时间。
    """获取自上次整理后新增的所有会话。"""
    session_ids = list_sessions_touched_since(
        cwd,
        last_at,
        current_session_id=current_session_id,
        session_dir=resolved_session_dir,
    )
    #如果不是强制模式： 间隔时间不够 → 退出     新会话太少 → 退出"""
    if not force:
        hours_since = (time.time() - last_at) / 3600
        if hours_since < settings.memory.auto_dream_min_hours:
            return None
        if len(session_ids) < settings.memory.auto_dream_min_sessions:
            return None
    """检查是否需要整理记忆，没有信号则退出。"""
    if not _has_dream_signal(session_ids, force=force):
        return None

    """加锁！   防止多个进程同时整理记忆导致文件损坏。   加锁失败 → 直接退出。"""
    prior_mtime = try_acquire_consolidation_lock(cwd, memory_dir=resolved_memory_dir)
    if prior_mtime is None:
        return None

    _ensure_listener_registered()
    """确保记忆目录、会话目录一定存在。"""
    resolved_memory_dir.mkdir(parents=True, exist_ok=True)
    resolved_session_dir.mkdir(parents=True, exist_ok=True)
    #给记忆文件拍个快照（记录修改时间），用于后续对比哪些文件变了。"""
    before = _memory_files_mtime_snapshot(resolved_memory_dir)
    #非预览模式 → 自动备份整个记忆目录  防止整理出错，可回滚。"""
    backup_dir = create_memory_backup(resolved_memory_dir, app_label=app_label) if not preview else None
    stale_candidates = find_stale_memory_candidates(cwd, memory_dir=resolved_memory_dir)
    stale_section = "\n".join(
        f"- {header.id or header.path.name}: {header.path.name} "
        f"(importance={header.importance}, updated_at={header.updated_at or 'unknown'})"
        for header in stale_candidates[:20]
    ) or "- (none)"
    extra = (
        f"Application context: `{app_label}`.\n"
        "Tool constraints for this run: only modify files under the memory directory. "
        "Use shell commands only for read-only inspection.\n\n"
        f"Sessions since last consolidation ({len(session_ids)}):\n"
        + "\n".join(f"- {session_id}" for session_id in session_ids)
        + "\n\nUsage-based stale candidates:\n"
        + stale_section
    )
    """!!!核心：构建记忆整理提示词
    让 LLM 做：读取会话、提取关键信息、合并旧记忆、去重、精简、结构化、写入 memory 文件"""
    prompt = build_consolidation_prompt(resolved_memory_dir, resolved_session_dir, extra, preview=preview)
    #获取源码根目录，构造环境变量。
    src_root = Path(__file__).resolve().parents[3]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    """构造子进程环境变量，告诉子进程：这是记忆整理进程、记忆目录在哪、配置目录在哪、使用哪个配置文件"""
    env = {
        _CHILD_ENV: "1",
        "OPENHARNESS_AUTODREAM_MEMORY_DIR": str(resolved_memory_dir),
        "OPENHARNESS_CONFIG_DIR": str(Path.home() / ".openharness"),
        "OPENHARNESS_PROFILE": settings.active_profile,
        "PYTHONPATH": str(src_root) + ((os.pathsep + existing_pythonpath) if existing_pythonpath else ""),
    }
    try:
        """构造子进程启动命令：python -m openharness ... 或 python -m ohmo ..."""
        argv = [
            sys.executable,
            "-m",
            runner_module,
        ]
        #记忆整理进程跳过权限检查（内部信任）。
        if runner_module == "openharness":
            argv.append("--dangerously-skip-permissions")
        if runner_module == "ohmo":
            workspace = resolved_memory_dir.parent
            argv.extend(["--workspace", str(workspace)])
            if settings.active_profile:
                argv.extend(["--profile", settings.active_profile])
        """传入使用的 AI 模型。"""
        if model:
            argv.extend(["--model", model])
        if runner_module == "openharness" and settings.provider != "anthropic_claude":
            if settings.base_url:
                argv.extend(["--base-url", settings.base_url])
            if settings.api_format:
                argv.extend(["--api-format", settings.api_format])
        try:
            #获取 API Key / 令牌。
            auth = settings.resolve_auth()
            if runner_module == "openharness" and auth.auth_kind == "api_key":
                argv.extend(["--api-key", auth.value])
            elif runner_module == "ohmo" and auth.auth_kind == "api_key":
                env["OPENHARNESS_API_KEY"] = auth.value
            elif auth.value:
                env["ANTHROPIC_AUTH_TOKEN"] = auth.value
                env.pop("ANTHROPIC_API_KEY", None)
                env.pop("OPENAI_API_KEY", None)
                env.pop("OPENHARNESS_API_KEY", None)
        except Exception:
            pass
        argv.extend(["--print", prompt])
        """!!!把整理提示词传给子进程。   创建并启动后台子进程 → 真正让 AI 开始 “做梦 / 整理记忆”。"""
        task = await get_task_manager().create_shell_task(
            description="dreaming",
            cwd=cwd,
            task_type="dream",
            env=env,
            argv=argv,
        )
        task.prompt = prompt #把提示词存到任务里。
    except Exception: #出错 → 释放锁、回滚，保证记忆不损坏。
        rollback_consolidation_lock(cwd, prior_mtime, memory_dir=resolved_memory_dir)
        raise
    #给任务打上元数据：整理阶段、整理多少会话、记忆目录、备份目录、是否强制、是否预览
    task.metadata.update(
        {
            "phase": "starting",
            "sessions_reviewing": str(len(session_ids)),
            "prior_mtime": str(prior_mtime),
            "memory_dir": str(resolved_memory_dir),
            "session_dir": str(resolved_session_dir),
            "force": str(force).lower(),
            "app_label": app_label,
            "runner_module": runner_module,
            "preview": str(preview).lower(),
            "backup_dir": str(backup_dir or ""),
        }
    )
    """定义任务完成回调：检查哪些记忆文件被修改、记录新增、修改、删除的文件、更新任务状态"""
    async def _mark_changed_on_completion(done: TaskRecord) -> None:
        if done.id != task.id or done.status != "completed":
            return
        changed = _files_changed_since(resolved_memory_dir, before)
        if backup_dir is not None:
            diff = diff_memory_dirs(backup_dir, resolved_memory_dir)
            done.metadata["files_added"] = "\n".join(diff["added"])
            done.metadata["files_changed"] = "\n".join(diff["changed"])
            done.metadata["files_removed"] = "\n".join(diff["removed"])
        if changed:
            done.metadata["phase"] = "updating"
            done.metadata["files_touched"] = "\n".join(changed)
    #注册监听器，任务结束时自动触发。
    get_task_manager().register_completion_listener(_mark_changed_on_completion)
    """返回正在运行的记忆整理任务。"""
    return task

""" 1. 读取新会话（sessions/*.json）
   2. 读取旧记忆（memory/*.md）
   3. LLM 整理记忆
   4. 写入新记忆文件
OpenHarness 长期记忆自动整理（Auto-Dream）的核心入口函数，作用是：
先做一系列轻量 “门槛校验”，满足条件才真正启动后台记忆整理，避免频繁、无效执行。
   """
async def execute_auto_dream(
    *,
    cwd: str | Path, #当前工作目录（项目根目录）
    settings: Settings,#系统配置（记忆开关、间隔时间、最小会话数）
    model: str | None = None,#用于整理记忆的 AI 模型
    current_session_id: str | None = None,
    memory_dir: str | Path | None = None,#长期记忆存放目录
    session_dir: str | Path | None = None,#对话会话存放目录
    app_label: str = "openharness",
    runner_module: str = "openharness",
    preview: bool = False, #是否预览模式（不真正写入记忆）
) -> TaskRecord | None:
    """Run the cheap auto-dream gates and start a background dream when eligible."""
    #如果是子进程，直接返回，不重复执行。
    if os.environ.get(_CHILD_ENV):
        return None
    if not _enabled(settings):
        return None

    cwd = Path(cwd).resolve()#把工作目录变成绝对路径，避免路径错误。
    resolved_memory_dir = _resolve_memory_dir(cwd, memory_dir)#确定记忆目录（用户没传就用默认：./memory）。
    resolved_session_dir = _resolve_session_dir(cwd, session_dir)#确定会话目录（用户没传就用默认：./sessions）。
    last_at = read_last_consolidated_at(cwd, memory_dir=resolved_memory_dir)#读取上一次记忆整理的时间。
    hours_since = (time.time() - last_at) / 3600  #计算距离上次整理过了多少小时。
    #如果没到最小间隔时间（比如 24 小时），不执行，直接返回。
    if hours_since < settings.memory.auto_dream_min_hours:
        return None
    #用记忆目录作为 key，记录当前时间。
    key = str(resolved_memory_dir)
    now = time.time()
    #10 分钟内已经扫描过会话 → 直接返回（防频繁扫描）。
    if now - _last_session_scan_at.get(key, 0) < SESSION_SCAN_INTERVAL_SECONDS:
        return None
    _last_session_scan_at[key] = now #记录本次扫描时间。

    #列出上次整理之后新增的会话 ID。
    session_ids = list_sessions_touched_since(
        cwd,
        last_at,
        current_session_id=current_session_id,
        session_dir=resolved_session_dir,
    )
    #如果新会话太少（比如不足 5 个），不整理。  _has_dream_signal检查是否需要整理（信号判断），没有则返回。
    if len(session_ids) < settings.memory.auto_dream_min_sessions:
        return None
    if not _has_dream_signal(session_ids, force=False):
        return None

    #所有门槛都通过 → 正式启动记忆整理。
    return await start_dream_now(
        cwd=cwd,
        settings=settings,
        model=model,
        current_session_id=current_session_id,
        force=False,
        memory_dir=resolved_memory_dir,
        session_dir=resolved_session_dir,
        app_label=app_label,
        runner_module=runner_module,
        preview=preview,
    )


def schedule_auto_dream(**kwargs: object) -> None:
    """Fire-and-forget auto-dream scheduling."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(execute_auto_dream(**kwargs))  # type: ignore[arg-type]
