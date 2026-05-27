"""Core tool-aware query loop."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiTextDeltaEvent,
    SupportsStreamingMessages,
)
from openharness.api.provider import is_model_multimodal
from openharness.api.usage import UsageSnapshot
from openharness.config.paths import get_data_dir
from openharness.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
)
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    CompactProgressEvent,
    ErrorEvent,
    StatusEvent,
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.hooks import HookEvent, HookExecutor
from openharness.permissions.checker import PermissionChecker
from openharness.services.tool_outputs import tool_output_inline_chars, tool_output_preview_chars
from openharness.tools.base import ToolExecutionContext
from openharness.tools.base import ToolRegistry

AUTO_COMPACT_STATUS_MESSAGE = "Auto-compacting conversation memory to keep things fast and focused."
REACTIVE_COMPACT_STATUS_MESSAGE = "Prompt too long; compacting conversation memory and retrying."
MAX_SAFE_COMPLETION_TOKENS = 128_000

log = logging.getLogger(__name__)


PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]

MAX_TRACKED_READ_FILES = 6
MAX_TRACKED_SKILLS = 8
MAX_TRACKED_ASYNC_AGENT_EVENTS = 8
MAX_TRACKED_ASYNC_AGENT_TASKS = 12
MAX_TRACKED_WORK_LOG = 10
MAX_TRACKED_USER_GOALS = 5
MAX_TRACKED_ACTIVE_ARTIFACTS = 8
MAX_TRACKED_VERIFIED_WORK = 10

"""AI 工具感知查询核心循环，处理对话流式响应、工具调用、权限校验、错误捕获、对话压缩、图片预处理，
管理任务状态，实现稳定的多轮 AI 交互与工具执行。    """

"""！上下文超长错误判断核心：
判断异常是否为提示词过长 / 上下文超限错误；决定是否触发对话压缩的关键判断函数。
统一识别各大模型的 “上下文超限 / 提示词过长” 报错
触发自动压缩对话逻辑 
"""
def _is_prompt_too_long_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "prompt too long",
            "context_length_exceeded",
            "context length",
            "maximum context",
            "context window",
            "input tokens exceed",
            "messages resulted in",
            "reduce the length of the messages",
            "configured limit",
            "too many tokens",
            "too large for the model",
            "maximum context length",
            "exceed_context",
            "exceeds the available context size",
            "available context size",
        )
    )

"""！令牌限制核心：限制单次请求输出令牌数，避免模型拒绝请求

限制单次请求最大输出令牌
避免模型因配置过大直接拒绝请求
动态适配模型限制
一句话总结：保证 API 请求不会因令牌过大失败。"""
def _bounded_completion_tokens(max_tokens: int, context_window_tokens: int | None = None) -> int:
    """Return a conservative per-request output token cap.

    Some OpenAI-compatible providers reject very large ``max_tokens`` before
    the request reaches model-side context management.  Keep oversized user
    config from making every turn fail while preserving normal defaults.
    """
    limit = MAX_SAFE_COMPLETION_TOKENS
    if context_window_tokens is not None and context_window_tokens > 0:
        limit = min(limit, int(context_window_tokens))
    return max(1, min(int(max_tokens), limit))

"""从异常中解析模型支持的最大完成令牌数   """
def _extract_completion_token_limit(exc: Exception) -> int | None:
    """Parse provider errors such as "supports at most 128000 completion tokens"."""
    text = str(exc).lower().replace(",", "")
    patterns = (
        r"supports at most\s+(\d+)\s+completion tokens",
        r"at most\s+(\d+)\s+completion tokens",
        r"max(?:imum)?(?:_completion)?[_\s-]tokens.*?(?:<=|less than or equal to|at most)\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return max(1, int(match.group(1)))
            except ValueError:
                return None
    return None

"""判断异常是否为输出令牌超限错误   """
def _is_completion_token_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("max_tokens" in text or "max_completion_tokens" in text)
        and ("too large" in text or "at most" in text or "completion tokens" in text)
    )

"""自定义异常，AI 交互轮次超出上限时抛出    """
class MaxTurnsExceeded(RuntimeError):
    """Raised when the agent exceeds the configured max_turns for one user prompt."""

    def __init__(self, max_turns: int) -> None:
        super().__init__(f"Exceeded maximum turn limit ({max_turns})")
        self.max_turns = max_turns


@dataclass
class QueryContext:
    """Context shared across a query run.一次 AI 对话的 “任务说明书 + 工具箱 + 权限卡 + 运行环境

    ToolRegistry这里跟tool里面联动”"""

    api_client: SupportsStreamingMessages #调用大模型 API 的客户端，负责发请求、收流式响应、处理消息；和 AI 模型说话的 “电话”
    tool_registry: ToolRegistry #存放所有可用工具（read_file、bash、web_search、skill 等）；AI 的工具箱
    permission_checker: PermissionChecker #判断 AI 能不能执行某个操作（删文件、执行命令、访问路径）；AI 的保安 / 权限门禁
    cwd: Path  #代码执行的当前目录；AI 当前所在的文件夹
    model: str #要调用的模型（gpt-4o、claude-3 等）；本次用哪个 AI 大脑
    system_prompt: str #给 AI 的身份设定、规则、行为要求；AI 的角色设定 ；例子：“你是一个编程助手…”
    max_tokens: int #AI 单次最多返回多少 token；AI 每轮说话的最长长度
    context_window_tokens: int | None = None #模型最大能接收多少 token（上下文长度）；AI 的短期记忆上限
    auto_compact_threshold_tokens: int | None = None #超过这个 token 数就自动压缩对话；防止上下文溢出；记忆太多时，自动开始精简记忆
    permission_prompt: PermissionPrompt | None = None #执行危险操作前，弹框问用户 “是否允许”
    ask_user_prompt: AskUserPrompt | None = None #AI 需要更多信息时，向用户提问
    max_turns: int | None = 200 #防止 AI 无限循环调用工具 ；AI 最多连续思考多少次
    hook_executor: HookExecutor | None = None #在工具执行前 / 后、对话开始 / 结束时触发自定义逻辑 ；事件触发器（插件系统）
    tool_metadata: dict[str, object] | None = None
    #AI 的长期记忆 + 任务进度条：存放工具执行的临时状态、记忆、任务进度：最近读了哪些文件、当前任务目标、激活的文档、异步任务状态

"""添加唯一元素到列表，超出长度限制则删除旧元素    """
def _append_capped_unique(bucket: list[Any], value: Any, *, limit: int) -> None:
    if value in bucket:
        bucket.remove(value)
    bucket.append(value)
    if len(bucket) > limit:
        del bucket[:-limit]

"""获取 / 初始化任务聚焦状态数据    """
def _task_focus_state(tool_metadata: dict[str, object] | None) -> dict[str, object]:
    if tool_metadata is None:
        return {}
    value = tool_metadata.setdefault(
        "task_focus_state",
        {
            "goal": "",
            "recent_goals": [],
            "active_artifacts": [],
            "verified_state": [],
            "next_step": "",
        },
    )
    if isinstance(value, dict):
        value.setdefault("goal", "")
        value.setdefault("recent_goals", [])
        value.setdefault("active_artifacts", [])
        value.setdefault("verified_state", [])
        value.setdefault("next_step", "")
        return value
    replacement = {
        "goal": "",
        "recent_goals": [],
        "active_artifacts": [],
        "verified_state": [],
        "next_step": "",
    }
    tool_metadata["task_focus_state"] = replacement
    return replacement

"""精简文本，保留前 240 字符用于状态记录    """
def _summarize_focus_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    return normalized[:240]

"""从用户输入里提取目标 → 存到程序状态 → 保留最近几条历史 → 更新当前主目标。
    就是AI 的短期记忆 + 目标聚焦功能。
   tool_metadata = 程序的状态 / 上下文数据（字典或空）  prompt = 用户刚才说的话"""
def remember_user_goal(
    tool_metadata: dict[str, object] | None,
    prompt: str,
) -> None:
    state = _task_focus_state(tool_metadata) #拿到当前对话的状态存储区（一个字典，用来存记忆）
    summary = _summarize_focus_text(prompt) #把用户说的话提炼成简短目标
    if not summary:
        return #如果提炼不出有效目标，就直接退出，不存
    recent_goals = state.setdefault("recent_goals", []) #在状态里创建 / 拿到 “最近目标列表”，默认是空列表
    if isinstance(recent_goals, list):
        #把新目标添加到历史列表，同时保证：不重复；不超过最大数量（避免内存爆炸）
        _append_capped_unique(recent_goals, summary, limit=MAX_TRACKED_USER_GOALS)
    state["goal"] = summary # 把最新目标设为当前主目标，后面 AI 就能直接读取 state["goal"] 知道用户要干嘛

"""记录当前活跃的文件 / 工具等产物    """
def _remember_active_artifact(
    tool_metadata: dict[str, object] | None,
    artifact: str,
) -> None:
    normalized = artifact.strip()
    if not normalized:
        return
    state = _task_focus_state(tool_metadata)
    artifacts = state.setdefault("active_artifacts", [])
    if isinstance(artifacts, list):
        _append_capped_unique(artifacts, normalized[:240], limit=MAX_TRACKED_ACTIVE_ARTIFACTS)

"""记录已确认完成的工作内容    """
def _remember_verified_work(
    tool_metadata: dict[str, object] | None,
    entry: str,
) -> None:
    normalized = entry.strip()
    if not normalized:
        return
    bucket = _tool_metadata_bucket(tool_metadata, "recent_verified_work")
    _append_capped_unique(bucket, normalized[:320], limit=MAX_TRACKED_VERIFIED_WORK)
    state = _task_focus_state(tool_metadata)
    verified_state = state.setdefault("verified_state", [])
    if isinstance(verified_state, list):
        _append_capped_unique(verified_state, normalized[:320], limit=MAX_TRACKED_VERIFIED_WORK)

"""获取工具元数据中的指定列表容器    """
def _tool_metadata_bucket(
    tool_metadata: dict[str, object] | None,
    key: str,
) -> list[Any]:
    if tool_metadata is None:
        return []
    value = tool_metadata.setdefault(key, [])
    if isinstance(value, list):
        return value
    replacement: list[Any] = []
    tool_metadata[key] = replacement
    return replacement

"""记录读取的文件信息，限制存储数量    """
def _remember_read_file(
    tool_metadata: dict[str, object] | None,
    *,
    path: str,
    offset: int,
    limit: int,
    output: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "read_file_state")
    preview_lines = [line.strip() for line in output.splitlines()[:6] if line.strip()]
    entry = {
        "path": path,
        "span": f"lines {offset + 1}-{offset + limit}",
        "preview": " | ".join(preview_lines)[:320],
        "timestamp": time.time(),
    }
    if isinstance(bucket, list):
        bucket[:] = [
            existing
            for existing in bucket
            if not isinstance(existing, dict) or str(existing.get("path") or "") != path
        ]
        bucket.append(entry)
        if len(bucket) > MAX_TRACKED_READ_FILES:
            del bucket[:-MAX_TRACKED_READ_FILES]

"""记录调用过的技能，限制存储数量    """
def _remember_skill_invocation(
    tool_metadata: dict[str, object] | None,
    *,
    skill_name: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "invoked_skills")
    normalized = skill_name.strip()
    if not normalized:
        return
    if normalized in bucket:
        bucket.remove(normalized)
    bucket.append(normalized)
    if len(bucket) > MAX_TRACKED_SKILLS:
        del bucket[:-MAX_TRACKED_SKILLS]

"""记录异步代理的操作事件    """
def _remember_async_agent_activity(
    tool_metadata: dict[str, object] | None,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    output: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "async_agent_state")
    if tool_name == "agent":
        description = str(tool_input.get("description") or tool_input.get("prompt") or "").strip()
        summary = f"Spawned async agent. {description}".strip()
        if output.strip():
            summary = f"{summary} [{output.strip()[:180]}]".strip()
    elif tool_name == "send_message":
        target = str(tool_input.get("task_id") or "").strip()
        summary = f"Sent follow-up message to async agent {target}".strip()
    else:
        summary = output.strip()[:220] or f"Async agent activity via {tool_name}"
    bucket.append(summary)
    if len(bucket) > MAX_TRACKED_ASYNC_AGENT_EVENTS:
        del bucket[:-MAX_TRACKED_ASYNC_AGENT_EVENTS]

"""解析创建的代理 ID 和任务 ID    """
def _parse_spawned_agent_identity(
    output: str,
    metadata: dict[str, object] | None = None,
) -> tuple[str, str] | None:
    if isinstance(metadata, dict):
        agent_id = str(metadata.get("agent_id") or "").strip()
        task_id = str(metadata.get("task_id") or "").strip()
        if agent_id and task_id:
            return agent_id, task_id
    match = re.search(r"Spawned agent (.+?) \(task_id=(\S+?)(?:[,)]|$)", output.strip())
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip()

"""记录异步代理任务信息    """
def _remember_async_agent_task(
    tool_metadata: dict[str, object] | None,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    output: str,
    result_metadata: dict[str, object] | None = None,
) -> None:
    if tool_name != "agent":
        return
    identity = _parse_spawned_agent_identity(output, result_metadata)
    if identity is None:
        return
    agent_id, task_id = identity
    bucket = _tool_metadata_bucket(tool_metadata, "async_agent_tasks")
    description = str(tool_input.get("description") or tool_input.get("prompt") or "").strip()
    entry = {
        "agent_id": agent_id,
        "task_id": task_id,
        "description": description[:240],
        "status": "spawned",
        "notification_sent": False,
        "spawned_at": time.time(),
    }
    bucket[:] = [
        existing
        for existing in bucket
        if not isinstance(existing, dict) or str(existing.get("task_id") or "") != task_id
    ]
    bucket.append(entry)
    if len(bucket) > MAX_TRACKED_ASYNC_AGENT_TASKS:
        del bucket[:-MAX_TRACKED_ASYNC_AGENT_TASKS]

"""记录工作日志，限制存储数量    """
def _remember_work_log(
    tool_metadata: dict[str, object] | None,
    *,
    entry: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "recent_work_log")
    normalized = entry.strip()
    if not normalized:
        return
    bucket.append(normalized[:320])
    if len(bucket) > MAX_TRACKED_WORK_LOG:
        del bucket[:-MAX_TRACKED_WORK_LOG]

"""更新 AI 的计划模式状态   """
def _update_plan_mode(tool_metadata: dict[str, object] | None, mode: str) -> None:
    if tool_metadata is None:
        return
    tool_metadata["permission_mode"] = mode

"""！！！根据工具执行结果，记录对应的状态信息    """
def _record_tool_carryover(
    context: QueryContext,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    tool_output: str,
    tool_result_metadata: dict[str, object] | None,
    is_error: bool,
    resolved_file_path: str | None,
) -> None:
    if is_error:
        return
    if resolved_file_path is not None:
        _remember_active_artifact(context.tool_metadata, resolved_file_path)
    if tool_name == "read_file" and resolved_file_path is not None:
        offset = int(tool_input.get("offset") or 0)
        limit = int(tool_input.get("limit") or 200)
        _remember_read_file(
            context.tool_metadata,
            path=resolved_file_path,
            offset=offset,
            limit=limit,
            output=tool_output,
        )
        _remember_verified_work(
            context.tool_metadata,
            f"Inspected file {resolved_file_path} (lines {offset + 1}-{offset + limit})",
        )
    elif tool_name == "skill":
        _remember_skill_invocation(
            context.tool_metadata,
            skill_name=str(tool_input.get("name") or ""),
        )
        skill_name = str(tool_input.get("name") or "").strip()
        if skill_name:
            _remember_active_artifact(context.tool_metadata, f"skill:{skill_name}")
            _remember_verified_work(context.tool_metadata, f"Loaded skill {skill_name}")
    elif tool_name in {"agent", "send_message"}:
        _remember_async_agent_activity(
            context.tool_metadata,
            tool_name=tool_name,
            tool_input=tool_input,
            output=tool_output,
        )
        _remember_async_agent_task(
            context.tool_metadata,
            tool_name=tool_name,
            tool_input=tool_input,
            output=tool_output,
            result_metadata=tool_result_metadata,
        )
        description = str(tool_input.get("description") or tool_input.get("prompt") or tool_name).strip()
        _remember_verified_work(
            context.tool_metadata,
            f"Confirmed async-agent activity via {tool_name}: {description[:180]}",
        )
    elif tool_name == "enter_plan_mode":
        _update_plan_mode(context.tool_metadata, "plan")
    elif tool_name == "exit_plan_mode":
        _update_plan_mode(context.tool_metadata, "default")
    elif tool_name == "web_fetch":
        url = str(tool_input.get("url") or "").strip()
        if url:
            _remember_active_artifact(context.tool_metadata, url)
            _remember_verified_work(context.tool_metadata, f"Fetched remote content from {url}")
    elif tool_name == "web_search":
        query = str(tool_input.get("query") or "").strip()
        if query:
            _remember_verified_work(context.tool_metadata, f"Ran web search for {query[:180]}")
    elif tool_name == "glob":
        pattern = str(tool_input.get("pattern") or "").strip()
        if pattern:
            _remember_verified_work(context.tool_metadata, f"Expanded glob pattern {pattern[:180]}")
    elif tool_name == "grep":
        pattern = str(tool_input.get("pattern") or "").strip()
        if pattern:
            _remember_verified_work(context.tool_metadata, f"Checked repository matches for grep pattern {pattern[:180]}")
    elif tool_name == "bash":
        command = str(tool_input.get("command") or "").strip()
        summary = tool_output.splitlines()[0].strip() if tool_output.strip() else "no output"
        _remember_verified_work(
            context.tool_metadata,
            f"Ran bash command {command[:160]} [{summary[:120]}]",
        )
    if tool_name == "read_file" and resolved_file_path is not None:
        _remember_work_log(
            context.tool_metadata,
            entry=f"Read file {resolved_file_path}",
        )
    elif tool_name == "bash":
        command = str(tool_input.get("command") or "").strip()
        summary = tool_output.splitlines()[0].strip() if tool_output.strip() else "no output"
        _remember_work_log(
            context.tool_metadata,
            entry=f"Ran bash: {command[:160]} [{summary[:120]}]",
        )
    elif tool_name == "grep":
        pattern = str(tool_input.get("pattern") or "").strip()
        _remember_work_log(
            context.tool_metadata,
            entry=f"Searched with grep pattern={pattern[:160]}",
        )
    elif tool_name == "skill":
        _remember_work_log(
            context.tool_metadata,
            entry=f"Loaded skill {str(tool_input.get('name') or '').strip()}",
        )
    elif tool_name in {"agent", "send_message"}:
        _remember_work_log(
            context.tool_metadata,
            entry=f"Async agent action via {tool_name}",
        )
    elif tool_name == "enter_plan_mode":
        _remember_work_log(context.tool_metadata, entry="Entered plan mode")
    elif tool_name == "exit_plan_mode":
        _remember_work_log(context.tool_metadata, entry="Exited plan mode")

"""创建并返回工具输出产物存储目录    """
def _tool_artifact_dir() -> Path:
    artifact_dir = get_data_dir() / "tool_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir

"""生成安全的工具产物文件名    """
def _safe_tool_artifact_name(tool_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name.strip())
    return (normalized or "tool")[:80]

"""超长工具输出写入文件，仅返回预览文本    """
def _offload_tool_output_if_needed(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
) -> tuple[str, Path | None]:
    inline_limit = tool_output_inline_chars()
    if len(output) <= inline_limit:
        return output, None

    artifact_path = (
        _tool_artifact_dir()
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{_safe_tool_artifact_name(tool_name)}-{uuid4().hex[:12]}.txt"
    )
    artifact_path.write_text(output, encoding="utf-8", errors="replace")
    preview = output[:tool_output_preview_chars()]
    omitted = max(0, len(output) - len(preview))
    inline = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Tool use id: {tool_use_id}\n"
        f"Original size: {len(output)} chars\n"
        f"Full output saved to: {artifact_path}\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        inline += f" ({omitted} chars omitted)"
    if preview:
        inline += f"\n\nPreview:\n{preview}"
    return inline, artifact_path


# ---------------------------------------------------------------------------
# Image preprocessing — convert ImageBlocks to text for non-multimodal models
# ---------------------------------------------------------------------------

_IMAGE_PREPROCESS_STATUS = "Converting image to text description via vision model…"

"""！！！多模态核心：图片处理唯一核心（非多模态模型下，将图片转为文本描述）
    判断模型是否支持多模态
不支持则自动调用图像描述工具把图片转文本
流式返回处理状态，保持 UI 响应
一句话总结：让非视觉模型也能处理图片输入。"""
async def _preprocess_images_in_messages(
    messages: list[ConversationMessage],
    context: QueryContext,
) -> AsyncIterator[StreamEvent]:
    """Scan messages for ImageBlocks and convert them to text if the active
    model does not support multimodal input.

    Yields status events during conversion so the UI stays responsive.
    """
    if is_model_multimodal(context.model):
        return

    vision_config = context.tool_metadata.get("vision_model_config")
    if not vision_config:
        # No vision model configured — skip preprocessing.
        return

    # Collect all ImageBlocks with their parent message index and block index
    pending: list[tuple[int, int, ImageBlock]] = []
    for msg_idx, msg in enumerate(messages):
        if msg.role != "user":
            continue
        for blk_idx, block in enumerate(msg.content):
            if isinstance(block, ImageBlock):
                pending.append((msg_idx, blk_idx, block))

    if not pending:
        return

    yield StatusEvent(message=_IMAGE_PREPROCESS_STATUS)

    # Process images in parallel
    async def _describe(msg_idx: int, blk_idx: int, block: ImageBlock) -> tuple[int, int, str]:
        tool = context.tool_registry.get("image_to_text")
        if tool is None:
            return msg_idx, blk_idx, "[Image: could not describe — image_to_text tool not available]"

        # Build tool input
        tool_input_data: dict[str, object] = {
            "image_data": block.data,
            "media_type": block.media_type,
            "prompt": "Describe this image in detail, including any text, "
                      "UI elements, code, diagrams, or visual information present.",
        }

        try:
            parsed = tool.input_model.model_validate(tool_input_data)
        except Exception:
            return msg_idx, blk_idx, "[Image: could not parse image data]"

        exec_context = ToolExecutionContext(
            cwd=context.cwd,
            metadata={
                "vision_model_config": vision_config,
                **(context.tool_metadata or {}),
            },
        )
        result = await tool.execute(parsed, exec_context)
        if result.is_error:
            return msg_idx, blk_idx, f"[Image description failed: {result.output}]"
        return msg_idx, blk_idx, result.output

    results = await asyncio.gather(*[_describe(mi, bi, blk) for mi, bi, blk in pending])

    # Replace ImageBlocks with TextBlocks in-place
    for msg_idx, blk_idx, description in results:
        msg = messages[msg_idx]
        msg.content[blk_idx] = TextBlock(text=description)

"""！！！ 整个文件的入口与大脑，是 AI 对话 + 工具调用的主循环 （核心查询循环，驱动 AI 多轮对话与工具调用全流程）

管理多轮 AI 对话（直到 AI 停止调用工具）
自动压缩超长对话上下文
预处理图片（非多模态模型转文字）
调用模型 API、流式返回结果
捕获错误（上下文超长、网络错误、令牌超限）
调度工具执行
维护对话消息队列
一句话总结：所有 AI 交互的总控中心，没有它整个系统无法运转。


初始化配置：初始化对话压缩状态、计算安全输出 Token 上限、准备进度推送队列
轮次循环管控：开启多轮对话循环，限制最大轮数，防止 AI 无限死循环
上下文自动压缩：每轮前置检测 Token，超长自动精简对话记忆，避免上下文溢出
图片兼容预处理：非多模态模型自动把图片转为文字描述，适配各类模型
流式调用大模型：请求模型接口，流式推送文字、处理接口重试事件，接收完整回复
全局异常容错：捕获 Token 超限、上下文过长、网络 / API 错误，自动降级、重试或终止会话
存入对话上下文：将 AI 回复写入消息列表，空回复直接丢弃保会话稳定
调度执行工具：无工具调用则结束会话；有工具则单工具串行、多工具并发执行，调用_execute_tool_call落地执行，工具结果回填对话，进入下一轮循环
"""
async def run_query(
    context: QueryContext,  # 上下文：包含模型、工具、权限、配置等所有信息
    messages: list[ConversationMessage],  # 对话历史列表
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]: # 返回值：异步迭代器，不断输出【流式事件】和【令牌用量】
    """Run the conversation loop until the model stops requesting tools.
    Auto-compaction is checked at the start of each turn.  When the
    estimated token count exceeds the model's auto-compact threshold,
    the engine first tries a cheap microcompact (clearing old tool result
    content) and, if that is not enough, performs a full LLM-based
    summarization of older messages.
    """
    """运行对话循环，直到模型停止请求调用工具。
       每一轮开始前都会检查是否需要自动压缩对话。
       当token超过阈值，先轻量压缩（清理旧工具结果），不够就 full 压缩（LLM总结旧消息）。"""
    # 延迟导入：只有用到时才加载压缩相关模块，节省资源
    from openharness.services.compact import (
        AutoCompactState,# 对话压缩状态管理
        auto_compact_if_needed,# 自动压缩对话的核心函数
    )

    compact_state = AutoCompactState() # 创建压缩状态实例
    reactive_compact_attempted = False # 是否已经做过【被动压缩】（报错后压缩）
    last_compaction_result: tuple[list[ConversationMessage], bool] = (messages, False) # 上次压缩结果：(压缩后的消息，是否真的压缩了)
    # 计算安全的最大输出token，防止模型拒绝请求
    effective_max_tokens = _bounded_completion_tokens(
        context.max_tokens,   # 用户配置的最大token
        context.context_window_tokens, # 模型上下文窗口大小
    )
    reported_token_clamp = False # 是否已经提示过“token被限制”

    # -------------------------- 内部函数：流式压缩对话
    async def _stream_compaction(
        *,
        trigger: str,          # 触发原因：auto / reactive
        force: bool = False,   # 是否强制压缩
    ) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
        nonlocal last_compaction_result  # 使用外部的变量
        progress_queue: asyncio.Queue[CompactProgressEvent] = asyncio.Queue()# 异步队列：用来传递压缩进度事件

        # 进度回调：把压缩进度放进队列
        async def _progress(event: CompactProgressEvent) -> None:
            await progress_queue.put(event)

        # 创建异步任务：执行自动压缩
        task = asyncio.create_task(
            auto_compact_if_needed(
                messages,
                api_client=context.api_client,
                model=context.model,
                system_prompt=context.system_prompt,
                state=compact_state,
                progress_callback=_progress, # 进度回调
                force=force,
                trigger=trigger,
                hook_executor=context.hook_executor,
                carryover_metadata=context.tool_metadata,
                context_window_tokens=context.context_window_tokens,
                auto_compact_threshold_tokens=context.auto_compact_threshold_tokens,
            )
        )

        # 循环读取队列，实时返回压缩进度事件
        while True:
            try:
                # 每50ms取一次进度，不阻塞主循环
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.05)
                yield event, None
            except asyncio.TimeoutError:# 压缩任务结束就退出
                if task.done():
                    break
                continue
        # 把队列剩余事件全部输出
        while not progress_queue.empty():
            yield progress_queue.get_nowait(), None
        # 保存最终压缩结果
        last_compaction_result = await task
        return

    # -------------------------- 主循环：一轮一轮和AI对话
    turn_count = 0 # 记录当前是第几轮对话
    # 循环条件：没达到最大轮数就一直跑
    """第五、然后循环 = ReAct 闭环 ，ReAct 的循环载体：一轮 = 一次 Reason → Act → Observe
    Coordinator 模式的核心就是多轮 ReAct 循环："""
    while context.max_turns is None or turn_count < context.max_turns:
        turn_count += 1 # 轮数+1

        # 如果token被限制了，且还没提示过 → 发一条状态提示
        if effective_max_tokens != context.max_tokens and not reported_token_clamp:
            reported_token_clamp = True
            yield StatusEvent(
                message=(
                    "Requested max_tokens="
                    f"{context.max_tokens} exceeds the safe per-request output cap; "
                    f"using {effective_max_tokens}."
                )
            ), None

        # --- auto-compact check before calling the model ---------------
            """====================== 步骤1：自动压缩对话
        coordinator2.1:每轮开始前检查 token,超了先轻量压缩（清理旧工具结果）,再超就 LLM 总结历史"""
        async for event, usage in _stream_compaction(trigger="auto"):
            yield event, usage  # 流式返回压缩进度

        # 拿到压缩后的消息
        compacted_messages, was_compacted = last_compaction_result
        if compacted_messages is not messages:
            messages[:] = compacted_messages # 替换成压缩后的对话
        # ---------------------------------------------------------------

        # --- image preprocessing: convert ImageBlocks to text for non-vision models ---
        # ====================== 步骤2：图片预处理：不支持多模态的模型 → 把图片转成文字描述
        async for event in _preprocess_images_in_messages(messages, context):
            yield event, None

        # ====================== 步骤3：调用AI模型流式输出 ======================
        final_message: ConversationMessage | None = None
        # 最终完整消息  ConversationMessage表示一条完整对话消息（用户 / 助手），可包含文本、图片、工具调用
        usage = UsageSnapshot() # token用量

        try: # 调用API，流式获取返回结果
            """!!!第一、ReAct = Reason】，把对话历史 messages、所有可用工具 发给 LLM,让模型思考并决定：直接回答用户,还是调用工具（tool_uses）
            api_client.stream_message这句是模型思考（调用 LLM）；模型根据上下文 → 决定要不要调用工具、调用什么工具。
            stream_message()：用于 AI 模型对话流式输出，对外提供流式对话 + 自动重试
            
            coordinator2.2:模型根据上下文决定：直接回答,还是调用工具,还是创建子代理（Coordinator 模式）
           
            3、主 AI 调用工具创建子代理（真正创建的地方） → 主 AI 思考后，会调用 agent 工具 = 创建子代理。
            tools 里包含：def get_coordinator_tools() -> list[str]:return ["agent", "send_message", "task_stop"]"""
            async for event in context.api_client.stream_message(
                ApiMessageRequest(
                    model=context.model,
                    messages=messages,    # messages全部对话历史
                    system_prompt=context.system_prompt,# ！！！4. 把全局注册好的技能传给 AI
                    max_tokens=effective_max_tokens,   # 安全token上限
                    tools=context.tool_registry.to_api_schema(),  # 给模型看的工具列表；to_api_schema把工具转换成AI能理解的API格式
                )
            ):
                # 如果是文本增量 → 直接流式返回
                if isinstance(event, ApiTextDeltaEvent):
                    yield AssistantTextDelta(text=event.text), None
                    continue

                # 如果是重试事件 → 提示用户
                if isinstance(event, ApiRetryEvent):
                    yield StatusEvent(
                        message=(
                            f"Request failed; retrying in {event.delay_seconds:.1f}s "
                            f"(attempt {event.attempt + 1} of {event.max_attempts}): {event.message}"
                        )
                    ), None
                    continue
                """stream_message结果：final_message 是 ReAct 的思考结果:里面有 AI 回复的文本;  关键：里面有 tool_uses 列表（要调用的工具）"""
                if isinstance(event, ApiMessageCompleteEvent):
                    final_message = event.message #final_message表示一条完整对话消息（用户 / 助手），可包含文本、图片、工具调用
                    usage = event.usage

        # ====================== 异常处理 ======================
        except Exception as exc:
            error_msg = str(exc)
            # 情况1：max_tokens太大被模型拒绝 → 自动降低
            if _is_completion_token_limit_error(exc):
                supported_limit = _extract_completion_token_limit(exc)
                if supported_limit is not None and effective_max_tokens > supported_limit:
                    previous_max_tokens = effective_max_tokens
                    effective_max_tokens = supported_limit
                    yield StatusEvent(
                        message=(
                            f"Model rejected max_tokens={previous_max_tokens}; "
                            f"retrying with provider limit {effective_max_tokens}."
                        )
                    ), None
                    turn_count = max(0, turn_count - 1) # 回退一轮，重试
                    continue

            # 情况2：提示词太长 → 强制压缩对话
            if not reactive_compact_attempted and _is_prompt_too_long_error(exc):
                reactive_compact_attempted = True
                yield StatusEvent(message=REACTIVE_COMPACT_STATUS_MESSAGE), None
                # 强制压缩
                async for event, usage in _stream_compaction(trigger="reactive", force=True):
                    yield event, usage
                compacted_messages, was_compacted = last_compaction_result
                if compacted_messages is not messages:
                    messages[:] = compacted_messages
                if was_compacted:
                    continue # 压缩成功 → 重试本轮

            # 情况3：网络错误
            if "connect" in error_msg.lower() or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                yield ErrorEvent(message=f"Network error: {error_msg}. Check your internet connection and try again."), None
            else:
                yield ErrorEvent(message=f"API error: {error_msg}"), None
            return # 出错退出

        # 如果模型没有返回任何消息 → 报错
        if final_message is None: #final_message表示一条完整对话消息（用户 / 助手），可包含文本、图片、工具调用
            raise RuntimeError("Model stream finished without a final message")

        # ====================== 特殊处理：协调员模式 ======================
        coordinator_context_message: ConversationMessage | None = None
        """coordinator1. 识别协调器模式: 这是框架判断 “我现在是中央大脑，不是普通助手” 的开关"""
        if context.system_prompt.startswith("You are a **coordinator**."):
            if messages and messages[-1].role == "user" and messages[-1].text.startswith("# Coordinator User Context"):
                coordinator_context_message = messages.pop()

        # 如果AI返回空消息 → 丢弃并警告
        if final_message.role == "assistant" and final_message.is_effectively_empty():
            log.warning("dropping empty assistant message from provider response")
            yield ErrorEvent(
                message=(
                    "Model returned an empty assistant message. "
                    "The turn was ignored to keep the session healthy."
                )
            ), usage
            return

        # ====================== 把AI回复加入对话历史 ======================
        messages.append(final_message)
        #messages对话历史，final_message表示一条完整对话消息（用户 / 助手）     发送事件：助手本轮完成
        yield AssistantTurnComplete(message=final_message, usage=usage), usage

        #  恢复协调员上下文消息
        """Coordinator 不直接干活，它只做三件事：理解用户总任务、拆分成小任务、分配给对应技能 / 子代理执行"""
        if coordinator_context_message is not None:
            messages.append(coordinator_context_message)

        # ====================== 如果没有工具调用 → 结束循环 ======================
        """第二【ReAct = Act】模型决定是否调用工具, 
         如果有 tool_uses → 进入 Act 阶段, 这是 ReAct 最关键的分支判断。
        ！hook_executor的作用是： 在 AI 结束对话、不再调用任何工具、准备退出之前，通知所有插件：“对话结束了”，让插件完成收尾工作。
        必须在 return 之前最后通知一次插件，"""
        if not final_message.tool_uses: # 没有工具 → 结束对话
            if context.hook_executor is not None:
                """事件 4：对话结束（AI 停止调用工具）:触发各类系统生命周期事件，由 HookExecutor 调度执行各类钩子逻辑。"""
       #hook_executor是事件通知器：它不是在执行逻辑，它是在发通知。通知内容：对话已停止（STOP），原因：AI 没有调用工具。
                await context.hook_executor.execute(
                    HookEvent.STOP,
                    {
                        "event": HookEvent.STOP.value,
                        "stop_reason": "tool_uses_empty",
                    },
                )
            return # 退出，对话结束

        # ====================== 开始执行工具 ======================
        tool_calls = final_message.tool_uses # AI要调用的工具列表，里面包含 send_message 调用

        # 情况1：只调用一个工具 → 顺序执行
        if len(tool_calls) == 1:
            # Single tool: sequential (stream events immediately)
            tc = tool_calls[0]
            # 发送事件：工具开始执行
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None
            """第三、如果有工具 → 进入 Act 阶段，
            AI 说 “我要调用工具” → 框架替 AI 执行 → 拿到结果   【ReAct = Act】真正执行工具 
            是 ReAct 的 A：Act（行动 / 执行）框架替模型去做：读文件、写代码、执行命令、查资料等。"""
            result = await _execute_tool_call(context, tc.name, tc.id, tc.input)

            # 发送事件：工具执行完成
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
                metadata=result.result_metadata,
            ), None
            tool_results = [result]

        else:# 情况2：多个工具 → 并发执行
            # 先全部发送“开始执行”事件  Multiple tools: execute concurrently, emit events after
            for tc in tool_calls:
                yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None

            # 定义并发执行函数
            async def _run(tc):
                return await _execute_tool_call(context, tc.name, tc.id, tc.input)

            # Use return_exceptions=True so a single failing tool does not abandon
            # its siblings as cancelled coroutines and leave the conversation with
            # un-replied tool_use blocks (Anthropic's API rejects the next request
            # on the session if any tool_use is missing a matching tool_result).
            # 并发执行所有工具，出错不影响其他工具
            """第三、【ReAct = Act】真正执行工具: 
            是 ReAct 的 A：Act（行动 / 执行）,框架替模型去做：读文件、写代码、执行命令、查资料等。"""
            raw_results = await asyncio.gather(
                *[_run(tc) for tc in tool_calls], return_exceptions=True
            )
            tool_results = []
            # 处理每个工具结果
            for tc, result in zip(tool_calls, raw_results):
                if isinstance(result, BaseException):
                    log.exception(
                        "tool execution raised: name=%s id=%s",
                        tc.name,
                        tc.id,
                        exc_info=result,
                    )
                    # AI 工具调用流程的「收尾关键段」，负责处理工具执行失败、返回结果、更新对话历史。  包装成错误结果
                    result = ToolResultBlock(
                        tool_use_id=tc.id, # 绑定工具调用ID（必须对应，模型才能识别）
                        content=f"Tool {tc.name} failed: {type(result).__name__}: {result}",
                        is_error=True,
                    )
                    # 全部执行完后，发送完成事件
                tool_results.append(result)

            #遍历所有工具调用，逐个向外发送「工具执行完成」事件。
            for tc, result in zip(tool_calls, tool_results):
                yield ToolExecutionCompleted(
                    tool_name=tc.name, # 工具名
                    output=result.content, # 工具输出（成功内容/失败提示）
                    is_error=result.is_error, # 是否失败
                    metadata=result.result_metadata, # 附带元信息
                ), None

        #把工具执行结果（成功 / 失败）塞回对话历史，并且假装是 “用户说的话”
        """第四、这是 ReAct 的 O：Observe（观察）:  工具结果塞回对话 → 模型能看到结果 → 进入下一轮思考。"""
        messages.append(ConversationMessage(role="user", content=tool_results))

    #如果设定了最大轮数（比如 200 轮），并且跑满了，抛出错误，防止 AI 无限循环。
    if context.max_turns is not None:
        raise MaxTurnsExceeded(context.max_turns)
    raise RuntimeError("Query loop exited without a max_turns limit or final response")

"""！！！负责真正调用并执行 AI 指定的所有工具：：接收 AI 要调用的工具 → 校验权限 → 执行工具 → 返回结果给 AI

 _execute_tool_call 就是 AI 工具调用的真正执行者，流程一共 6 步：
执行前插件检查
查找工具是否存在
校验工具参数格式
权限检查（安全核心）
危险操作要问用户允许吗
真正执行工具
处理结果、记录历史、返回给 AI

Agent 所有 “动作” 的唯一执行入口所有工具调用、权限、安全、钩子、真正执行，全在这里。
它控制：
工具是否存在
参数校验
权限检查（安全核心）
危险操作确认
真正执行 tool.execute ()
结果封装
错误处理

_execute_tool_call = AI 的 “工具执行管家”
流程就像：
AI 说：我要用 bash 命令
管家先问插件：能让它用吗？
管家检查权限：这个命令危不危险？
危险 → 问用户：允许吗？
允许 → 管家真正去执行命令
执行完 → 告诉插件执行完毕
把结果带回给 AI
 """
async def _execute_tool_call(
    context: QueryContext,# 本次对话的全部上下文（配置、模型、权限、状态）
    tool_name: str, # 要执行的工具名称（如 bash, read_file, web_search）
    tool_use_id: str, # 工具调用的唯一ID（用于和AI的请求对应）
    tool_input: dict[str, object], # 工具参数（AI传过来的）
) -> ToolResultBlock: # 返回值：工具执行结果块

    """1、执行前插件检查（PRE_TOOL_USE） 插件可以在这里拦截、记录、修改工具调用; 插件说 “阻止” → 直接不执行"""
    if context.hook_executor is not None:
        """工具执行前:触发各类系统生命周期事件，由 HookExecutor 调度执行各类钩子逻辑。"""
        pre_hooks = await context.hook_executor.execute(
            HookEvent.PRE_TOOL_USE, # 事件类型：使用工具之前
            {"tool_name": tool_name, "tool_input": tool_input, "event": HookEvent.PRE_TOOL_USE.value},
        )

        # 如果钩子阻止了本次执行 → 直接返回错误
        if pre_hooks.blocked:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre_hooks.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )
    # 日志：记录工具开始执行
    log.debug("tool_call start: %s id=%s", tool_name, tool_use_id)


    """2、查找工具是否存在,从MCP工具注册表中获取这个工具（找不到就是无效工具）"""
    tool = context.tool_registry.get(tool_name)

    # 如果工具不存在 → 返回错误
    if tool is None:
        log.warning("unknown tool: %s", tool_name)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    """！！3、校验工具参数格式
    验证AI传过来的工具参数是否符合工具要求（格式校验） tool_input需关注"""
    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    # 参数格式错误 → 返回错误
    except Exception as exc:
        log.warning("invalid input for %s: %s", tool_name, exc)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Invalid input for {tool_name}: {exc}",
            is_error=True,
        )

    # Normalize common tool inputs before permission checks so path rules apply
    # consistently across built-in tools that use `file_path`, `path`, or
    # directory-scoped roots such as `glob`/`grep`.

    """!!!4、权限检查（最核心的安全机制）---解析工具要操作的文件路径（统一格式）
    _resolve_permission_file_path是解析工具调用中的文件路径，用于权限校验"""
    _file_path = _resolve_permission_file_path(context.cwd, tool_input, parsed_input)

    # 解析工具要执行的命令（如 bash 命令） _extract_permission_command提取工具调用中的执行命令，用于权限校验
    _command = _extract_permission_command(tool_input, parsed_input)
    # 日志：打印权限检查信息
    log.debug("permission check: %s read_only=%s path=%s cmd=%s",
              tool_name, tool.is_read_only(parsed_input), _file_path, _command and _command[:80])

    """!!!安全权限检查（最关键） 检查 AI 是否有权限执行这个操作；    危险操作会弹窗问用户是否允许；不允许 → 直接拒绝执行"""
    decision = context.permission_checker.evaluate(
        tool_name,
        is_read_only=tool.is_read_only(parsed_input),
        file_path=_file_path,
        command=_command,
    )
    # 如果权限不通过
    if not decision.allowed:
        # 如果需要用户确认（比如执行危险命令）
        if decision.requires_confirmation and context.permission_prompt is not None:
            log.debug("permission prompt for %s: %s", tool_name, decision.reason)
            # 发送通知事件
            if context.hook_executor is not None:
                await context.hook_executor.execute(
                    HookEvent.NOTIFICATION,
                    {
                        "event": HookEvent.NOTIFICATION.value,
                        "notification_type": "permission_prompt",
                        "tool_name": tool_name,
                        "reason": decision.reason,
                    },
                )
            """步骤 5：危险操作 → 问用户允许吗      弹出确认框问用户：是否允许执行？"""
            confirmed = await context.permission_prompt(tool_name, decision.reason)
            # 用户拒绝 → 返回权限错误
            if not confirmed:
                log.debug("permission denied by user for %s", tool_name)
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=decision.reason or f"Permission denied for {tool_name}",
                    is_error=True,
                )
        # 不需要确认、直接禁止
        else:
            log.debug("permission blocked for %s: %s", tool_name, decision.reason)
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=decision.reason or f"Permission denied for {tool_name}",
                is_error=True,
            )
    """日志：开始执行工具"""
    log.debug("executing %s ...", tool_name)
    t0 = time.monotonic()  # 记录开始时间

    """!!!6、执行工具（真正干活的地方） :唯一真正和系统交互的地方 """
    result = await tool.execute(
        parsed_input, # 校验后的 :AI传过来的工具参数
        ToolExecutionContext( # 工具执行上下文
            cwd=context.cwd,
            metadata={
                "tool_registry": context.tool_registry,
                "ask_user_prompt": context.ask_user_prompt,
                **(context.tool_metadata or {}),
            },
            hook_executor=context.hook_executor,
        ),
    )
    elapsed = time.monotonic() - t0 # 计算执行耗时
    # 日志：打印执行结果
    log.debug("executed %s in %.2fs err=%s output_len=%d",
              tool_name, elapsed, result.is_error, len(result.output or ""))

    """处理超长输出（自动截断 + 保存文件）   如果工具输出太长 → 截断并保存到文件"""
    inline_output, artifact_path = _offload_tool_output_if_needed(
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        output=result.output,
    )
    # 如果保存成文件了 → 记录这个文件
    if artifact_path is not None:
        _remember_active_artifact(context.tool_metadata, str(artifact_path))

    """构造最终返回给 AI 的结果"""
    tool_result = ToolResultBlock(
        tool_use_id=tool_use_id, # 对应AI的调用ID
        content=inline_output, # 返回给AI看的内容（短文本）
        is_error=result.is_error, # 是否出错
        result_metadata=dict(result.metadata or {}),
    )

    """记录工具执行历史（给 AI 做记忆用）--记录本次工具执行到任务状态（AI的长期记忆）。 
    可能包含send_message 调用"""
    _record_tool_carryover(
        context,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_result.content,
        tool_result_metadata=result.metadata,
        is_error=tool_result.is_error,
        resolved_file_path=_file_path,
    )

    """!!!执行工具后插件通知（POST_TOOL_USE）   告诉插件：工具执行完了;插件可以记录日志、统计、监控
    
    工具执行后:后续触发各类系统生命周期事件，由 HookExecutor 调度执行各类钩子逻辑。"""
    if context.hook_executor is not None:
        await context.hook_executor.execute(
            HookEvent.POST_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result.content,
                "tool_is_error": tool_result.is_error,
                "event": HookEvent.POST_TOOL_USE.value,
            },
        )
    """返回结果给 AI"""
    return tool_result

"""解析工具调用中的文件路径，用于权限校验    """
def _resolve_permission_file_path(
    cwd: Path,
    raw_input: dict[str, object],
    parsed_input: object,
) -> str | None:
    for key in ("file_path", "path", "root"):
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = cwd / path
            return str(path.resolve())

    for attr in ("file_path", "path", "root"):
        value = getattr(parsed_input, attr, None)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = cwd / path
            return str(path.resolve())

    return None

"""提取工具调用中的执行命令，用于权限校验   """
def _extract_permission_command(
    raw_input: dict[str, object],
    parsed_input: object,
) -> str | None:
    value = raw_input.get("command")
    if isinstance(value, str) and value.strip():
        return value

    value = getattr(parsed_input, "command", None)
    if isinstance(value, str) and value.strip():
        return value

    return None
