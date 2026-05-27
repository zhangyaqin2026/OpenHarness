"""Higher-level system prompt assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openharness.config.paths import (
    get_project_active_repo_context_path,
    get_project_issue_file,
    get_project_pr_comments_file,
)
from openharness.config.settings import Settings
from openharness.coordinator.coordinator_mode import get_coordinator_system_prompt, is_coordinator_mode
from openharness.memory import load_memory_prompt
from openharness.memory.relevance import format_relevant_memories, select_relevant_memories
from openharness.memory.usage import mark_memory_used
from openharness.personalization.rules import load_local_rules
from openharness.permissions.modes import PermissionMode
from openharness.prompts.claudemd import load_claude_md_prompt
from openharness.prompts.system_prompt import build_system_prompt
from openharness.skills.loader import load_skill_registry

"""OpenHarness AI 助手的系统提示词组装工具，负责把基础规则、技能、任务、环境、记忆等内容拼接成完整指令发给大模型。
核心就是拼装最终给 AI 看的完整指令。

build_runtime_system_prompt = 最终生成 AI 指令的总入口
_build_skills_section = 告诉 AI 自己有什么技能
_build_delegation_section = 告诉 AI 怎么用子任务
整体功能：拼装给 AI 看的完整系统指令"""

"""生成 AI 可用的技能列表，告诉 AI 能调用哪些工具、怎么用。重点：让 AI 知道自己有什么能力。"""
def _build_skills_section(
    cwd: str | Path,
    *,
    extra_skill_dirs: Iterable[str | Path] | None = None,
    extra_plugin_roots: Iterable[str | Path] | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Build a system prompt section listing available skills."""
    registry = load_skill_registry(
        cwd,
        extra_skill_dirs=extra_skill_dirs,
        extra_plugin_roots=extra_plugin_roots,
        settings=settings,
    )
    skills = [skill for skill in registry.list_skills() if not skill.disable_model_invocation]
    if not skills:
        return None
    lines = [
        "# Available Skills",
        "",
        "The following skills are available via the `skill` tool. "
        "When a user's request matches a skill, invoke it with `skill(name=\"<skill_name>\")` "
        "to load detailed instructions before proceeding. "
        "User-invocable skills can also be run directly by the user as `/<skill-name>`.",
        "",
    ]
    for skill in skills:
        command_name = skill.command_name or skill.name
        display = f" ({skill.display_name})" if skill.display_name else ""
        lines.append(f"- **{command_name}**{display}: {skill.description}")
    return "\n".join(lines)

"""生成子任务 / 子代理说明，告诉 AI 什么时候可以拆分任务、用子 agent。"""
def _build_delegation_section() -> str:
    """Build a concise section describing delegation and worker usage."""
    return "\n".join(
        [
            "# Delegation And Subagents",
            "",
            "OpenHarness can delegate background work with the `agent` tool.",
            "Use it when the user explicitly asks for a subagent, background worker, or parallel investigation, "
            "or when the task clearly benefits from splitting off a focused worker.",
            "",
            "Default pattern:",
            '- Spawn with `agent(description=..., prompt=..., subagent_type=\"worker\")`.',
            "- Inspect running or recorded workers with `/agents`.",
            "- Inspect one worker in detail with `/agents show TASK_ID`.",
            "- Send follow-up instructions with `send_message(task_id=..., message=...)`.",
            "- Read worker output with `task_output(task_id=...)`.",
            "",
            "Prefer a normal direct answer for simple tasks. Use subagents only when they materially help.",
        ]
    )


def _build_permission_mode_section(settings: Settings) -> str:
    """Build current permission-mode guidance for the model."""
    mode = settings.permission.mode
    if mode == PermissionMode.PLAN:
        guidance = (
            "Plan mode is enabled. Treat this session as read-only planning and analysis. "
            "Do not call mutating tools such as file writes, edits, package installs, "
            "state-changing shell commands, or task-spawning actions unless the user exits plan mode."
        )
    elif mode == PermissionMode.FULL_AUTO:
        guidance = (
            "Full-auto permission mode is enabled. You may use mutating tools when they are necessary "
            "for the user's request, while still keeping changes scoped and intentional."
        )
    else:
        guidance = (
            "Default permission mode is enabled. Read-only tools can run directly; mutating tools "
            "may require explicit user approval."
        )
    return f"# Current Permission Mode\n{guidance}"


"""!!!把所有提示词片段拼成最终完整指令。包括：基础规则 + 技能 + 环境 + 项目信息 + 记忆 + 任务配置。整个文件的灵魂。"""
def build_runtime_system_prompt(
    settings: Settings,
    *,
    cwd: str | Path,
    latest_user_prompt: str | None = None,
    extra_skill_dirs: Iterable[str | Path] | None = None,
    extra_plugin_roots: Iterable[str | Path] | None = None,
    include_project_memory: bool = True,
) -> str:
    """Build the runtime system prompt with project instructions and memory."""
    """如果是协调模式，用协调者专用提示词;  否则用普通系统提示词（支持自定义）"""
    if is_coordinator_mode():
        sections = [get_coordinator_system_prompt()]
    else:
        sections = [build_system_prompt(custom_prompt=settings.system_prompt, cwd=str(cwd))]
    """非协调模式且没自定义提示词，重新生成默认提示词覆盖。"""
    if not is_coordinator_mode() and settings.system_prompt is None:
        sections[0] = build_system_prompt(cwd=str(cwd))

    sections.append(_build_permission_mode_section(settings))

    """开启快速模式，添加提示：简洁回答、少用工具、快速完成任务。"""
    if settings.fast_mode:
        sections.append(
            "# Session Mode\nFast mode is enabled. Prefer concise replies, minimal tool use, and quicker progress over exhaustive exploration."
        )
    """添加推理配置，告诉 AI 努力程度、迭代次数，让 AI 按此执行。"""
    sections.append(
        "# Reasoning Settings\n"
        f"- Effort: {settings.effort}\n"
        f"- Passes: {settings.passes}\n"
        "Adjust depth and iteration count to match these settings while still completing the task."
    )
    """生成技能列表，非协调模式就加到提示词里。"""
    skills_section = _build_skills_section(
        cwd,
        extra_skill_dirs=extra_skill_dirs,
        extra_plugin_roots=extra_plugin_roots,
        settings=settings,
    )
    if skills_section and not is_coordinator_mode():
        sections.append(skills_section)
    """非协调模式，添加任务委托规则。"""
    if not is_coordinator_mode():
        sections.append(_build_delegation_section())
    """加载项目目录里的 claude.md 自定义提示，有就加入。"""
    claude_md = load_claude_md_prompt(cwd)
    if claude_md:
        sections.append(claude_md)
    """加载本地环境规则，有就加入提示词。"""
    local_rules = load_local_rules()
    if local_rules:
        sections.append(f"# Local Environment Rules\n\n{local_rules}")

    """循环加载项目问题、PR 评论、仓库上下文文件，存在且有内容就加入提示词（最多 12000 字符）。"""
    for title, path in (
        ("Issue Context", get_project_issue_file(cwd)),
        ("Pull Request Comments", get_project_pr_comments_file(cwd)),
        ("Active Repo Context", get_project_active_repo_context_path(cwd)),
    ):
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sections.append(f"# {title}\n\n```md\n{content[:12000]}\n```")

    """开启记忆功能，加载项目记忆并加入提示词。"""
    if include_project_memory and settings.memory.enabled:
        memory_section = load_memory_prompt(
            cwd,
            max_entrypoint_lines=settings.memory.max_entrypoint_lines,
            max_entrypoint_bytes=settings.memory.max_entrypoint_bytes,
        )
        if memory_section:
            sections.append(memory_section)
        """根据用户最新输入，搜索相关记忆，读取内容并加入提示词，让 AI 知道历史相关信息。"""
        if latest_user_prompt:
            relevant = select_relevant_memories(
                latest_user_prompt,
                cwd,
                max_results=settings.memory.max_files,
            )
            if relevant:
                try:
                    headers = [item.header for item in relevant]
                    mark_memory_used(cwd, headers, memory_dir=headers[0].path.parent)
                except OSError:
                    pass
                sections.append(format_relevant_memories(relevant))

    return "\n\n".join(section for section in sections if section.strip())
