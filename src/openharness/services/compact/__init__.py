"""Conversation compaction — microcompact and full LLM-based summarization.

Faithfully translated from Claude Code's compaction system:
- Microcompact: clear old tool result content to reduce token count cheaply
- Full compact: call the LLM to produce a structured summary of older messages
- Auto-compact: trigger compaction automatically when token count exceeds threshold
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4

from openharness.engine.messages import (
    ConversationMessage,
    ContentBlock,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    sanitize_conversation_messages,
)
from openharness.engine.stream_events import CompactProgressEvent
from openharness.hooks import HookEvent, HookExecutor
from openharness.services.tool_outputs import is_microcompactable_tool_result
from openharness.services.token_estimation import estimate_tokens

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from Claude Code microCompact.ts / autoCompact.ts)
# ---------------------------------------------------------------------------

COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "bash",
    "grep",
    "glob",
    "web_search",
    "web_fetch",
    "edit_file",
    "write_file",
})

TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"

# Auto-compact thresholds
AUTOCOMPACT_BUFFER_TOKENS = 13_000
MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
COMPACT_TIMEOUT_SECONDS = 25
MAX_COMPACT_STREAMING_RETRIES = 2
MAX_PTL_RETRIES = 3
SESSION_MEMORY_KEEP_RECENT = 12
SESSION_MEMORY_MAX_LINES = 48
SESSION_MEMORY_MAX_CHARS = 4_000
CONTEXT_COLLAPSE_TEXT_CHAR_LIMIT = 2_400
CONTEXT_COLLAPSE_HEAD_CHARS = 900
CONTEXT_COLLAPSE_TAIL_CHARS = 500
MAX_COMPACT_ATTACHMENTS = 6
MAX_DISCOVERED_TOOLS = 12

# Microcompact defaults
DEFAULT_KEEP_RECENT = 5
DEFAULT_GAP_THRESHOLD_MINUTES = 60

# Token estimation padding (conservative)
TOKEN_ESTIMATION_PADDING = 4 / 3
_DEFAULT_VISION_IMAGE_TOKEN_ESTIMATE = 3_072

# Default context windows per model family
_DEFAULT_CONTEXT_WINDOW = 200_000
PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"
ERROR_MESSAGE_INCOMPLETE_RESPONSE = "Compaction interrupted before a complete summary was returned."

CompactTrigger = Literal["auto", "manual", "reactive"]
CompactProgressCallback = Callable[[CompactProgressEvent], Awaitable[None]]
CompactionKind = Literal["full", "session_memory"]


@dataclass
class CompactAttachment:
    """Structured compact asset carried across a compaction boundary."""

    kind: str
    title: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactionResult:
    """Structured compaction result, inspired by Claude Code's result shape."""

    trigger: CompactTrigger
    compact_kind: CompactionKind
    boundary_marker: ConversationMessage
    summary_messages: list[ConversationMessage]
    messages_to_keep: list[ConversationMessage]
    attachments: list[CompactAttachment]
    hook_results: list[CompactAttachment]
    compact_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
"""OpenHarness的对话token估算函数，用于统计消息列表的总token数。
先初始化计数器，获取单张图片的预估token，遍历所有消息和内容块，分别计算文本、工具结果、工具调用、图片的token并累加，
最后乘以4/3的安全冗余系数，返回整数结果。核心作用是精准估算对话占用token，为上下文压缩提供判断依据。 """
def estimate_message_tokens(messages: list[ConversationMessage]) -> int:
# 定义一个名为estimate_message_tokens的函数，参数是对话消息列表，返回值为整数（token总数）
    """Estimate total tokens for a conversation, including the 4/3 padding."""
    # 函数文档：估算对话的总token数，包含4/3的安全冗余系数
    total = 0
    # 初始化总token计数器，初始值为0
    image_token_estimate = _vision_token_budget_per_image()
    # 调用函数，获取单张图片消耗的预估token数，赋值给变量
    for msg in messages:
    # 遍历传入的每一条对话消息
        for block in msg.content:
        # 遍历单条消息中的每一个内容块（文本、图片、工具等）
            if isinstance(block, TextBlock):
            # 判断当前内容块是否为纯文本块
                total += estimate_tokens(block.text)
                # 计算文本的token数，累加到总token数
            elif isinstance(block, ToolResultBlock):
            # 判断当前内容块是否为工具执行结果块
                total += estimate_tokens(block.content)
                # 计算工具结果内容的token数，累加到总token数
            elif isinstance(block, ToolUseBlock):
            # 判断当前内容块是否为工具调用块
                total += estimate_tokens(block.name)
                # 计算工具名称的token数，累加到总token数
                total += estimate_tokens(str(block.input))
                # 计算工具输入参数的token数，累加到总token数
            elif isinstance(block, ImageBlock):
            # 判断当前内容块是否为图片块
                total += image_token_estimate
                # 把图片的预估token数，累加到总token数
    return int(total * TOKEN_ESTIMATION_PADDING)
# 总token数乘以安全冗余系数，转换为整数后返回

def estimate_conversation_tokens(messages: list[ConversationMessage]) -> int:
    """Alias kept for backward compatibility."""
    return estimate_message_tokens(messages)


def _vision_token_budget_per_image() -> int:
    raw = os.environ.get("OPENHARNESS_IMAGE_TOKEN_ESTIMATE", "").strip()
    if raw:
        try:
            return max(64, int(raw))
        except ValueError:
            log.warning("Ignoring invalid OPENHARNESS_IMAGE_TOKEN_ESTIMATE=%r", raw)
    return _DEFAULT_VISION_IMAGE_TOKEN_ESTIMATE


def _replace_images_with_compaction_placeholders(
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    """Strip image payloads from summarizer-only compact requests."""
    replaced: list[ConversationMessage] = []
    for message in messages:
        next_content: list[ContentBlock] = []
        changed = False
        for block in message.content:
            if isinstance(block, ImageBlock):
                changed = True
                label = block.source_path.strip() or "inline"
                next_content.append(
                    TextBlock(
                        text=f"[Image omitted from compaction summarization; source: {label}.]\n"
                    )
                )
            else:
                next_content.append(block)
        if changed:
            replaced.append(message.model_copy(update={"content": next_content}))
        else:
            replaced.append(message)
    return replaced


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_metadata(item) for item in value]
    return str(value)

"""记录精简版检查点数据：接收元数据、检查点标识、触发条件、消息 / 令牌数量等信息，组装成标准数据格式；
可选添加重试次数、详情数据，还能把新数据追加到历史元数据中并更新最新记录，最后返回组装好的检查点数据。 """
def _record_compact_checkpoint(
    carryover_metadata: dict[str, Any] | None, #可选的历史元数据字典
    *,
    checkpoint: str, #必填的检查点标识字符串
    trigger: CompactTrigger,#必填的检查点触发条件
    message_count: int,#必填的消息数量、令牌数量
    token_count: int,
    attempt: int | None = None,#可选的重试次数、详情字典
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建基础数据字典，存入核心检查点信息"""
    payload: dict[str, Any] = {
        "checkpoint": checkpoint,
        "trigger": trigger,
        "message_count": message_count,
        "token_count": token_count,
    }
    """如果有重试次数，添加到数据中"""
    if attempt is not None:
        payload["attempt"] = attempt
        """如果有详情数据，清洗后合并到数据中"""
    if details:
        payload.update(_sanitize_metadata(details))
        """如果有历史元数据,   checkpoints获取 / 创建历史检查点列表"""
    if carryover_metadata is not None:
        checkpoints = carryover_metadata.setdefault("compact_checkpoints", [])
        if isinstance(checkpoints, list):
            checkpoints.append(payload)#把新数据追加到历史列表
        carryover_metadata["compact_last"] = payload #更新最新检查点记录
    return payload #返回最终组装好的检查点数据


async def _emit_progress(
    callback: CompactProgressCallback | None,
    *,
    phase: Literal[
        "hooks_start",
        "context_collapse_start",
        "context_collapse_end",
        "session_memory_start",
        "session_memory_end",
        "compact_start",
        "compact_retry",
        "compact_end",
        "compact_failed",
    ],
    trigger: CompactTrigger,
    message: str | None = None,
    attempt: int | None = None,
    checkpoint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    await callback(
        CompactProgressEvent(
            phase=phase,
            trigger=trigger,
            message=message,
            attempt=attempt,
            checkpoint=checkpoint,
            metadata=_sanitize_metadata(metadata) if metadata else None,
        )
    )


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


def _group_messages_by_prompt_round(
    messages: list[ConversationMessage],
) -> list[list[ConversationMessage]]:
    groups: list[list[ConversationMessage]] = []
    current: list[ConversationMessage] = []
    for message in messages:
        starts_new_round = (
            message.role == "user"
            and not any(isinstance(block, ToolResultBlock) for block in message.content)
            and bool(message.text.strip())
        )
        if starts_new_round and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return groups


def _collapse_text(text: str) -> str:
    if len(text) <= CONTEXT_COLLAPSE_TEXT_CHAR_LIMIT:
        return text
    omitted = len(text) - CONTEXT_COLLAPSE_HEAD_CHARS - CONTEXT_COLLAPSE_TAIL_CHARS
    head = text[:CONTEXT_COLLAPSE_HEAD_CHARS].rstrip()
    tail = text[-CONTEXT_COLLAPSE_TAIL_CHARS:].lstrip()
    return f"{head}\n...[collapsed {omitted} chars]...\n{tail}"


def try_context_collapse(
    messages: list[ConversationMessage],
    *,
    preserve_recent: int,
) -> list[ConversationMessage] | None:
    """Deterministically shrink oversized text blocks before full compact."""
    if len(messages) <= preserve_recent + 2:
        return None

    older, newer = _split_preserving_tool_pairs(messages, preserve_recent=preserve_recent)
    changed = False
    collapsed_older: list[ConversationMessage] = []
    for message in older:
        new_blocks: list[ContentBlock] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                collapsed = _collapse_text(block.text)
                if collapsed != block.text:
                    changed = True
                new_blocks.append(TextBlock(text=collapsed))
            elif isinstance(block, ToolResultBlock):
                collapsed = _collapse_text(block.content)
                if collapsed != block.content:
                    changed = True
                new_blocks.append(
                    ToolResultBlock(
                        tool_use_id=block.tool_use_id,
                        content=collapsed,
                        is_error=block.is_error,
                    )
                )
            else:
                new_blocks.append(block)
        collapsed_older.append(ConversationMessage(role=message.role, content=new_blocks))

    if not changed:
        return None

    result = [*collapsed_older, *newer]
    if estimate_message_tokens(result) >= estimate_message_tokens(messages):
        return None
    return result


def truncate_head_for_ptl_retry(
    messages: list[ConversationMessage],
) -> list[ConversationMessage] | None:
    """Drop the oldest prompt rounds when the compact request itself is too large."""
    groups = _group_messages_by_prompt_round(messages)
    if len(groups) < 2:
        return None

    drop_count = max(1, len(groups) // 5)
    drop_count = min(drop_count, len(groups) - 1)
    retained = [message for group in groups[drop_count:] for message in group]
    if not retained:
        return None
    if retained[0].role == "assistant":
        return [ConversationMessage.from_user_text(PTL_RETRY_MARKER), *retained]
    return retained


def _extract_attachment_paths(messages: list[ConversationMessage]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    path_pattern = re.compile(r"path:\s*([^)\\n]+)")
    attachment_pattern = re.compile(r"\[attachment:\s*([^\]]+)\]")
    for message in messages:
        for block in message.content:
            if isinstance(block, ImageBlock) and block.source_path:
                path = str(Path(block.source_path).expanduser())
                if path not in seen:
                    seen.add(path)
                    found.append(path)
            elif isinstance(block, TextBlock):
                for match in path_pattern.findall(block.text):
                    path = match.strip()
                    if path and path not in seen:
                        seen.add(path)
                        found.append(path)
                for match in attachment_pattern.findall(block.text):
                    path = match.strip()
                    if path and "download failed" not in path and path not in seen:
                        seen.add(path)
                        found.append(path)
            if len(found) >= MAX_COMPACT_ATTACHMENTS:
                return found
    return found


def _extract_discovered_tools(messages: list[ConversationMessage]) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()
    for message in messages:
        for tool_use in message.tool_uses:
            if tool_use.name and tool_use.name not in seen:
                seen.add(tool_use.name)
                discovered.append(tool_use.name)
            if len(discovered) >= MAX_DISCOVERED_TOOLS:
                return discovered
    return discovered


def _create_attachment(kind: str, title: str, lines: list[str], *, metadata: dict[str, Any] | None = None) -> CompactAttachment | None:
    filtered = [line.rstrip() for line in lines if line and line.strip()]
    if not filtered:
        return None
    return CompactAttachment(
        kind=kind,
        title=title,
        body="\n".join(filtered),
        metadata=_sanitize_metadata(metadata or {}),
    )


def render_compact_attachment(attachment: CompactAttachment) -> ConversationMessage:
    """Serialize a structured compact attachment into a conversation message."""
    header = f"[Compact attachment: {attachment.kind}] {attachment.title}".strip()
    text = f"{header}\n{attachment.body}".strip()
    return ConversationMessage.from_user_text(text)


def create_compact_boundary_message(metadata: dict[str, Any]) -> ConversationMessage:
    """Create a boundary marker message for post-compact conversation rebuild."""
    lines = [
        "[Compact boundary marker]",
        "Earlier conversation was compacted. Use the summary and preserved assets below as the continuity boundary.",
    ]
    trigger = str(metadata.get("trigger") or "").strip()
    compact_kind = str(metadata.get("compact_kind") or "").strip()
    pre_messages = metadata.get("pre_compact_message_count")
    pre_tokens = metadata.get("pre_compact_token_count")
    post_messages = metadata.get("post_compact_message_count")
    post_tokens = metadata.get("post_compact_token_count")
    if trigger:
        lines.append(f"Trigger: {trigger}")
    if compact_kind:
        lines.append(f"Compaction kind: {compact_kind}")
    if pre_messages is not None or pre_tokens is not None:
        lines.append(
            "Pre-compact footprint: "
            f"messages={pre_messages if pre_messages is not None else 'unknown'}, "
            f"tokens={pre_tokens if pre_tokens is not None else 'unknown'}"
        )
    if post_messages is not None or post_tokens is not None:
        lines.append(
            "Post-compact footprint: "
            f"messages={post_messages if post_messages is not None else 'unknown'}, "
            f"tokens={post_tokens if post_tokens is not None else 'unknown'}"
        )
    anchor = str(metadata.get("preserved_segment_anchor") or "").strip()
    if anchor:
        lines.append(f"Preserved segment anchor: {anchor}")
    return ConversationMessage.from_user_text("\n".join(lines))

"""按固定顺序组装压缩后的对话消息，整合边界标记、摘要、保留消息、附件与钩子消息，返回最终可用列表。"""
# 定义函数，传入压缩结果对象，返回整理后的消息列表
def build_post_compact_messages(result: CompactionResult) -> list[ConversationMessage]:
    # 函数说明：按指定顺序重建压缩后的消息列表
    """Rebuild the post-compact message list in Claude Code's ordering."""
    # 处理结果中的附件，生成附件消息
    attachment_messages = [render_compact_attachment(attachment) for attachment in result.attachments]
    # 处理结果中的钩子结果，生成对应的消息
    hook_messages = [render_compact_attachment(attachment) for attachment in result.hook_results]
    # 按固定顺序拼接所有消息并返回
    return [
        result.boundary_marker,
        *result.summary_messages,
        *result.messages_to_keep,
        *attachment_messages,
        *hook_messages,
    ]

def _boundary_crosses_tool_pair(previous: ConversationMessage, current: ConversationMessage) -> bool:
    """Return True when a preserve boundary would split a tool_use/result pair."""

    if previous.role != "assistant" or current.role != "user":
        return False
    pending_tool_ids = {block.id for block in previous.content if isinstance(block, ToolUseBlock)}
    if not pending_tool_ids:
        return False
    result_ids = {block.tool_use_id for block in current.content if isinstance(block, ToolResultBlock)}
    return bool(pending_tool_ids & result_ids)


def _split_preserving_tool_pairs(
    messages: list[ConversationMessage],
    *,
    preserve_recent: int,
) -> tuple[list[ConversationMessage], list[ConversationMessage]]:
    """Split older/newer segments without cutting through a tool_use/result pair.

    The preserved segment is also sanitized so trailing orphan tool_use blocks
    never survive the compaction boundary.
    """

    if len(messages) <= preserve_recent:
        return [], sanitize_conversation_messages(list(messages))

    split_index = max(0, len(messages) - preserve_recent)
    while split_index > 0 and _boundary_crosses_tool_pair(messages[split_index - 1], messages[split_index]):
        split_index -= 1

    older = list(messages[:split_index])
    newer = sanitize_conversation_messages(list(messages[split_index:]))
    return older, newer


def _sanitize_compaction_segments(result: CompactionResult) -> None:
    """Normalize summary+preserved messages into a provider-safe sequence."""

    if not result.summary_messages and not result.messages_to_keep:
        return
    combined = [*result.summary_messages, *result.messages_to_keep]
    sanitized = sanitize_conversation_messages(combined)
    summary_count = len(result.summary_messages)
    result.summary_messages = sanitized[:summary_count]
    result.messages_to_keep = sanitized[summary_count:]


def _create_recent_attachments_attachment_if_needed(
    attachment_paths: list[str],
) -> CompactAttachment | None:
    if not attachment_paths:
        return None
    return _create_attachment(
        "recent_attachments",
        "Recent local attachments",
        ["Keep these local attachment paths in working memory:"] + [f"- {path}" for path in attachment_paths],
        metadata={"paths": attachment_paths},
    )


def create_recent_files_attachment_if_needed(
    read_file_state: Any,
) -> CompactAttachment | None:
    if not isinstance(read_file_state, list) or not read_file_state:
        return None
    lines = ["Recently read files that may still matter:"]
    entries: list[dict[str, Any]] = []
    normalized_entries = [
        entry
        for entry in read_file_state
        if isinstance(entry, dict) and str(entry.get("path") or "").strip()
    ]
    normalized_entries.sort(
        key=lambda entry: float(entry.get("timestamp") or 0.0),
        reverse=True,
    )
    for entry in normalized_entries[:4]:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        span = str(entry.get("span") or "").strip()
        preview = str(entry.get("preview") or "").strip()
        timestamp = entry.get("timestamp")
        if not path:
            continue
        bullet = f"- {path}"
        if span:
            bullet += f" ({span})"
        lines.append(bullet)
        if preview:
            lines.append(f"  Preview: {preview}")
        entries.append({"path": path, "span": span, "preview": preview, "timestamp": timestamp})
    return _create_attachment("recent_files", "Recently read files", lines, metadata={"entries": entries})


def create_task_focus_attachment_if_needed(
    metadata: dict[str, Any],
) -> CompactAttachment | None:
    state = metadata.get("task_focus_state")
    if not isinstance(state, dict):
        return None
    goal = str(state.get("goal") or "").strip()
    recent_goals = [
        str(item).strip()
        for item in state.get("recent_goals", [])
        if str(item).strip()
    ]
    active_artifacts = [
        str(item).strip()
        for item in state.get("active_artifacts", [])
        if str(item).strip()
    ]
    verified_state = [
        str(item).strip()
        for item in state.get("verified_state", [])
        if str(item).strip()
    ]
    next_step = str(state.get("next_step") or "").strip()
    if not any((goal, recent_goals, active_artifacts, verified_state, next_step)):
        return None
    lines = ["Current working focus to preserve across compaction:"]
    if goal:
        lines.append(f"- Goal: {goal}")
    if recent_goals:
        lines.append("- Recent user goals that still matter:")
        lines.extend(f"  - {item}" for item in recent_goals[-3:])
    if active_artifacts:
        lines.append("- Active artifacts in play:")
        lines.extend(f"  - {item}" for item in active_artifacts[-5:])
    if verified_state:
        lines.append("- Verified state already established:")
        lines.extend(f"  - {item}" for item in verified_state[-4:])
    if next_step:
        lines.append(f"- Suggested next step: {next_step}")
    return _create_attachment(
        "task_focus",
        "Current working focus",
        lines,
        metadata={
            "goal": goal,
            "recent_goals": recent_goals[-3:],
            "active_artifacts": active_artifacts[-5:],
            "verified_state": verified_state[-4:],
            "next_step": next_step,
        },
    )


def create_recent_verified_work_attachment_if_needed(
    verified_work: Any,
) -> CompactAttachment | None:
    if not isinstance(verified_work, list) or not verified_work:
        return None
    entries = [str(entry).strip() for entry in verified_work[-8:] if str(entry).strip()]
    if not entries:
        return None
    return _create_attachment(
        "recent_verified_work",
        "Recently verified work",
        ["These steps or conclusions were explicitly verified before compaction:"] + [f"- {entry}" for entry in entries],
        metadata={"entries": entries},
    )


def create_plan_attachment_if_needed(metadata: dict[str, Any]) -> CompactAttachment | None:
    permission_mode = str(metadata.get("permission_mode") or "").strip().lower()
    if permission_mode != "plan":
        return None
    lines = [
        "Plan mode is still active for this session.",
        "Do not execute mutating tools until the user explicitly exits plan mode.",
    ]
    plan_summary = str(metadata.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"Current plan summary: {plan_summary}")
    return _create_attachment(
        "plan",
        "Plan mode context",
        lines,
        metadata={"permission_mode": permission_mode, "plan_summary": plan_summary},
    )


def create_invoked_skills_attachment_if_needed(
    invoked_skills: Any,
) -> CompactAttachment | None:
    if not isinstance(invoked_skills, list) or not invoked_skills:
        return None
    normalized = [str(skill).strip() for skill in invoked_skills[-8:] if str(skill).strip()]
    if not normalized:
        return None
    return _create_attachment(
        "invoked_skills",
        "Skills used earlier in the session",
        ["The following skills were invoked and may still shape the next step:", "- " + ", ".join(normalized)],
        metadata={"skills": normalized},
    )


def create_async_agent_attachment_if_needed(
    async_agent_state: Any,
) -> CompactAttachment | None:
    if not isinstance(async_agent_state, list) or not async_agent_state:
        return None
    entries = [str(entry).strip() for entry in async_agent_state[-6:] if str(entry).strip()]
    if not entries:
        return None
    return _create_attachment(
        "async_agents",
        "Async agent and background task state",
        ["Recent async-agent/background-task activity:"] + [f"- {entry}" for entry in entries],
        metadata={"entries": entries},
    )


def create_work_log_attachment_if_needed(
    recent_work_log: Any,
) -> CompactAttachment | None:
    if not isinstance(recent_work_log, list) or not recent_work_log:
        return None
    entries = [str(entry).strip() for entry in recent_work_log[-8:] if str(entry).strip()]
    if not entries:
        return None
    return _create_attachment(
        "recent_work_log",
        "Recent execution checkpoints",
        ["Recent work and verification steps taken in this session:"] + [f"- {entry}" for entry in entries],
        metadata={"entries": entries},
    )


def _create_hook_attachments(hook_note: str | None) -> list[CompactAttachment]:
    if not hook_note or not hook_note.strip():
        return []
    attachment = _create_attachment(
        "hook_results",
        "Compact hook notes",
        [hook_note.strip()],
        metadata={"note": hook_note.strip()},
    )
    return [attachment] if attachment is not None else []


def _build_compact_attachments(
    messages: list[ConversationMessage],
    *,
    metadata: dict[str, Any] | None,
) -> list[CompactAttachment]:
    metadata = metadata or {}
    attachments: list[CompactAttachment] = []
    attachment_paths = _extract_attachment_paths(messages)
    builders = [
        create_task_focus_attachment_if_needed(metadata),
        create_recent_verified_work_attachment_if_needed(metadata.get("recent_verified_work")),
        _create_recent_attachments_attachment_if_needed(attachment_paths),
        create_recent_files_attachment_if_needed(metadata.get("read_file_state")),
        create_plan_attachment_if_needed(metadata),
        create_invoked_skills_attachment_if_needed(metadata.get("invoked_skills")),
        create_async_agent_attachment_if_needed(metadata.get("async_agent_state")),
        create_work_log_attachment_if_needed(metadata.get("recent_work_log")),
    ]
    attachments.extend(attachment for attachment in builders if attachment is not None)
    return attachments


def _finalize_compaction_result(result: CompactionResult) -> CompactionResult:
    _sanitize_compaction_segments(result)
    messages = build_post_compact_messages(result)
    result.compact_metadata.setdefault("post_compact_message_count", len(messages))
    result.compact_metadata.setdefault("post_compact_token_count", estimate_message_tokens(messages))
    result.boundary_marker = create_compact_boundary_message(result.compact_metadata)
    return result


def _metadata_has_checkpoint(metadata: dict[str, Any] | None, checkpoint: str) -> bool:
    if metadata is None:
        return False
    checkpoints = metadata.get("compact_checkpoints")
    if not isinstance(checkpoints, list):
        return False
    return any(isinstance(entry, dict) and entry.get("checkpoint") == checkpoint for entry in checkpoints)


def _build_passthrough_compaction_result(
    messages: list[ConversationMessage],
    *,
    trigger: CompactTrigger,
    compact_kind: CompactionKind,
    metadata: dict[str, Any] | None = None,
) -> CompactionResult:
    compact_metadata = {
        "trigger": trigger,
        "compact_kind": compact_kind,
        "pre_compact_message_count": len(messages),
        "pre_compact_token_count": estimate_message_tokens(messages),
        **_sanitize_metadata(metadata or {}),
    }
    result = CompactionResult(
        trigger=trigger,
        compact_kind=compact_kind,
        boundary_marker=create_compact_boundary_message(compact_metadata),
        summary_messages=[],
        messages_to_keep=list(messages),
        attachments=[],
        hook_results=[],
        compact_metadata=compact_metadata,
    )
    return _finalize_compaction_result(result)


# ---------------------------------------------------------------------------
# Microcompact — clear old tool results to reduce tokens cheaply
# ---------------------------------------------------------------------------

def _collect_compactable_tool_ids(messages: list[ConversationMessage]) -> list[str]:
    """Walk messages and collect tool_use IDs whose results are compactable."""
    ordered_ids: list[str] = []
    tool_names: dict[str, str] = {}
    result_content: dict[str, str] = {}
    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                ordered_ids.append(block.id)
                tool_names[block.id] = block.name
            elif isinstance(block, ToolResultBlock):
                result_content[block.tool_use_id] = block.content
    return [
        tool_id
        for tool_id in ordered_ids
        if tool_names.get(tool_id, "") in COMPACTABLE_TOOLS
        or is_microcompactable_tool_result(
            tool_names.get(tool_id, ""),
            result_content.get(tool_id, ""),
        )
    ]


def microcompact_messages(
    messages: list[ConversationMessage],
    *,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> tuple[list[ConversationMessage], int]:
    """Clear old compactable tool results, keeping the most recent *keep_recent*.

    This is the cheap first pass — no LLM call required. Tool result content
    is replaced with :data:`TIME_BASED_MC_CLEARED_MESSAGE`.

    Returns:
        (messages, tokens_saved) — messages are mutated in place for efficiency.
    """
    keep_recent = max(1, keep_recent)  # never clear ALL results
    all_ids = _collect_compactable_tool_ids(messages)

    if len(all_ids) <= keep_recent:
        return messages, 0

    keep_set = set(all_ids[-keep_recent:])
    clear_set = set(all_ids) - keep_set

    tokens_saved = 0
    for msg in messages:
        if msg.role != "user":
            continue
        new_content: list[ContentBlock] = []
        for block in msg.content:
            if (
                isinstance(block, ToolResultBlock)
                and block.tool_use_id in clear_set
                and block.content != TIME_BASED_MC_CLEARED_MESSAGE
            ):
                tokens_saved += estimate_tokens(block.content)
                new_content.append(
                    ToolResultBlock(
                        tool_use_id=block.tool_use_id,
                        content=TIME_BASED_MC_CLEARED_MESSAGE,
                        is_error=block.is_error,
                    )
                )
            else:
                new_content.append(block)
        msg.content = new_content

    if tokens_saved > 0:
        log.info("Microcompact cleared %d tool results, saved ~%d tokens", len(clear_set), tokens_saved)

    return messages, tokens_saved


def _summarize_message_for_memory(message: ConversationMessage) -> str:
    text = " ".join(message.text.split())
    if text:
        text = text[:160]
        return f"{message.role}: {text}"
    tool_uses = [block.name for block in message.tool_uses]
    if tool_uses:
        return f"{message.role}: tool calls -> {', '.join(tool_uses[:4])}"
    if any(isinstance(block, ToolResultBlock) for block in message.content):
        return f"{message.role}: tool results returned"
    return f"{message.role}: [non-text content]"


def _build_session_memory_message(messages: list[ConversationMessage]) -> ConversationMessage | None:
    lines: list[str] = []
    total_chars = 0
    for message in messages:
        line = _summarize_message_for_memory(message)
        if not line:
            continue
        projected = total_chars + len(line) + 1
        if lines and (len(lines) >= SESSION_MEMORY_MAX_LINES or projected >= SESSION_MEMORY_MAX_CHARS):
            lines.append("... earlier context condensed ...")
            break
        lines.append(line)
        total_chars = projected
    if not lines:
        return None
    body = "\n".join(lines)
    return ConversationMessage.from_user_text(
        "Session memory summary from earlier in this conversation:\n" + body
    )

"""轻量级对话压缩工具，在调用 AI 压缩前使用。先判断消息数量是否满足压缩条件，不满足则直接退出；
将消息分为历史和最新两部分，把历史转为精简记忆摘要。
若压缩后 token 和消息数未减少则退出，否则记录压缩信息，生成压缩结果，返回压缩后数据，高效降低对话长度。"""
def try_session_memory_compaction(
    messages: list[ConversationMessage],
    *,
    preserve_recent: int = SESSION_MEMORY_KEEP_RECENT,
    trigger: CompactTrigger = "auto",
    metadata: dict[str, Any] | None = None,
) -> CompactionResult | None:
# 定义轻量级对话压缩函数，输入消息列表、保留最新消息数、触发类型、元数据，返回压缩结果/空
    """Cheap deterministic compaction for long chats before full LLM compaction."""
# 函数注释：完整AI压缩前，对长对话做轻量确定性压缩
    if len(messages) <= preserve_recent + 4:
        return None
# 消息总数过少，无需压缩，直接返回空
    older, newer = _split_preserving_tool_pairs(messages, preserve_recent=preserve_recent)
# 将消息拆分为历史消息older和需要保留的最新消息newer
    summary_message = _build_session_memory_message(older)
# 把历史消息打包成精简的会话记忆摘要
    if summary_message is None:
        return None
# 摘要生成失败，不压缩，返回空
    provisional = [summary_message, *newer]
# 拼接摘要+最新消息，生成临时压缩后的消息列表
    if (
        estimate_message_tokens(provisional) >= estimate_message_tokens(messages)
        and len(provisional) >= len(messages)
    ):
        return None
# 压缩后token/数量未变少，压缩无效，返回空
    compact_metadata = {
        "trigger": trigger,
        "compact_kind": "session_memory",
        "pre_compact_message_count": len(messages),
        "pre_compact_token_count": estimate_message_tokens(messages),
        "preserve_recent": preserve_recent,
        "used_session_memory": True,
        "pre_compact_discovered_tools": _extract_discovered_tools(older),
        "attachments": _extract_attachment_paths(older),
    }
# 记录压缩相关的元数据
    result = CompactionResult(
        trigger=trigger,
        compact_kind="session_memory",
        boundary_marker=create_compact_boundary_message(compact_metadata),
        summary_messages=[summary_message],
        messages_to_keep=list(newer),
        attachments=_build_compact_attachments(older, metadata=metadata),
        hook_results=[],
        compact_metadata=compact_metadata,
    )
# 构建标准的压缩结果对象
    return _finalize_compaction_result(result)
# 处理并返回最终的有效压缩结果
# ---------------------------------------------------------------------------
# Full compact — LLM-based summarization
# ---------------------------------------------------------------------------

NO_TOOLS_PREAMBLE = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use read_file, bash, grep, glob, edit_file, write_file, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

BASE_COMPACT_PROMPT = """\
Your task is to create a detailed summary of the conversation so far. This summary will replace the earlier messages, so it must capture all important information.

First, draft your analysis inside <analysis> tags. Walk through the conversation chronologically and extract:
- Every user request and intent (explicit and implicit)
- The approach taken and technical decisions made
- Specific code, files, and configurations discussed (with paths and line numbers where available)
- All errors encountered and how they were fixed
- Any user feedback or corrections

Then, produce a structured summary inside <summary> tags with these sections:

1. **Primary Request and Intent**: All user requests in full detail, including nuances and constraints.
2. **Key Technical Concepts**: Technologies, frameworks, patterns, and conventions discussed.
3. **Files and Code Sections**: Every file examined or modified, with specific code snippets and line numbers.
4. **Errors and Fixes**: Every error encountered, its cause, and how it was resolved.
5. **Problem Solving**: Problems solved and approaches that worked vs. didn't work.
6. **All User Messages**: Non-tool-result user messages (preserve exact wording for context).
7. **Pending Tasks**: Explicitly requested work that hasn't been completed yet.
8. **Current Work**: Detailed description of the last task being worked on before compaction.
9. **Optional Next Step**: The single most logical next step, directly aligned with the user's recent request.
"""

NO_TOOLS_TRAILER = """
REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task."""


def get_compact_prompt(custom_instructions: str | None = None) -> str:
    """Build the full compaction prompt sent to the model."""
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    prompt += NO_TOOLS_TRAILER
    return prompt


def format_compact_summary(raw_summary: str) -> str:
    """Strip the <analysis> scratchpad and extract the <summary> content."""
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw_summary)
    m = re.search(r"<summary>([\s\S]*?)</summary>", text)
    if m:
        text = text.replace(m.group(0), f"Summary:\n{m.group(1).strip()}")
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()

"""生成对话压缩后的提示文案。格式化历史摘要，拼接上下文超限说明语句，依据配置追加消息保留标识与禁止追问指令，
产出完整文本，用以替换压缩历史，保障对话无缝接续。"""
def build_compact_summary_message(
    summary: str,
    *,
    suppress_follow_up: bool = False,
    recent_preserved: bool = False,
) -> str:
# 定义函数，参数：压缩摘要、是否禁止追问、是否保留最近消息，返回拼接后的提示文本
    """Create the injected user message that replaces compacted history."""
    # 函数说明：生成替换压缩后历史的用户消息
    formatted = format_compact_summary(summary)
    # 格式化压缩摘要内容
    text = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The summary below covers the earlier portion of the "
        "conversation.\n\n"
        f"{formatted}"
    )
    # 拼接基础提示文本，说明上下文超限，用摘要替代历史对话
    if recent_preserved:
        text += "\n\nRecent messages are preserved verbatim."
    # 如果保留最近消息，追加说明文字
    if suppress_follow_up:
        text += (
            "\nContinue the conversation from where it left off without asking "
            "the user any further questions. Resume directly — do not acknowledge "
            "the summary, do not recap what was happening, do not preface with "
            '"I\'ll continue" or similar. Pick up the last task as if the break '
            "never happened."
        )
    # 如果禁止追问，追加指令：直接继续对话，不回应摘要、不提问、不重复说明
    return text
# 返回最终拼接完成的完整文本

# ---------------------------------------------------------------------------
# Auto-compact tracking
# ---------------------------------------------------------------------------

@dataclass
class AutoCompactState:
    """Mutable state that persists across query loop turns."""

    compacted: bool = False
    turn_counter: int = 0
    turn_id: str = ""
    consecutive_failures: int = 0


# ---------------------------------------------------------------------------
# Context window helpers
# ---------------------------------------------------------------------------

def get_context_window(model: str, *, context_window_tokens: int | None = None) -> int:
    """Return the context window size for a model (conservative defaults)."""
    if context_window_tokens is not None and context_window_tokens > 0:
        return int(context_window_tokens)
    m = model.lower()
    if "opus" in m:
        return 200_000
    if "sonnet" in m:
        return 200_000
    if "haiku" in m:
        return 200_000
    # Kimi / other providers — be conservative
    return _DEFAULT_CONTEXT_WINDOW


def get_autocompact_threshold(
    model: str,
    *,
    context_window_tokens: int | None = None,
    auto_compact_threshold_tokens: int | None = None,
) -> int:
    """Calculate the token count at which auto-compact fires."""
    if auto_compact_threshold_tokens is not None and auto_compact_threshold_tokens > 0:
        return int(auto_compact_threshold_tokens)
    context_window = get_context_window(model, context_window_tokens=context_window_tokens)
    reserved = min(MAX_OUTPUT_TOKENS_FOR_SUMMARY, 20_000)
    effective = context_window - reserved
    return effective - AUTOCOMPACT_BUFFER_TOKENS


def should_autocompact(
    messages: list[ConversationMessage],
    model: str,
    state: AutoCompactState,
    *,
    context_window_tokens: int | None = None,
    auto_compact_threshold_tokens: int | None = None,
) -> bool:
    """Return True when the conversation should be auto-compacted."""
    if state.consecutive_failures >= MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES:
        return False
    token_count = estimate_message_tokens(messages)
    threshold = get_autocompact_threshold(
        model,
        context_window_tokens=context_window_tokens,
        auto_compact_threshold_tokens=auto_compact_threshold_tokens,
    )
    return token_count >= threshold


# ---------------------------------------------------------------------------
# Full compact execution (calls the LLM)
# ---------------------------------------------------------------------------
# 定义异步函数：调用大模型总结，实现对话完整压缩
async def compact_conversation(
    messages: list[ConversationMessage],  # 输入：待压缩的对话消息列表
    *,                                      # 关键字参数分隔符，后面必须用key=value传参
    api_client: Any,                        # 输入：调用AI接口的客户端对象
    model: str,                             # 输入：使用的AI大模型名称
    system_prompt: str = "",                # 输入：给AI的系统提示词，默认空
    preserve_recent: int = 6,                # 输入：保留最近N条消息不压缩，默认6条
    custom_instructions: str | None = None,  # 输入：自定义压缩指令，可选
    suppress_follow_up: bool = True,        # 输入：是否禁止AI追问，默认开启
    trigger: CompactTrigger = "manual",     # 输入：压缩触发方式，默认手动触发
    progress_callback: CompactProgressCallback | None = None,  # 输入：进度回调函数
    emit_hooks_start: bool = True,          # 输入：是否触发开始钩子，默认开启
    hook_executor: HookExecutor | None = None,  # 输入：钩子执行器对象
    carryover_metadata: dict[str, Any] | None = None,  # 输入：传递的追踪元数据
) -> CompactionResult:                      # 输出：返回压缩结果对象
    """文档注释：调用 LLM 总结实现消息压缩
    1. 先微压缩（低成本减少令牌）
    2. 拆分消息：旧消息（总结）+ 新消息（保留）
    3. 调用 LLM 生成结构化总结
    4. 用总结替换旧消息 + 保留新消息
    """
    # 导入AI接口请求相关的依赖类
    from openharness.api.client import ApiMessageRequest, ApiMessageCompleteEvent

    # 判断：如果对话消息总数 ≤ 需要保留的数量
    if len(messages) <= preserve_recent:
        # 直接返回不压缩的结果，无需处理
        return _build_passthrough_compaction_result(
            messages,
            trigger=trigger,
            compact_kind="full",
            metadata={"reason": "conversation already within preserve_recent window"},
        )

    # 第一步：执行轻量微压缩，降低token消耗，返回压缩后消息和节省的token数
    messages, tokens_freed = microcompact_messages(messages, keep_recent=DEFAULT_KEEP_RECENT)

    """计算压缩前的消息的总token数
    estimate_message_tokens是OpenHarness的对话token估算函数，用于统计消息列表的总token数。
先初始化计数器，获取单张图片的预估token，遍历所有消息和内容块，分别计算文本、工具结果、工具调用、图片的token并累加，
最后乘以4/3的安全冗余系数，返回整数结果。核心作用是精准估算对话占用token，为上下文压缩提供判断依据。 """
    pre_compact_tokens = estimate_message_tokens(messages)
    # 打印日志：记录当前压缩的消息数量和token数
    log.info("Compacting conversation: %d messages, ~%d tokens", len(messages), pre_compact_tokens)

    # 第二步：拆分消息，older=需要总结的旧消息，newer=直接保留的新消息
    older, newer = _split_preserving_tool_pairs(messages, preserve_recent=preserve_recent)

    # 第三步：构建AI压缩请求
    compact_prompt = get_compact_prompt(custom_instructions)  # 获取总结用的提示词
    # 组装待总结消息 = 旧消息 + 压缩提示词
    compact_messages = list(older) + [ConversationMessage.from_user_text(compact_prompt)]
    attachment_paths = _extract_attachment_paths(older)  # 从旧消息中提取附件路径
    discovered_tools = _extract_discovered_tools(older)  # 从旧消息中提取工具信息

    # 构建前置钩子需要传递的参数数据
    hook_payload = {
        "event": HookEvent.PRE_COMPACT.value,
        "trigger": trigger,
        "model": model,
        "message_count": len(messages),
        "token_count": pre_compact_tokens,
        "preserve_recent": preserve_recent,
        "attachments": attachment_paths,
        "discovered_tools": discovered_tools,
        **(carryover_metadata or {}),
    }

    # 记录压缩准备阶段的检查点数据
    start_checkpoint = _record_compact_checkpoint(
        carryover_metadata,
        checkpoint="compact_prepare",
        trigger=trigger,
        message_count=len(messages),
        token_count=pre_compact_tokens,
        details={
            "preserve_recent": preserve_recent,
            "attachments": attachment_paths,
            "discovered_tools": discovered_tools,
        },
    )

    # 判断：如果开启了开始钩子
    if emit_hooks_start:
        # 发送压缩准备开始的进度通知
        await _emit_progress(
            progress_callback,
            phase="hooks_start",
            trigger=trigger,
            message="Preparing conversation compaction.",
            checkpoint="compact_hooks_start",
            metadata=start_checkpoint,
        )

    # 判断：如果存在钩子执行器
    if hook_executor is not None:
        # 执行压缩前置钩子
        hook_result = await hook_executor.execute(HookEvent.PRE_COMPACT, hook_payload)
        # 判断：如果钩子返回阻止压缩
        if hook_result.blocked:
            # 获取阻止原因
            reason = hook_result.reason or "pre-compact hook blocked compaction"
            # 记录压缩失败检查点
            failed_checkpoint = _record_compact_checkpoint(
                carryover_metadata,
                checkpoint="compact_failed",
                trigger=trigger,
                message_count=len(messages),
                token_count=pre_compact_tokens,
                details={"reason": reason},
            )
            # 发送压缩失败进度
            await _emit_progress(
                progress_callback,
                phase="compact_failed",
                trigger=trigger,
                message=reason,
                checkpoint="compact_failed",
                metadata=failed_checkpoint,
            )
            # 返回不压缩的原始结果
            return _build_passthrough_compaction_result(
                messages,
                trigger=trigger,
                compact_kind="full",
                metadata={"reason": reason},
            )

    # 记录压缩正式开始的检查点
    compact_start_checkpoint = _record_compact_checkpoint(
        carryover_metadata,
        checkpoint="compact_start",
        trigger=trigger,
        message_count=len(messages),
        token_count=pre_compact_tokens,
        details={"preserve_recent": preserve_recent},
    )

    # 发送压缩开始的进度通知
    await _emit_progress(
        progress_callback,
        phase="compact_start",
        trigger=trigger,
        message="Compacting conversation memory.",
        checkpoint="compact_start",
        metadata=compact_start_checkpoint,
    )

    # 初始化变量
    summary_text = ""                              # 存储AI返回的总结文本
    messages_to_summarize = compact_messages       # 待总结的消息
    retry_messages = messages_to_summarize        # 重试时使用的消息
    ptl_retries = 0                                # prompt超长重试次数

    # 定义内部异步函数：流式调用AI接口获取总结内容
    async def _collect_summary(summary_request_messages: list[ConversationMessage]) -> str:
        collected = ""  # 存储收集到的流式响应
        # 把消息中的图片替换为占位符
        summary_request_messages = _replace_images_with_compaction_placeholders(
            summary_request_messages
        )
        # 调用AI流式接口发起请求
        stream = api_client.stream_message(
            ApiMessageRequest(
                model=model,
                messages=summary_request_messages,
                system_prompt=system_prompt or "You are a conversation summarizer.",
                max_tokens=MAX_OUTPUT_TOKENS_FOR_SUMMARY,
                tools=[],  # 压缩过程不使用工具调用
            )
        )
        # 如果stream是异步对象，先等待完成
        if inspect.isawaitable(stream):
            stream = await stream
        # 判断stream是否支持异步迭代
        if not hasattr(stream, "__aiter__"):
            raise RuntimeError("Compaction client did not provide a streaming response.")
        # 遍历流式响应，收集完整总结
        async for event in stream:
            if isinstance(event, ApiMessageCompleteEvent):
                collected = event.message.text
        # 如果收集到内容，返回总结；否则抛异常
        if collected.strip():
            return collected
        raise RuntimeError(ERROR_MESSAGE_INCOMPLETE_RESPONSE)

    # 循环重试调用AI总结，支持超时、超长重试
    for attempt in range(1, MAX_COMPACT_STREAMING_RETRIES + 2):
        try:
            # 带超时控制，调用AI总结函数
            summary_text = await asyncio.wait_for(
                _collect_summary(retry_messages),
                timeout=COMPACT_TIMEOUT_SECONDS,
            )
            break  # 成功获取总结，退出重试循环
        except Exception as exc:
            # 异常1：提示词超长，且未达到最大重试次数
            if _is_prompt_too_long_error(exc) and ptl_retries < MAX_PTL_RETRIES:
                # 裁剪消息头部（删除最早的内容）
                truncated = truncate_head_for_ptl_retry(retry_messages[:-1])
                if truncated:
                    ptl_retries += 1
                    retry_messages = [*truncated, retry_messages[-1]]
                    # 发送重试进度并记录检查点
                    await _emit_progress(
                        progress_callback,
                        phase="compact_retry",
                        trigger=trigger,
                        message="Compaction prompt was too large; retrying with older context trimmed.",
                        attempt=ptl_retries,
                        checkpoint="compact_retry_prompt_too_long",
                        metadata=_record_compact_checkpoint(
                            carryover_metadata,
                            checkpoint="compact_retry_prompt_too_long",
                            trigger=trigger,
                            message_count=len(retry_messages),
                            token_count=estimate_message_tokens(retry_messages),
                            attempt=ptl_retries,
                            details={"ptl_retries": ptl_retries},
                        ),
                    )
                    continue
            # 异常2：超过最大重试次数
            if attempt > MAX_COMPACT_STREAMING_RETRIES:
                # 发送失败进度，记录检查点，抛出异常
                await _emit_progress(
                    progress_callback,
                    phase="compact_failed",
                    trigger=trigger,
                    message=str(exc),
                    attempt=attempt,
                    checkpoint="compact_failed",
                    metadata=_record_compact_checkpoint(
                        carryover_metadata,
                        checkpoint="compact_failed",
                        trigger=trigger,
                        message_count=len(retry_messages),
                        token_count=estimate_message_tokens(retry_messages),
                        attempt=attempt,
                        details={"reason": str(exc)},
                    ),
                )
                raise
            # 其他异常：记录重试，继续循环
            await _emit_progress(
                progress_callback,
                phase="compact_retry",
                trigger=trigger,
                message=str(exc),
                attempt=attempt,
                checkpoint="compact_retry",
                metadata=_record_compact_checkpoint(
                    carryover_metadata,
                    checkpoint="compact_retry",
                    trigger=trigger,
                    message_count=len(retry_messages),
                    token_count=estimate_message_tokens(retry_messages),
                    attempt=attempt,
                    details={"reason": str(exc)},
                ),
            )

    # 判断：如果AI没有返回有效总结内容
    if not summary_text:
        # 发送失败进度，记录检查点
        await _emit_progress(
            progress_callback,
            phase="compact_failed",
            trigger=trigger,
            message=ERROR_MESSAGE_INCOMPLETE_RESPONSE,
            checkpoint="compact_failed",
            metadata=_record_compact_checkpoint(
                carryover_metadata,
                checkpoint="compact_failed",
                trigger=trigger,
                message_count=len(messages),
                token_count=pre_compact_tokens,
                details={"reason": ERROR_MESSAGE_INCOMPLETE_RESPONSE},
            ),
        )
        # 打印警告日志
        log.warning("Compact summary was empty — returning original messages")
        # 返回原始不压缩结果
        return _build_passthrough_compaction_result(
            messages,
            trigger=trigger,
            compact_kind="full",
            metadata={"reason": ERROR_MESSAGE_INCOMPLETE_RESPONSE},
        )

    """生成标准的总结消息内容 
    第四步：构建压缩后的新消息列表 build_compact_summary_message函数用来生成对话压缩后的提示话术，
    整理历史摘要并附上对应指令，替换掉精简后的旧聊天记录。"""
    summary_content = build_compact_summary_message(
        summary_text,
        suppress_follow_up=suppress_follow_up,
        recent_preserved=len(newer) > 0,
    )
    summary_msg = ConversationMessage.from_user_text(summary_content)  # 转为消息对象
    initial_post_compact_tokens = estimate_message_tokens([summary_msg, *newer])  # 统计压缩后token

    # 判断：如果存在后置钩子执行器
    if hook_executor is not None:
        # 执行压缩完成后置钩子
        post_hook_result = await hook_executor.execute(
            HookEvent.POST_COMPACT,
            {
                "event": HookEvent.POST_COMPACT.value,
                "trigger": trigger,
                "model": model,
                "pre_compact_message_count": len(messages),
                "post_compact_message_count": len(newer) + 1,
                "pre_compact_tokens": pre_compact_tokens,
                "post_compact_tokens": initial_post_compact_tokens,
                "attachments": attachment_paths,
                "discovered_tools": discovered_tools,
                **(carryover_metadata or {}),
            },
        )
        # 处理钩子返回的内容
        hook_note = post_hook_result.reason or "\n".join(
            result.output.strip()
            for result in post_hook_result.results
            if result.output.strip()
        )
        hook_attachments = _create_hook_attachments(hook_note)  # 创建钩子附件
    else:
        hook_attachments = []  # 无钩子则附件为空

    # 构建压缩结果的完整元数据
    compact_metadata = {
        "trigger": trigger,
        "compact_kind": "full",
        "pre_compact_message_count": len(messages),
        "pre_compact_token_count": pre_compact_tokens,
        "preserve_recent": preserve_recent,
        "tokens_freed_by_microcompact": tokens_freed,
        "pre_compact_discovered_tools": discovered_tools,
        "used_head_truncation_retry": ptl_retries > 0,
        "used_context_collapse": _metadata_has_checkpoint(carryover_metadata, "query_context_collapse_end"),
        "used_session_memory": False,
        "retry_attempts": max(0, attempt - 1 if "attempt" in locals() else 0),
        "attachments": attachment_paths,
    }

    # 判断：如果有传递过来的元数据，合并检查点信息
    if carryover_metadata is not None:
        checkpoints = carryover_metadata.get("compact_checkpoints")
        if isinstance(checkpoints, list):
            compact_metadata["compact_checkpoints"] = checkpoints
        compact_last = carryover_metadata.get("compact_last")
        if isinstance(compact_last, dict):
            compact_metadata["compact_last"] = compact_last

    # 创建最终的压缩结果对象
    compaction_result = CompactionResult(
        trigger=trigger,
        compact_kind="full",
        boundary_marker=create_compact_boundary_message(compact_metadata),
        summary_messages=[summary_msg],
        messages_to_keep=list(newer),
        attachments=_build_compact_attachments(older, metadata=carryover_metadata),
        hook_results=hook_attachments,
        compact_metadata=compact_metadata,
    )

    # 最终格式化压缩结果
    compaction_result = _finalize_compaction_result(compaction_result)
    """build_post_compact_messages按固定顺序组装压缩后的对话消息，整合边界标记、摘要、保留消息、附件与钩子消息，返回最终可用列表。"""
    post_compact_messages = build_post_compact_messages(compaction_result)  # 生成最终压缩消息
    post_compact_tokens = estimate_message_tokens(post_compact_messages)  # 统计最终token

    # 给结果补充统计元数据
    compaction_result.compact_metadata["post_compact_message_count"] = len(post_compact_messages)
    compaction_result.compact_metadata["post_compact_token_count"] = post_compact_tokens
    compaction_result.boundary_marker = create_compact_boundary_message(compaction_result.compact_metadata)

    # 打印压缩完成日志，展示压缩效果
    log.info(
        "Compaction done: %d -> %d messages, ~%d -> ~%d tokens (saved ~%d)",
        len(messages), len(post_compact_messages),
        pre_compact_tokens, post_compact_tokens,
        pre_compact_tokens - post_compact_tokens,
    )

    # 发送压缩完成的进度通知
    await _emit_progress(
        progress_callback,
        phase="compact_end",
        trigger=trigger,
        message="Conversation compaction complete.",
        checkpoint="compact_end",
        metadata=_record_compact_checkpoint(
            carryover_metadata,
            checkpoint="compact_end",
            trigger=trigger,
            message_count=len(post_compact_messages),
            token_count=post_compact_tokens,
            details={
                "pre_compact_message_count": len(messages),
                "post_compact_message_count": len(post_compact_messages),
                "pre_compact_tokens": pre_compact_tokens,
                "post_compact_tokens": post_compact_tokens,
                "tokens_saved": pre_compact_tokens - post_compact_tokens,
                "attachments": attachment_paths,
                "discovered_tools": discovered_tools,
            },
        ),
    )

    # 返回最终的对话压缩结果
    return compaction_result

# ---------------------------------------------------------------------------
# Auto-compact integration (called from query loop)
# ---------------------------------------------------------------------------
"""按需分层压缩（微压缩→上下文折叠→会话记忆→AI 全压缩），逐级降低 token 占用
核心作用：防止对话超出模型 token 限制，保证对话流畅持续进行

这是 OpenHarness 对话上下文自动压缩核心函数，每次对话前执行。先判断是否需要压缩，无需则直接返回；
需要则依次执行轻量压缩、上下文折叠、会话记忆压缩，都无效就执行完整 AI 压缩。
全程记录日志与状态，失败会计数，最终返回压缩后的消息和是否压缩标记，保障对话不超出模型 token 限制。

messages = 对话消息列表，api_client=AI 接口客户端，model = 使用模型，
system_prompt = 系统提示词，state = 压缩状态，preserve_recent = 保留最近消息数，
force = 强制压缩，trigger = 压缩触发方式，hook_executor为回调、carryover_metadata元数据、
context_window_tokens、auto_compact_threshold_tokens是token 限制配置。"""
async def auto_compact_if_needed(
    messages: list[ConversationMessage],
    *,
    api_client: Any,
    model: str,
    system_prompt: str = "",
    state: AutoCompactState,
    preserve_recent: int = 6,
    progress_callback: CompactProgressCallback | None = None,
    force: bool = False,
    trigger: CompactTrigger = "auto",
    hook_executor: HookExecutor | None = None,
    carryover_metadata: dict[str, Any] | None = None,
    context_window_tokens: int | None = None,
    auto_compact_threshold_tokens: int | None = None,
) -> tuple[list[ConversationMessage], bool]:
    """Check if auto-compact should fire, and if so, compact.

    Call this at the start of each query loop turn.

    Returns:
        (messages, was_compacted) — if compacted, messages is the new list.
    """
    """!!!压缩总开关，决定是否启动压缩流程。    
    核心是判断是否需要压缩的配置与消息：非强制压缩且无需压缩时，直接返回原消息，不执行压缩。"""
    if not force and not should_autocompact(
        messages,
        model,
        state,
        context_window_tokens=context_window_tokens,
        auto_compact_threshold_tokens=auto_compact_threshold_tokens,
    ):
        return messages, False

    """carryover_metadata = 元数据，trigger = 触发方式，消息数量、token 数等统计值。
    打印压缩触发日志，
    _record_compact_checkpoint记录压缩触发的检查点数据。"""
    log.info("Auto-compact triggered (failures=%d)", state.consecutive_failures)
    _record_compact_checkpoint(
        carryover_metadata,
        checkpoint=f"query_{trigger}_triggered",
        trigger=trigger,
        message_count=len(messages),
        token_count=estimate_message_tokens(messages),
        details={"consecutive_failures": state.consecutive_failures},
    )

    # Try microcompact first — may be enough
    """!!执行轻量微压缩，释放token。
    messages = 原始消息列表。 执行轻量微压缩，释放 token；若释放后无需压缩，直接返回结果。"""
    messages, tokens_freed = microcompact_messages(messages)
    # 记录微压缩完成检查点
    _record_compact_checkpoint(
        carryover_metadata,
        checkpoint="query_microcompact_end",
        trigger=trigger,
        message_count=len(messages),
        token_count=estimate_message_tokens(messages),
        details={"tokens_freed": tokens_freed},
    )
    # 微压缩有效且无需继续压缩，直接返回结果
    if tokens_freed > 0 and not should_autocompact(
        messages,
        model,
        state,
        context_window_tokens=context_window_tokens,
        auto_compact_threshold_tokens=auto_compact_threshold_tokens,
    ):
        log.info("Microcompact freed ~%d tokens, auto-compact no longer needed", tokens_freed)
        return messages, True

    """尝试执行上下文折叠
    preserve_recent = 保留最近消息数量。  尝试折叠超大上下文，更新消息，压缩达标则直接返回。"""
    context_collapsed = try_context_collapse(messages, preserve_recent=preserve_recent)
    # 上下文折叠成功
    if context_collapsed is not None:
        # 发送上下文折叠开始的进度通知
        await _emit_progress(
            progress_callback,
            phase="context_collapse_start",
            trigger=trigger,
            message="Collapsing oversized context before full compaction.",
            checkpoint="query_context_collapse_start",
            metadata=_record_compact_checkpoint(
                carryover_metadata,
                checkpoint="query_context_collapse_start",
                trigger=trigger,
                message_count=len(messages),
                token_count=estimate_message_tokens(messages),
            ),
        )
        # 更新消息为折叠后的内容
        messages = context_collapsed
        # 发送上下文折叠完成的进度通知
        await _emit_progress(
            progress_callback,
            phase="context_collapse_end",
            trigger=trigger,
            message="Context collapse complete.",
            checkpoint="query_context_collapse_end",
            metadata=_record_compact_checkpoint(
                carryover_metadata,
                checkpoint="query_context_collapse_end",
                trigger=trigger,
                message_count=len(messages),
                token_count=estimate_message_tokens(messages),
            ),
        )
        # 达标则返回压缩后消息
        if not force and not should_autocompact(
            messages,
            model,
            state,
            context_window_tokens=context_window_tokens,
            auto_compact_threshold_tokens=auto_compact_threshold_tokens,
        ):
            return messages, True

    """!!核心压缩方案，长期对话优化关键。     尝试轻量级会话记忆压缩
    将会话历史压缩为记忆摘要，更新状态，返回压缩后消息。
    
    try_session_memory_compaction函数是轻量级对话压缩方法，在 AI 压缩前执行。
    消息过短则不处理，拆分历史与最新消息，将历史生成摘要。若压缩无效直接退出，
    否则记录压缩信息并生成压缩结果，返回优化后的对话数据。"""
    session_memory = try_session_memory_compaction(
        messages,  # 传入当前所有对话消息
        # 保留最近消息数量：取传入值和默认最小值中更大的那个
        preserve_recent=max(preserve_recent, SESSION_MEMORY_KEEP_RECENT),
        trigger=trigger,  # 压缩触发条件（如长度超限/自动触发）
        metadata=carryover_metadata,  # 传入需要传递的元数据（记录点）
    )

    # 如果轻量化-会话记忆压缩成功（返回了有效会话记忆）
    if session_memory is not None:
        # 发送会话记忆压缩开始进度通知
        await _emit_progress(
            progress_callback,  # 进度回调函数
            phase="session_memory_start",  # 阶段：会话记忆开始
            trigger=trigger,
            message="Condensing earlier conversation into session memory.",  # 提示语
            checkpoint="query_session_memory_start",  # 检查点标识
            # 调用  记录精简版检查点数据的函数_record_compact_checkpoint，记录压缩开始的检查点元数据
            metadata=_record_compact_checkpoint(
                carryover_metadata,
                checkpoint="query_session_memory_start",
                trigger=trigger,
                message_count=len(messages),  # 压缩前消息总数
                token_count=estimate_message_tokens(messages),  # 压缩前令牌总数
            ),
        )

        # 异步发送进度通知：会话记忆生成完成
        await _emit_progress(
            progress_callback,
            phase="session_memory_end",  # 阶段：会话记忆结束
            trigger=trigger,
            message="Session memory condensation complete.",
            checkpoint="query_session_memory_end",
            # 记录压缩完成的检查点
            metadata=_record_compact_checkpoint(
                carryover_metadata,
                checkpoint="query_session_memory_end",
                trigger=trigger,
                # 压缩后的新消息数量,"""build_post_compact_messages按固定顺序组装压缩后的对话消息，整合边界标记、摘要、保留消息、附件与钩子消息，返回最终可用列表。"""
                message_count=len(build_post_compact_messages(session_memory)),
                # 压缩后的新消息令牌数
                token_count=estimate_message_tokens(build_post_compact_messages(session_memory)),
            ),
        )

        # 更新状态：标记已压缩
        state.compacted = True
        # 对话轮次 +1
        state.turn_counter += 1
        # 生成新的唯一轮次ID
        state.turn_id = uuid4().hex
        # 连续压缩失败次数重置为0
        state.consecutive_failures = 0
        # 返回压缩后的消息 + 压缩成功标记(True)
        return build_post_compact_messages(session_memory), True

    """!!!轻量化压缩失败，进入【完整压缩兜底流程】,执行最终完整 AI 压缩，异常处理保障流程不崩溃
    最终压缩兜底，异常处理保障流程不崩溃。
    compact_conversation调用大模型总结，实现对话完整压缩，最终返回压缩结果对象
    调用 LLM 总结实现消息压缩
    1. 先微压缩（低成本减少令牌）
    2. 拆分消息：旧消息（总结）+ 新消息（保留）
    3. 调用 LLM 生成结构化总结
    4. 用总结替换旧消息 + 保留新消息
    
    执行最终完整 AI 压缩：成功则返回结果；失败则记录错误、增加失败计数，返回原消息。"""
    try:
        # 异步调用完整对话压缩函数（AI 重写/总结长对话）
        result = await compact_conversation(
            messages,
            api_client=api_client,  # AI 接口客户端
            model=model,  # 使用的模型
            system_prompt=system_prompt,  # 系统提示词
            preserve_recent=preserve_recent,  # 保留最近消息数
            suppress_follow_up=True,  # 不生成追问
            trigger=trigger,  # 触发条件
            progress_callback=progress_callback,  # 进度回调
            hook_executor=hook_executor,  # 钩子执行器
            carryover_metadata=carryover_metadata,  # 传递元数据
        )

        # 完整压缩成功，更新状态
        state.compacted = True
        state.turn_counter += 1
        state.turn_id = uuid4().hex
        state.consecutive_failures = 0
        # 返回压缩后的消息 + 成功标记
        return build_post_compact_messages(result), True

    # 压缩过程出现任何异常（网络/模型/超时等）
    except Exception as exc:
        # 连续失败次数 +1
        state.consecutive_failures += 1
        # 记录【压缩失败】检查点，带上错误原因和失败次数
        _record_compact_checkpoint(
            carryover_metadata,
            checkpoint=f"query_{trigger}_failed",
            trigger=trigger,
            message_count=len(messages),
            token_count=estimate_message_tokens(messages),
            # 错误详情：异常信息 + 当前连续失败次数
            details={"reason": str(exc), "consecutive_failures": state.consecutive_failures},
        )

        # 打印错误日志
        log.error(
            "Auto-compact failed (attempt %d/%d): %s",
            state.consecutive_failures,  # 当前第几次失败
            MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,  # 最大允许失败次数
            exc,  # 异常对象
        )
        # 压缩失败：返回原始消息 + 失败标记(False)
        return messages, False
# ---------------------------------------------------------------------------
# Legacy compat
# ---------------------------------------------------------------------------
"""用于生成最近对话的精简摘要，截取最新消息，提取角色和文本内容，限制长度后拼接返回，用于快速展示对话历史。"""
# 定义生成对话摘要的函数，参数为消息列表、最大截取条数，返回字符串
def summarize_messages(
    messages: list[ConversationMessage],
    *,
    max_messages: int = 8,
) -> str:
    """Produce a compact textual summary of recent messages (legacy)."""
    # 截取消息列表末尾最新的max_messages条数据
    selected = messages[-max_messages:]
    # 创建空列表，用于存储格式化后的每一行内容
    lines: list[str] = []
    # 遍历选中的所有消息
    for message in selected:
        # 去除消息文本的前后空白字符
        text = message.text.strip()
        # 如果文本为空，跳过这条消息
        if not text:
            continue
        # 拼接角色和文本内容，最多保留300字符，加入列表
        lines.append(f"{message.role}: {text[:300]}")
    # 用换行符连接所有行，返回最终摘要字符串
    return "\n".join(lines)

"""旧版对话压缩工具，将早期消息生成摘要替换，保留最新消息，返回整洁的压缩后对话列表。"""
# 定义旧版消息压缩函数，参数为消息列表、保留最新消息数，返回压缩后的消息列表
def compact_messages(
    messages: list[ConversationMessage],
    *,
    preserve_recent: int = 6,
) -> list[ConversationMessage]:
    # 函数说明：用摘要替换早期对话历史（旧版方法）
    """Replace older conversation history with a synthetic summary (legacy)."""
    # 如果消息总数小于等于需要保留的数量，直接清理并返回原消息
    if len(messages) <= preserve_recent:
        return sanitize_conversation_messages(list(messages))
    # 将消息拆分为历史消息和需要保留的最新消息
    older, newer = _split_preserving_tool_pairs(messages, preserve_recent=preserve_recent)
    # 对历史消息生成摘要
    summary = summarize_messages(older)
    # 如果生成的摘要为空，直接返回最新消息
    if not summary:
        return list(newer)
    # 组装摘要消息+最新消息，清理格式后返回最终结果
    return sanitize_conversation_messages([
        ConversationMessage(
            role="user",
            content=[TextBlock(text=f"[conversation summary]\n{summary}")],
        ),
        *newer,
    ])

__all__ = [
    "AUTO_COMPACT_BUFFER_TOKENS",
    "AutoCompactState",
    "CompactAttachment",
    "CompactionResult",
    "COMPACTABLE_TOOLS",
    "TIME_BASED_MC_CLEARED_MESSAGE",
    "auto_compact_if_needed",
    "build_post_compact_messages",
    "build_compact_summary_message",
    "compact_conversation",
    "compact_messages",
    "create_compact_boundary_message",
    "estimate_conversation_tokens",
    "estimate_message_tokens",
    "format_compact_summary",
    "get_autocompact_threshold",
    "get_compact_prompt",
    "microcompact_messages",
    "should_autocompact",
    "summarize_messages",
]
