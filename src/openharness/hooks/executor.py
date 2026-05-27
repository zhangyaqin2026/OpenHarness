"""Hook execution engine."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from openharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest, SupportsStreamingMessages
from openharness.engine.messages import ConversationMessage
from openharness.hooks.events import HookEvent
from openharness.hooks.loader import HookRegistry
from openharness.hooks.schemas import (
    AgentHookDefinition,
    CommandHookDefinition,
    HookDefinition,
    HttpHookDefinition,
    PromptHookDefinition,
)
from openharness.hooks.types import AggregatedHookResult, HookResult
from openharness.sandbox import SandboxUnavailableError
from openharness.utils.shell import create_shell_subprocess

"""钩子 = 系统事件触发时自动运行的扩展逻辑。

OpenHarness 的钩子执行引擎，负责在系统生命周期事件触发时，运行注册的各类钩子（命令、HTTP、提示词、代理）。
它管理执行上下文，匹配钩子规则，异步执行钩子逻辑，返回执行结果，实现系统的扩展、拦截、校验与自定义处理，是框架灵活扩展的核心模块。

你必须记住的 3 个重点（重中之重）
HookExecutor 是整个钩子系统的心脏
execute () 是唯一入口，所有钩子都从这里触发
它支持 4 种钩子：命令、HTTP、提示词、代理
用最简单一句话总结
HookExecutor 就是当系统发生某个事件时，自动找到并运行所有对应的钩子，并把结果汇总回来的中央执行器。


_run_prompt_like_hook：AI 钩子核心，用大模型做智能校验
HookExecutionContext：保证钩子正确运行的上下文环境
总结
代码是系统扩展核心，支持命令 / HTTP/AI 三类钩子
重点是执行器HookExecutor、执行方法execute、AI 钩子_run_prompt_like_hook
作用是让框架可扩展、可拦截、可自定义处理流程
"""

"""钩子执行上下文数据类，存储执行所需的目录、API 客户端、默认模型，为钩子运行提供环境。"""
@dataclass
class HookExecutionContext:
    """Context passed into hook execution."""

    cwd: Path
    api_client: SupportsStreamingMessages
    default_model: str

"""!!钩子执行器主类，初始化时接收钩子注册表和执行上下文。"""
class HookExecutor:
    """Execute hooks for lifecycle events. 在系统生命周期事件上执行钩子。"""
    #类的构造函数，创建 HookExecutor 时必须传入两个东西：registry：钩子注册表（里面存了所有钩子）；context：执行上下文（运行环境：目录、API 客户端、模型）
    def __init__(self, registry: HookRegistry, context: HookExecutionContext) -> None:
        self._registry = registry
        self._context = context

    """更新钩子注册表，替换当前正在使用的钩子注册表。"""
    def update_registry(self, registry: HookRegistry) -> None:
        """Replace the active hook registry."""
        self._registry = registry

    """动态更新执行上下文（API 客户端、模型）。 
    api_client：AI 接口客户端（发请求用）；default_model：默认使用的大模型"""
    def update_context(
        self,
        *,
        api_client: SupportsStreamingMessages | None = None,
        default_model: str | None = None,
    ) -> None:
        """Update the active hook execution context."""
        if api_client is not None:
            self._context.api_client = api_client #替换上下文里的 API 客户端。
        if default_model is not None:
            self._context.default_model = default_model #更新默认使用的大模型。

    """!!执行指定事件的所有匹配钩子，分发到对应处理函数，汇总结果返回。
    异步执行钩子，传入：event：触发的事件（如工具调用、消息发送、任务开始）；payload：事件数据（工具名、消息内容等）
    返回：所有钩子的执行结果"""
    async def execute(self, event: HookEvent, payload: dict[str, Any]) -> AggregatedHookResult:
        """Execute all matching hooks for an event."""
        results: list[HookResult] = []
        #results创建空列表，用来存放所有钩子的执行结果。   遍历当前事件注册的所有钩子。
        for hook in self._registry.get(event):
            if not _matches_hook(hook, payload):
                #检查这个钩子是否匹配当前事件（比如工具名是否符合）；不匹配就跳过，不执行这个钩子。
                continue
            if isinstance(hook, CommandHookDefinition):
                #如果这个钩子是 命令型钩子（执行 Shell 命令），运行命令钩子，并把结果加入列表。
                results.append(await self._run_command_hook(hook, event, payload))
            elif isinstance(hook, HttpHookDefinition):
                #如果是 HTTP 请求钩子，发送 HTTP 请求，并记录结果。
                results.append(await self._run_http_hook(hook, event, payload))
            elif isinstance(hook, PromptHookDefinition):
                #如果是 提示词校验钩子（AI 判断是否通过）。 运行 AI 校验钩子，不是代理模式。
                results.append(await self._run_prompt_like_hook(hook, event, payload, agent_mode=False))
            elif isinstance(hook, AgentHookDefinition):
                #如果是 代理智能钩子（更强的 AI 判断）。运行 AI 代理钩子。
                results.append(await self._run_prompt_like_hook(hook, event, payload, agent_mode=True))
        #把所有钩子的执行结果汇总返回。
        return AggregatedHookResult(results=results)

    """执行命令型钩子，运行 Shell 命令，注入参数，捕获输出与退出码。"""
    async def _run_command_hook(
        self,
        hook: CommandHookDefinition,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> HookResult:
        command = _inject_arguments(hook.command, payload, shell_escape=True)
        try:
            process = await create_shell_subprocess(
                command,
                cwd=self._context.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "OPENHARNESS_HOOK_EVENT": event.value,
                    "OPENHARNESS_HOOK_PAYLOAD": json.dumps(payload),
                },
            )
        except SandboxUnavailableError as exc:
            return HookResult(
                hook_type=hook.type,
                success=False,
                blocked=hook.block_on_failure,
                reason=str(exc),
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=hook.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return HookResult(
                hook_type=hook.type,
                success=False,
                blocked=hook.block_on_failure,
                reason=f"command hook timed out after {hook.timeout_seconds}s",
            )

        output = "\n".join(
            part for part in (
                stdout.decode("utf-8", errors="replace").strip(),
                stderr.decode("utf-8", errors="replace").strip(),
            ) if part
        )
        success = process.returncode == 0
        return HookResult(
            hook_type=hook.type,
            success=success,
            output=output,
            blocked=hook.block_on_failure and not success,
            reason=output or f"command hook failed with exit code {process.returncode}",
            metadata={"returncode": process.returncode},
        )

    """执行 HTTP 钩子，向指定 URL 发送 POST 请求，传递事件与载荷。"""
    async def _run_http_hook(
        self,
        hook: HttpHookDefinition,
        event: HookEvent,
        payload: dict[str, Any],
    ) -> HookResult:
        try:
            async with httpx.AsyncClient(timeout=hook.timeout_seconds) as client:
                response = await client.post(
                    hook.url,
                    json={"event": event.value, "payload": payload},
                    headers=hook.headers,
                )
            success = response.is_success
            output = response.text
            return HookResult(
                hook_type=hook.type,
                success=success,
                output=output,
                blocked=hook.block_on_failure and not success,
                reason=output or f"http hook returned {response.status_code}",
                metadata={"status_code": response.status_code},
            )
        except Exception as exc:
            return HookResult(
                hook_type=hook.type,
                success=False,
                blocked=hook.block_on_failure,
                reason=str(exc),
            )

    """!!!执行提示词 / 代理型钩子，调用 AI 判断是否通过，解析结构化结果。"""
    async def _run_prompt_like_hook(
        self,
        hook: PromptHookDefinition | AgentHookDefinition,
        event: HookEvent,
        payload: dict[str, Any],
        *,
        agent_mode: bool,
    ) -> HookResult:
        prompt = _inject_arguments(hook.prompt, payload)
        prefix = (
            "You are validating whether a hook condition passes in OpenHarness. "
            "Return strict JSON: {\"ok\": true} or {\"ok\": false, \"reason\": \"...\"}."
        )
        if agent_mode:
            prefix += " Be more thorough and reason over the payload before deciding."
        request = ApiMessageRequest(
            model=hook.model or self._context.default_model,
            messages=[ConversationMessage.from_user_text(prompt)],
            system_prompt=prefix,
            max_tokens=512,
        )

        text_chunks: list[str] = []
        final_event: ApiMessageCompleteEvent | None = None
        async for event_item in self._context.api_client.stream_message(request):
            if isinstance(event_item, ApiMessageCompleteEvent):
                final_event = event_item
            else:
                text_chunks.append(event_item.text)

        text = "".join(text_chunks)
        if final_event is not None and final_event.message.text:
            text = final_event.message.text

        parsed = _parse_hook_json(text)
        if parsed["ok"]:
            return HookResult(hook_type=hook.type, success=True, output=text)
        return HookResult(
            hook_type=hook.type,
            success=False,
            output=text,
            blocked=hook.block_on_failure,
            reason=parsed.get("reason", "hook rejected the event"),
        )

"""匹配钩子规则，判断当前事件是否满足钩子触发条件。"""
def _matches_hook(hook: HookDefinition, payload: dict[str, Any]) -> bool:
    matcher = getattr(hook, "matcher", None)
    if not matcher:
        return True
    subject = str(payload.get("tool_name") or payload.get("prompt") or payload.get("event") or "")
    return fnmatch.fnmatch(subject, matcher)

"""将载荷数据注入钩子模板，替换$ARGUMENTS变量，支持 Shell 转义。"""
def _inject_arguments(
    template: str, payload: dict[str, Any], *, shell_escape: bool = False
) -> str:
    serialized = json.dumps(payload, ensure_ascii=True)
    if shell_escape:
        serialized = shlex.quote(serialized)
    return template.replace("$ARGUMENTS", serialized)

"""解析 AI 钩子返回的 JSON，提取ok和reason，兼容非 JSON 格式。"""
def _parse_hook_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool):
            return parsed
    except json.JSONDecodeError:
        pass
    lowered = text.strip().lower()
    if lowered in {"ok", "true", "yes"}:
        return {"ok": True}
    return {"ok": False, "reason": text.strip() or "hook returned invalid JSON"}
