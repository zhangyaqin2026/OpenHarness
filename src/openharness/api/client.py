"""Anthropic API client wrapper with retry logic."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from anthropic import APIError, APIStatusError, AsyncAnthropic

from openharness.api.errors import (
    AuthenticationFailure,
    OpenHarnessApiError,
    RateLimitFailure,
    RequestFailure,
)
from openharness.auth.external import (
    claude_attribution_header,
    claude_oauth_betas,
    claude_oauth_headers,
    get_claude_code_session_id,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, assistant_message_from_api

log = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
OAUTH_BETA_HEADER = "oauth-2025-04-20"

"""Anthropic Claude API 异步客户端封装，核心实现流式调用、自动重试、错误转换、OAuth 鉴权。
用于 AI 模型对话流式输出，保障接口调用稳定、安全、可重试，是系统对接大模型的核心通信模块。 

AnthropicApiClient：整个文件的核心
stream_message()：对外提供流式对话 + 自动重试
_is_retryable / _get_retry_delay：保障接口稳定性
支持OAuth 鉴权、异常转换、流式输出、重试退避"""

#封装模型调用请求参数（模型、消息、系统提示等）
@dataclass(frozen=True)
class ApiMessageRequest:
    """Input parameters for a model invocation."""

    model: str
    messages: list[ConversationMessage]
    system_prompt: str | None = None
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)

#流式返回的文本片段事件
@dataclass(frozen=True)
class ApiTextDeltaEvent:
    """Incremental text produced by the model."""

    text: str

#流式结束事件，包含完整消息与 token 用量
@dataclass(frozen=True)
class ApiMessageCompleteEvent:
    """Terminal event containing the full assistant message."""

    message: ConversationMessage
    usage: UsageSnapshot
    stop_reason: str | None = None

#API 重试事件，返回重试信息
@dataclass(frozen=True)
class ApiRetryEvent:
    """A recoverable upstream failure that will be retried automatically."""

    message: str
    attempt: int
    max_attempts: int
    delay_seconds: float


ApiStreamEvent = ApiTextDeltaEvent | ApiMessageCompleteEvent | ApiRetryEvent

#流式消息接口协议，规范调用格式
class SupportsStreamingMessages(Protocol):
    """Protocol used by the query engine in tests and production."""

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield streamed events for the request."""

#判断异常是否可重试（网络 / 500/429 等）
def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, APIError):
        return True  # Network errors are retryable
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False

#计算指数退避重试延迟，带抖动
def _get_retry_delay(attempt: int, exc: Exception | None = None) -> float:
    """Calculate delay with exponential backoff and jitter."""
    import random

    # Check for Retry-After header
    if isinstance(exc, APIStatusError):
        retry_after = getattr(exc, "headers", {})
        if hasattr(retry_after, "get"):
            val = retry_after.get("retry-after")
            if val:
                try:
                    return min(float(val), MAX_DELAY)
                except (ValueError, TypeError):
                    pass

    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, delay * 0.25)
    return delay + jitter

"""!!负责创建 API 客户端、流式调用、自动重试、鉴权刷新、错误处理"""
class AnthropicApiClient:
    """Thin wrapper around the Anthropic async SDK with retry logic."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        auth_token: str | None = None,
        base_url: str | None = None,
        claude_oauth: bool = False,
        auth_token_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._auth_token = auth_token
        self._base_url = base_url
        self._claude_oauth = claude_oauth
        self._auth_token_resolver = auth_token_resolver
        self._session_id = get_claude_code_session_id() if claude_oauth else ""
        self._client = self._create_client()

    def _create_client(self) -> AsyncAnthropic:
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._auth_token:
            kwargs["auth_token"] = self._auth_token
            kwargs["default_headers"] = (
                claude_oauth_headers()
                if self._claude_oauth
                else {"anthropic-beta": OAUTH_BETA_HEADER}
            )
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return AsyncAnthropic(**kwargs)

    def _refresh_client_auth(self) -> None:
        if not self._claude_oauth or self._auth_token_resolver is None:
            return
        next_token = self._auth_token_resolver()
        if next_token and next_token != self._auth_token:
            self._auth_token = next_token
            self._client = self._create_client()

    """!!!这段 stream_message 是 Harness 最核心、最关键的对外接口。
    对外提供流式调用，自动重试最多 3 次，处理所有异常
    1. 提供流式能力（核心功能）让调用方可以像 ChatGPT 一样逐字接收回复，而不是等全部生成完,大幅提升用户体验
    2. 自动重试（高可用）网络抖动、瞬时限流、服务短暂不可用 → 自动恢复,无需上层业务处理
    3. 统一异常处理（稳定性）所有错误统一捕获、翻译、抛出;上层代码不用写大量 try/except
    4. 重试可见性（可观测）通过 yield ApiRetryEvent 让前端显示重试状态;日志完整，便于监控告警
    
    stream_message调用的_stream_once底层依赖 OpenAI/模型API / 网络请求库、
    stream_message被上层调用方：API 路由、聊天服务、前端流式接口，agent流程   调用
    
    stream_message 是 Harness 的 “中控层” / “门面层”
它的定位非常清晰：承上-接收上层业务的调用请求返回流式事件给上层
               启下-调用认证模块调用底层请求模块调用重试工具调用异常翻译
    """
    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Yield text deltas and the final assistant message with retry on transient errors.
        stream_message：方法名，发送消息并流式返回结果"""
        last_error: Exception | None = None

        """重试循环：MAX_RETRIES 是最大重试次数（比如 3），+1 表示包含首次调用, try包裹核心逻辑，捕获所有可能的异常。"""
        for attempt in range(MAX_RETRIES + 1):
            try:
                """!刷新 token    自动续期   保证 API 调用有权限"""
                self._refresh_client_auth()

                """!_stream_once真正发送请求的底层流式模块  这是真正干活的方法：建立流式连接、调用模型 API、逐块返回数据
                stream_message 本身不发请求，它只做：重试 + 异常 + 流式转发
                真正发请求的是 _stream_once。"""
                async for event in self._stream_once(request):
                    """把流式片段往外抛给调用方（前端 / 上层服务）"""
                    yield event
                return  # Success
            except OpenHarnessApiError: #捕获认证类错误（token 无效、无权限）
                raise  # Auth errors are not retried
            except Exception as exc: #捕获其他所有异常（网络超时、500、限流、断开连接等）,把错误存到 last_error
                last_error = exc
                if attempt >= MAX_RETRIES or not _is_retryable(exc):
                    #满足任一条件则不再重试：已达到最大重试次数;错误不可重试（如参数错误、400、404）
                    if isinstance(exc, APIError):
                        raise _translate_api_error(exc) from exc
                    raise RequestFailure(str(exc)) from exc

                delay = _get_retry_delay(attempt, exc) #计算重试等待时间（指数退避 / 限流等待时间）。
                status = getattr(exc, "status_code", "?")
                log.warning(
                    "API request failed (attempt %d/%d, status=%s), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES + 1, status, delay, exc,
                )
                yield ApiRetryEvent(
                    message=str(exc),
                    attempt=attempt + 1,
                    max_attempts=MAX_RETRIES + 1,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay) #异步等待，不阻塞进程，等待后进入下一次重试。

        if last_error is not None:
            if isinstance(last_error, APIError):
                raise _translate_api_error(last_error) from last_error
            raise RequestFailure(str(last_error)) from last_error

    """单次流式请求执行，解析返回片段与最终消息"""
    async def _stream_once(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        """Single attempt at streaming a message."""
        params: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_api_param() for message in request.messages],
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            params["system"] = request.system_prompt
        if self._claude_oauth:
            attribution = claude_attribution_header()
            params["system"] = (
                f"{attribution}\n{params['system']}"
                if params.get("system")
                else attribution
            )
        if request.tools:
            params["tools"] = request.tools
        if self._claude_oauth:
            params["betas"] = claude_oauth_betas()
            params["metadata"] = {
                "user_id": json.dumps(
                    {
                        "device_id": "openharness",
                        "session_id": self._session_id,
                        "account_uuid": "",
                    },
                    separators=(",", ":"),
                )
            }
            params["extra_headers"] = {"x-client-request-id": str(uuid.uuid4())}

        try:
            stream_api = self._client.beta.messages if self._claude_oauth else self._client.messages
            async with stream_api.stream(**params) as stream:
                async for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) != "text_delta":
                        continue
                    text = getattr(delta, "text", "")
                    if text:
                        yield ApiTextDeltaEvent(text=text)

                final_message = await stream.get_final_message()
        except APIError as exc:
            if isinstance(exc, APIStatusError) and exc.status_code in RETRYABLE_STATUS_CODES:
                raise  # Let retry logic handle it
            raise _translate_api_error(exc) from exc

        usage = getattr(final_message, "usage", None)
        yield ApiMessageCompleteEvent(
            message=assistant_message_from_api(final_message),
            usage=UsageSnapshot(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            ),
            stop_reason=getattr(final_message, "stop_reason", None),
        )

"""将官方 API 错误转为系统内部错误类型"""
def _translate_api_error(exc: APIError) -> OpenHarnessApiError:
    name = exc.__class__.__name__
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return AuthenticationFailure(str(exc))
    if name == "RateLimitError":
        return RateLimitFailure(str(exc))
    return RequestFailure(str(exc))
