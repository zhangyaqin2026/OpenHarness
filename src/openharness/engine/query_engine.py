"""High-level conversation engine."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from openharness.api.client import SupportsStreamingMessages
from openharness.engine.cost_tracker import CostTracker
from openharness.coordinator.coordinator_mode import get_coordinator_user_context
from openharness.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, sanitize_conversation_messages
from openharness.engine.query import AskUserPrompt, PermissionPrompt, QueryContext, remember_user_goal, run_query
from openharness.engine.stream_events import AssistantTurnComplete, StreamEvent
from openharness.config.settings import Settings
from openharness.hooks import HookEvent, HookExecutor
from openharness.permissions.checker import PermissionChecker
from openharness.services.autodream.service import schedule_auto_dream
from openharness.tools.base import ToolRegistry

"""这段代码是 AI 高级对话引擎核心类 QueryEngine，负责管理对话全流程，存储对话历史、配置参数，
处理用户消息提交、工具调用、成本追踪，支持流式响应和后台任务，实现完整的 AI 交互控制。

tool_metadata: dict[str, object] | None = None,的意思是上层传进来一个字典 → 保存到 QueryEngine 内部
如果上层没传 → 创建一个空字典。从此这个字典就跟着 QueryEngine 一直活着"""
class QueryEngine:
    """Owns conversation history and the tool-aware model loop.
    初始化引擎，接收各类核心参数，赋值给实例属性，创建对话历史列表和成本追踪器。"""
    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        tool_registry: ToolRegistry,
        permission_checker: PermissionChecker,
        cwd: str | Path,
        model: str,
        system_prompt: str,
        max_tokens: int = 4096,
        context_window_tokens: int | None = None,
        auto_compact_threshold_tokens: int | None = None,
        max_turns: int | None = 8,
        permission_prompt: PermissionPrompt | None = None,
        ask_user_prompt: AskUserPrompt | None = None,
        hook_executor: HookExecutor | None = None,
        tool_metadata: dict[str, object] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._api_client = api_client
        self._tool_registry = tool_registry
        self._permission_checker = permission_checker
        self._cwd = Path(cwd).resolve()
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._context_window_tokens = context_window_tokens
        self._auto_compact_threshold_tokens = auto_compact_threshold_tokens
        self._max_turns = max_turns
        self._permission_prompt = permission_prompt
        self._ask_user_prompt = ask_user_prompt
        self._hook_executor = hook_executor
        self._tool_metadata = tool_metadata or {}
        self._settings = settings
        self._messages: list[ConversationMessage] = []
        self._cost_tracker = CostTracker()

    @property
    def messages(self) -> list[ConversationMessage]:
        """Return the current conversation history."""
        return list(self._messages)

    @property
    def max_turns(self) -> int | None:
        """Return the maximum number of agentic turns per user input, if capped."""
        return self._max_turns

    @property
    def api_client(self) -> SupportsStreamingMessages:
        """Return the active API client."""
        return self._api_client

    @property
    def model(self) -> str:
        """Return the active model identifier."""
        return self._model

    @property
    def system_prompt(self) -> str:
        """Return the active system prompt."""
        return self._system_prompt

    @property
    def tool_metadata(self) -> dict[str, object]:
        """Return the mutable tool metadata/carry-over state."""
        return self._tool_metadata

    @property
    def total_usage(self):
        """Return the total usage across all turns."""
        return self._cost_tracker.total

    """ 清空对话历史，重置成本追踪器。   """
    def clear(self) -> None:
        """Clear the in-memory conversation history."""
        self._messages.clear()
        self._cost_tracker = CostTracker()

    """更新对话使用的系统提示词。"""
    def set_system_prompt(self, prompt: str) -> None:
        """Update the active system prompt for future turns."""
        self._system_prompt = prompt

    """ 更新对话使用的 AI 模型。     """
    def set_model(self, model: str) -> None:
        """Update the active model for future turns."""
        self._model = model

    """更新对话使用的 API 客户端。  """
    def set_api_client(self, api_client: SupportsStreamingMessages) -> None:
        """Update the active API client for future turns."""
        self._api_client = api_client

    """设置单次用户输入的最大交互轮次，保证数值不小于 1。    """
    def set_max_turns(self, max_turns: int | None) -> None:
        """Update the maximum number of agentic turns per user input."""
        self._max_turns = None if max_turns is None else max(1, int(max_turns))

    """更新对话的权限校验器。    """
    def set_permission_checker(self, checker: PermissionChecker) -> None:
        """Update the active permission checker for future turns."""
        self._permission_checker = checker

    """！构建协调器上下文用户消息，无上下文则返回空。  返回值ConversationMessage 对象：成功构造的用户对话消息，包含角色和文本内容 
      多代理模式最关键的机制：
      Coordinator 维护一个全局共享上下文。 包含：子代理权限、工具列表、任务目标、文件状态、会话信息
      每次 AI 思考前，自动把这段上下文注入对话。所有代理都能看到统一的 “全局任务地图”
      """
    # 从协调器的全局运行环境中提取工具上下文信息，组装成一条标准的用户对话消息；如果没有工具上下文，就返回空值。
    def _build_coordinator_context_message(self) -> ConversationMessage | None:
        # 方法说明：创建一条携带协调器运行环境的用户消息
        """Build a synthetic user message carrying coordinator runtime context."""
        # 获取协调器的全局用户上下文数据
        context = get_coordinator_user_context()
        # 从上下文中取出 worker 工具相关的环境信息
        worker_tools_context = context.get("workerToolsContext")
        # 如果没有工具上下文信息
        if not worker_tools_context:
            # 直接返回空，不创建消息
            return None
        # 构造并返回一条用户角色的对话消息
        return ConversationMessage(
            # 消息角色：用户
            role="user",
            # 消息内容：标题 + 工具上下文文本
            content=[TextBlock(text=f"# Coordinator User Context\n\n{worker_tools_context}")],
        )

    """定义方法，传入对话消息列表，无返回值：   替换当前内存中的对话历史。    """
    def load_messages(self, messages: list[ConversationMessage]) -> None:
        """Replace the in-memory conversation history. 把传入的消息列表赋值给引擎的对话历史"""
        self._messages = list(messages)

    """后台执行自动记忆整合任务，不阻塞主流程。   
     self：代表当前这个类的实例自己（面向对象固定写法） -> None：表示这个方法执行完不返回任何值"""
    def _schedule_auto_dream(self) -> None:
        # 方法说明：用户轮次结束后，后台执行记忆整合，不影响主流程
        """Fire-and-forget background memory consolidation after a user turn."""
        # 如果没有配置信息
        if self._settings is None:
            # 直接退出，不执行
            return
        # 从系统的工具元数据里，拿出自动记忆（autodream）需要用到的上下文信息，存到变量 context 里
        context = self._tool_metadata.get("autodream_context")
        # 如果 context 是字典 → 转成关键字参数 kwargs； 如果不是 → 给一个空字典，避免报错。  这是防御性编程，防止程序崩溃。
        kwargs = dict(context) if isinstance(context, dict) else {}
        # 调用真正的后台任务函数schedule_auto_dream：把记忆整理任务扔到后台线程 / 进程去跑。
        schedule_auto_dream(
            # 传给后台任务：当前程序在哪个文件夹运行。
            cwd=self._cwd,
            # 把系统配置传给后台记忆任务。
            settings=self._settings,
            # 告诉后台任务：当前用的是哪个 AI 大模型。
            model=self._model,
            # 拿到当前对话的唯一 ID（会话 ID），确保记忆整理是针对这一次对话的。
            current_session_id=str(self._tool_metadata.get("session_id") or ""),
            # 传入其他扩展参数
            **kwargs,
        )

    """！判断能不能接续
    判断对话是否存在未完成的工具调用，需要继续交互。    """
    # 定义方法，返回布尔值：是否有未完成的工具调用
    def has_pending_continuation(self) -> bool:
        # 方法说明：对话以工具结果结尾，等待模型继续处理时返回True
        """Return True when the conversation ends with tool results awaiting a follow-up model turn."""
        # 如果没有任何对话消息
        if not self._messages:
            # 返回False
            return False
        # 拿到最后一条消息
        last = self._messages[-1]
        # 如果最后一条消息不是用户发的
        if last.role != "user":
            return False
        # 如果最后一条消息里没有工具执行结果块
        if not any(isinstance(block, ToolResultBlock) for block in last.content):
            return False
        # 倒着遍历最后一条之前的所有消息
        for msg in reversed(self._messages[:-1]):
            # 跳过不是助手发送的消息
            if msg.role != "assistant":
                continue
            # 检查助手是否调用过工具，有则返回True
            return bool(msg.tool_uses)
        # 遍历完都没找到，返回False
        return False

    """！！用户发了一条消息 → 把消息加入对话 → 启动 AI 思考 + 工具调用的完整流程 → 流式把结果返回给前端。  用户 ↔ AI 引擎的真正入口  
     prompt就是用户刚刚输入的那句话     返回值AsyncIterator实现流式输出
     
     用户发消息
进入 submit_message
构建 QueryContext
调用 run_query(context, query_messages)
run_query 内部：模型决定是否调用 Skill
"""
    async def submit_message(self, prompt: str | ConversationMessage) -> AsyncIterator[StreamEvent]:
        """Append a user message and execute the query loop."""
        #如果输入已经是标准消息 → 直接用；  如果是字符串 → 包装成 “用户说的话” 标准格式；    目的：统一格式，后面流程不报错
        user_message = (
            prompt
            if isinstance(prompt, ConversationMessage)
            else ConversationMessage.from_user_text(prompt)
        )

        """记住用户的目标（给 AI 长期记忆）: 把用户的问题记录下来存到 tool_metadata 里,
         作用：让 AI 记住 “用户想干什么”，多轮对话不忘事"""
        if user_message.text.strip() and not self._tool_metadata.pop("_suppress_next_user_goal", False):
            remember_user_goal(self._tool_metadata, user_message.text)

        """清理对话历史 + 加入用户消息   
        sanitize：清理脏数据、空消息，保证对话合法;  append：把用户这句话加到历史记录里;  作用：AI 才能看到上下文"""
        self._messages = sanitize_conversation_messages(self._messages)
        self._messages.append(user_message)

        """事件 1 用户提交消息：后续触发各类系统生命周期事件，由 HookExecutor 调度执行各类钩子逻辑。
        触发插件事件（用户提交消息）   hook_executor 在这里的作用：→ 告诉所有插件：用户刚刚发了一条消息！"""
        if self._hook_executor is not None:
            await self._hook_executor.execute(
                HookEvent.USER_PROMPT_SUBMIT,
                {
                    "event": HookEvent.USER_PROMPT_SUBMIT.value,
                    "prompt": user_message.text,
                },
            )

        """所有后面 AI 运行需要的东西，全部打包在这里QueryContext
        里面包含：用哪个模型、用哪些工具、权限规则、系统提示词、最大轮数、插件钩子、记忆数据"""
        context = QueryContext(
            api_client=self._api_client,
            tool_registry=self._tool_registry,  #ToolRegistry这里跟tool里面联动
            permission_checker=self._permission_checker,
            cwd=self._cwd,
            model=self._model,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            context_window_tokens=self._context_window_tokens,
            auto_compact_threshold_tokens=self._auto_compact_threshold_tokens,
            max_turns=self._max_turns,
            permission_prompt=self._permission_prompt,
            ask_user_prompt=self._ask_user_prompt,
            hook_executor=self._hook_executor,
            tool_metadata=self._tool_metadata,
        )

        """准备要发给 AI 的对话历史     如果是协调员模式，额外加一段系统上下文。目的：给 AI 看的完整消息列表"""
        query_messages = list(self._messages)
        # 构建协调器上下文消息
        coordinator_context = self._build_coordinator_context_message()
        # 如果有上下文消息
        if coordinator_context is not None:
            # 加入查询消息列表
            """coordinator2. 注入全局共享上下文"""
            query_messages.append(coordinator_context)
        try:
            """启动 AI 的大脑 + 工具调用:流式返回事件（文字、工具、状态）、AI 完成一轮思考后，更新本地历史、统计 token 费用 
            coordinator3： 多轮 ReAct 驱动循环执行子任务"""
            async for event, usage in run_query(context, query_messages):
                # 如果是助手完成一轮回复
                if isinstance(event, AssistantTurnComplete):
                    # 更新本地对话历史
                    self._messages = list(query_messages)
                # 如果有计费数据
                if usage is not None:
                    # 加入成本追踪器
                    self._cost_tracker.add(usage) ##链接cost_tracker
                # 把事件流式返回出去
                yield event
        finally:
            """无论成功失败，最后都执行auto_dream = 自动总结、自动记忆、自动整理对话"""
            self._schedule_auto_dream()

    """！接续工具调用，不用重新发消息
    接续未完成的对话，不新增用户消息，继续执行中断的工具调用，流式返回结果并统计成本。  """
    async def continue_pending(self, *, max_turns: int | None = None) -> AsyncIterator[StreamEvent]:
        """Continue an interrupted tool loop without appending a new user message."""
        # 创建查询上下文
        context = QueryContext(
            api_client=self._api_client,
            tool_registry=self._tool_registry, #ToolRegistry这里跟tool里面联动
            permission_checker=self._permission_checker,
            cwd=self._cwd,
            model=self._model,
            system_prompt=self._system_prompt,
            max_tokens=self._max_tokens,
            context_window_tokens=self._context_window_tokens,
            auto_compact_threshold_tokens=self._auto_compact_threshold_tokens,
            max_turns=max_turns if max_turns is not None else self._max_turns,
            permission_prompt=self._permission_prompt,
            ask_user_prompt=self._ask_user_prompt,
            hook_executor=self._hook_executor,
            tool_metadata=self._tool_metadata,
        )
        # 异步执行查询，遍历事件与用量
        async for event, usage in run_query(context, self._messages):
            # 有计费就记录
            if usage is not None:
                self._cost_tracker.add(usage)
            # 流式返回事件
            yield event
