"""Tool abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from pydantic import BaseModel # Pydantic数据模型，用于校验工具的输入参数

if TYPE_CHECKING:
    from openharness.hooks.executor import HookExecutor

"""AI 工具系统的核心抽象层，定义了工具执行上下文、执行结果、工具基类和注册器"""

"""工具干活时的「工作台」
工作台在哪里？→ cwd（目录）
工作台有什么额外材料？→ metadata
工作台有没有额外流程？→ hook_executor"""
@dataclass
class ToolExecutionContext:

    cwd: Path # 当前工作目录（工具运行时的文件路径环境）
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据，默认空字典
    hook_executor: HookExecutor | None = None # 钩子执行器（工具运行前后的扩展逻辑），默认空

"""工具干完活交上来的「统一报告」
不管你是读文件、写文件、查网络、跑命令……所有工具必须返回这个格式：
输出内容
成功 / 失败
额外信息"""
@dataclass(frozen=True) # frozen=True：实例创建后不可修改，保证结果不可变、安全
class ToolResult:
    """Normalized tool execution result."""
    output: str # 工具执行的输出内容（必须有）
    is_error: bool = False # 是否执行失败，默认成功
    metadata: dict[str, Any] = field(default_factory=dict) # 结果附带的额外信息


"""BaseTool 是所有工具的「模具 / 标准」，规定：工具必须有名字、描述、输入格式;工具必须实现 execute 方法;工具必须能返回统一的 API 结构"""
class BaseTool(ABC): # 继承ABC → 这是抽象类，不能直接实例化，必须被继承
    """Base class for all OpenHarness tools."""
    # 所有工具必须定义这3个类属性
    name: str # 工具唯一名称（如：file_read）
    description: str # 工具描述（给AI看，告诉它这个工具干嘛用）
    input_model: type[BaseModel] # 工具输入参数的校验模型（Pydantic）

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        """Execute the tool."""

    def is_read_only(self, arguments: BaseModel) -> bool:
        """判断工具是否只读（不修改文件/系统），默认返回False Return whether the invocation is read-only."""
        del arguments # 删掉不用的参数
        return False

    def to_api_schema(self) -> dict[str, Any]:
        """把工具转换成AI能理解的API格式 Return the tool schema expected by the Anthropic Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }

"""ToolRegistry 是工具的「工具箱 / 管理器」，负责收纳、查找、暴露所有工具给 AI 使用。"""
class ToolRegistry:
    """工具注册器，用于注册、查找、管理所有工具，并生成 API 可用的工具 schema。
    Map tool names to implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {} # 内部字典：工具名 → 工具实例

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance. """
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool by name. 根据名字查找工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools. 获取所有已注册的工具"""
        return list(self._tools.values())

    def to_api_schema(self) -> list[dict[str, Any]]:
        """Return all tool schemas in API format. 把所有工具转换成AI API需要的格式"""
        return [tool.to_api_schema() for tool in self._tools.values()]
