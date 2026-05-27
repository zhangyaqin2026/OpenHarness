"""Load hooks from settings."""

from __future__ import annotations

from collections import defaultdict
from openharness.hooks.events import HookEvent
from openharness.hooks.schemas import HookDefinition

"""OpenHarness 的钩子注册与加载模块。
HookRegistry负责按事件分类存储钩子，提供注册、查询、生成摘要功能；
load_hook_registry从系统配置和插件中读取钩子定义，统一注册到注册表，为整个钩子系统提供数据管理。

必须掌握的重点（着重）
HookRegistry：钩子系统的中央存储器，所有钩子都存在这里
register + get：注册表最核心的两个方法（存钩子、取钩子）
load_hook_registry：唯一入口，加载配置与插件的所有钩子
总结
代码负责管理、存储、加载钩子
是整个 Hook 系统的数据底座
执行器HookExecutor必须依赖它才能工作
"""

"""钩子注册表，按事件类型存储所有钩子。"""
class HookRegistry:
    """Store hooks grouped by event."""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookDefinition]] = defaultdict(list)

    """register：将一个钩子注册到对应事件下。"""
    def register(self, event: HookEvent, hook: HookDefinition) -> None:
        """Register one hook."""
        self._hooks[event].append(hook)

    """get：根据事件获取所有对应的钩子列表。"""
    def get(self, event: HookEvent) -> list[HookDefinition]:
        """Return hooks registered for an event."""
        return list(self._hooks.get(event, []))

    """summary：生成人类可读的钩子汇总信息，用于日志 / 调试。"""
    def summary(self) -> str:
        """Return a human-readable hook summary."""
        lines: list[str] = []
        for event in HookEvent:
            hooks = self.get(event)
            if not hooks:
                continue
            lines.append(f"{event.value}:")
            for hook in hooks:
                matcher = getattr(hook, "matcher", None)
                detail = getattr(hook, "command", None) or getattr(hook, "prompt", None) or getattr(hook, "url", None) or ""
                suffix = f" matcher={matcher}" if matcher else ""
                lines.append(f"  - {hook.type}{suffix}: {detail}")
        return "\n".join(lines)

"""load_hook_registry：从配置 + 插件加载所有钩子，创建并返回注册表。"""
def load_hook_registry(settings, plugins=None) -> HookRegistry:
    """Load hooks from the current settings object."""
    registry = HookRegistry()
    for raw_event, hooks in settings.hooks.items():
        try:
            event = HookEvent(raw_event)
        except ValueError:
            continue
        for hook in hooks:
            registry.register(event, hook)
    for plugin in plugins or []:
        if not plugin.enabled:
            continue
        for raw_event, hooks in plugin.hooks.items():
            try:
                event = HookEvent(raw_event)
            except ValueError:
                continue
            for hook in hooks:
                registry.register(event, hook)
    return registry
