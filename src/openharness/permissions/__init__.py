"""Permission helpers for OpenHarness."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from openharness.permissions.checker import PermissionChecker, PermissionDecision
    from openharness.permissions.modes import PermissionMode

__all__ = ["PermissionChecker", "PermissionDecision", "PermissionMode"]

"""OpenHarness 的权限模块导出文件，作用是统一对外暴露权限相关类，并实现延迟加载（用到时才真正导入），
让导入更简洁、启动更快，不重复写复杂路径。"""
def __getattr__(name: str):
    if name in {"PermissionChecker", "PermissionDecision"}:
        from openharness.permissions.checker import PermissionChecker, PermissionDecision

        return {
            "PermissionChecker": PermissionChecker,
            "PermissionDecision": PermissionDecision,
        }[name]
    if name == "PermissionMode":
        from openharness.permissions.modes import PermissionMode

        return PermissionMode
    raise AttributeError(name)
