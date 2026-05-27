"""Plugin exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from openharness.plugins.schemas import PluginManifest
    from openharness.plugins.types import LoadedPlugin

"""OpenHarness 插件系统的导出入口文件，统一暴露插件相关的类和函数，实现延迟导入优化性能，让外部能方便使用插件功能，是插件模块的统一出口。

getattr：插件模块核心优化机制，实现延迟加载
all：统一暴露插件接口，方便外部调用
文件是插件系统统一出口，负责接口暴露与延迟加载，提升性能。"""

#定义可导出的公开接口，规范外部调用的功能列表。
__all__ = [
    "LoadedPlugin",
    "PluginManifest",
    "discover_plugin_paths",
    "get_project_plugins_dir",
    "get_user_plugins_dir",
    "install_plugin_from_path",
    "load_plugins",
    "uninstall_plugin",
]

"""getattr：核心重点，动态延迟导入函数 / 类，用到时才加载，提升启动速度。"""
def __getattr__(name: str):
    if name in {"discover_plugin_paths", "get_project_plugins_dir", "get_user_plugins_dir", "load_plugins"}:
        from openharness.plugins.loader import (
            discover_plugin_paths,
            get_project_plugins_dir,
            get_user_plugins_dir,
            load_plugins,
        )

        return {
            "discover_plugin_paths": discover_plugin_paths,
            "get_project_plugins_dir": get_project_plugins_dir,
            "get_user_plugins_dir": get_user_plugins_dir,
            "load_plugins": load_plugins,
        }[name]
    if name in {"install_plugin_from_path", "uninstall_plugin"}:
        from openharness.plugins.installer import install_plugin_from_path, uninstall_plugin

        return {
            "install_plugin_from_path": install_plugin_from_path,
            "uninstall_plugin": uninstall_plugin,
        }[name]
    if name == "PluginManifest":
        from openharness.plugins.schemas import PluginManifest

        return PluginManifest
    if name == "LoadedPlugin":
        from openharness.plugins.types import LoadedPlugin

        return LoadedPlugin
    raise AttributeError(name)
