"""Plugin manifest schemas."""

from __future__ import annotations

from pydantic import BaseModel

"""定义了 OpenHarness 的插件配置规范。使用 Pydantic 模型声明插件清单格式，包含名称、版本、路径等基础信息，
以及技能、工具、钩子、代理等扩展配置，用于读取和校验插件的 json 配置文件，是整个插件系统加载与识别的标准格式。"""

"""PluginManifest —— 插件系统的配置格式标准"""
class PluginManifest(BaseModel):
    """Plugin manifest stored in plugin.json or .claude-plugin/plugin.json."""

    name: str
    version: str = "0.0.0"
    description: str = ""
    enabled_by_default: bool = True
    skills_dir: str = "skills"
    tools_dir: str = "tools"
    hooks_file: str = "hooks.json"
    mcp_file: str = "mcp.json"
    # Extended fields: optional author, commands, agents, etc.
    author: dict | None = None
    commands: str | list | dict | None = None
    agents: str | list | None = None
    skills: str | list | None = None
    hooks: str | dict | list | None = None
