"""Skill data models."""

from __future__ import annotations

from dataclasses import dataclass

"""技能的标准数据格式
作用：统一记录技能的名称、说明、内容、配置等全部信息"""
@dataclass(frozen=True)
class SkillDefinition:
    """A loaded skill."""

    name: str
    description: str
    content: str
    source: str
    path: str | None = None
    base_dir: str | None = None
    command_name: str | None = None
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    user_invocable: bool = True
    disable_model_invocation: bool = False
    model: str | None = None
    argument_hint: str | None = None
