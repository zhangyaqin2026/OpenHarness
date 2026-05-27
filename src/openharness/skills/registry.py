"""Skill registry."""

from __future__ import annotations

from openharness.skills.types import SkillDefinition

"""实现了AI 技能注册表，用于统一存储、注册、查找、列出所有技能，是管理 AI 技能的核心容器，保证技能能被快速调用和管理  
 SkillRegistry：最核心、最重要，技能的总管理器
功能：注册、存储、查找、列出 AI 所有技能
 """
class SkillRegistry:
    """Store loaded skills by name."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    """注册一个技能，用名称、命令名等作为键存储。"""
    def register(self, skill: SkillDefinition) -> None:
        """Register one skill."""
        for key in (skill.name, skill.command_name, skill.display_name, *skill.aliases):
            if key:
                self._skills[key] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Return a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return all skills sorted by name."""
        unique: dict[tuple[str, str | None], SkillDefinition] = {}
        for skill in self._skills.values():
            unique[(skill.source, skill.path or skill.name)] = skill
        return sorted(unique.values(), key=lambda skill: skill.command_name or skill.name)
