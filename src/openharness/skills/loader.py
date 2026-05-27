"""Skill loading from bundled, user, compatibility, and project directories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from openharness.config.paths import get_config_dir
from openharness.config.settings import load_settings
from openharness.skills._frontmatter import (
    optional_frontmatter_str,
    parse_bool_frontmatter,
    parse_skill_frontmatter,
    parse_skill_metadata,
)
from openharness.skills.bundled import get_bundled_skills
from openharness.skills.registry import SkillRegistry
from openharness.skills.types import SkillDefinition

logger = logging.getLogger(__name__)

_USER_COMPAT_SKILL_DIRS = (
    (".claude", "skills"),
    (".agents", "skills"),
)
_DEFAULT_PROJECT_SKILL_DIRS = (".openharness/skills", ".agents/skills", ".claude/skills")

"""负责加载 AI 助手的所有技能，从系统、用户、项目、插件目录读取技能文件，注册到技能管理器，
让 AI 知道能使用哪些功能，是 AI 技能系统的核心加载模块。  
 
 load_skill_registry：总入口、最核心，加载所有技能
load_skills_from_dirs：实际读取并解析技能文件
整体功能：给 AI加载、注册、管理所有可用技能
"""

#作用：获取用户级技能目录路径，确保目录存在。
def get_user_skills_dir() -> Path:
    """Return the OpenHarness user skills directory."""
    path = get_config_dir() / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path

#获取所有用户级技能目录列表。
def get_user_skill_dirs() -> list[Path]:
    """Return user-level skill directories loaded by default."""
    return [get_user_skills_dir(), *(Path.home().joinpath(*parts) for parts in _USER_COMPAT_SKILL_DIRS)]

"""!!AI 技能系统的总入口:
把系统自带、用户自定义、项目自带、插件自带的所有技能，全部装进一个大箱子SkillRegistry里，然后把箱子SkillRegistry返回给 AI 用。"""
def load_skill_registry(
    cwd: str | Path | None = None, # 当前工作目录（在哪里找项目技能）
    *,  # 关键字分隔，后面必须用名字传参
    extra_skill_dirs: Iterable[str | Path] | None = None,  # 额外的技能目录
    extra_plugin_roots: Iterable[str | Path] | None = None,# 额外的插件目录
    settings=None,
) -> SkillRegistry:
    """Load bundled, user-defined, project, and plugin skills."""
    """创建一个空的“技能注册表” = 一个大盒子，SkillRegistry用来装所有技能"""
    registry = SkillRegistry()
    for skill in get_bundled_skills():
        #第一步：加载【系统自带的内置技能】，全部注册到盒子里
        registry.register(skill)
    for skill in load_user_skills():
        # 第二步：加载【用户自己写的技能】，全部放进盒子
        registry.register(skill)
    for skill in load_skills_from_dirs(extra_skill_dirs, source="user"):
        # 第三步：加载【额外指定目录的技能】，放进盒子
        registry.register(skill)
    # 如果没传配置，就自动加载系统配置
    resolved_settings = settings or load_settings()
    # 如果传入了目录，并且配置允许加载项目技能
    if cwd is not None and getattr(resolved_settings, "allow_project_skills", True):
        # 找到项目里所有的技能文件夹
        project_dirs = discover_project_skill_dirs(
            cwd,
            getattr(resolved_settings, "project_skill_dirs", list(_DEFAULT_PROJECT_SKILL_DIRS)),
        )
        # 第四步：加载【项目自带的技能】，注册进盒子
        for skill in load_skills_from_dirs(project_dirs, source="project", create_missing=False):
            registry.register(skill)

    # 如果传入了目录，就继续加载插件
    if cwd is not None:
        # 动态导入插件加载器（避免循环引用）
        from openharness.plugins.loader import load_plugins

        # 加载所有插件; 如果插件没开启，跳过
        for plugin in load_plugins(resolved_settings, cwd, extra_roots=extra_plugin_roots):
            if not plugin.enabled:
                continue
          # 第五步：把【插件里带的技能】全部注册进盒子,
            for skill in plugin.skills:
                registry.register(skill)
    return registry # 所有技能加载完毕，把装满技能的盒子返回出去

#加载用户目录下的自定义技能。
def load_user_skills() -> list[SkillDefinition]:
    """Load markdown skills from user-level OpenHarness and compatibility directories."""
    return load_skills_from_dirs(get_user_skill_dirs(), source="user")

"""从当前目录向上查找项目技能目录。"""
def discover_project_skill_dirs(
    cwd: str | Path,
    project_skill_dirs: Iterable[str] | None = None,
) -> list[Path]:
    """Return existing project skill directories from cwd up to the git root.

    Directories are ordered from least-specific to most-specific so later registry
    entries can override broader project or user skills deterministically.
    """
    start = Path(cwd).expanduser().resolve()
    if not start.exists():
        start = start.parent
    if start.is_file():
        start = start.parent

    relative_dirs = _valid_project_skill_dirs(project_skill_dirs or _DEFAULT_PROJECT_SKILL_DIRS)
    git_root = _find_git_root(start)
    home = Path.home().resolve()
    current = start
    levels: list[Path] = []
    while True:
        levels.append(current)
        if git_root is not None and current == git_root:
            break
        if git_root is None and current == home:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    roots: list[Path] = []
    seen: set[Path] = set()
    for base in reversed(levels):
        for rel in relative_dirs:
            candidate = (base / rel).resolve()
            if candidate in seen or not candidate.is_dir():
                continue
            seen.add(candidate)
            roots.append(candidate)
    return roots

"""校验项目技能目录是否安全合法。"""
def _valid_project_skill_dirs(project_skill_dirs: Iterable[str]) -> list[Path]:
    """Return safe relative project skill paths."""
    paths: list[Path] = []
    for raw in project_skill_dirs:
        value = str(raw).strip()
        if not value:
            continue
        rel = Path(value)
        if rel.is_absolute() or ".." in rel.parts:
            logger.warning("Ignoring unsafe project skill dir: %s", raw)
            continue
        paths.append(rel)
    return paths

#查找当前项目的 Git 根目录。
def _find_git_root(start: Path) -> Path | None:
    """Find the nearest git root containing start, if any."""
    current = start
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent

"""！！！实际读取并解析技能文件  从指定目录中加载所有技能文件，返回技能列表
directories：要扫描的目录
# source：技能来源（user/project）
# create_missing：目录不存在时是否自动创建

这个函数扫描文件夹 → 找 SKILL.md → 读取解析 → 变成技能对象
是真正把技能文件加载进 AI 的核心函数
"""
def load_skills_from_dirs(
    directories: Iterable[str | Path] | None,
    *,
    source: str = "user",
    create_missing: bool = True,
) -> list[SkillDefinition]:
    """Load markdown skills from one or more directories.

    Supported layout:
    - ``<root>/<skill-dir>/SKILL.md``
    """
    # 创建空列表，用于存放最终加载好的所有技能
    skills: list[SkillDefinition] = []
    # 如果没有传入任何目录，直接返回空列表
    if not directories:
        return skills

    # 创建集合，用于记录已经加载过的文件，避免重复加载
    seen: set[Path] = set()
    for directory in directories:
        # 遍历每一个传入的目录，把目录路径标准化（绝对路径、解析 ~）
        root = Path(directory).expanduser().resolve()
        if create_missing:
            # 如果允许创建不存在的目录，就自动创建
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            # 如果不允许创建，且目录不存在，直接跳过这个目录
            continue
        candidates: list[Path] = []
        for child in sorted(root.iterdir()):
            # 遍历当前目录下的所有子文件/子文件夹，并排序
            if child.is_dir():
                skill_path = child / "SKILL.md"
                if skill_path.exists():
                    candidates.append(skill_path)
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            content = path.read_text(encoding="utf-8")
            default_name = path.parent.name
            metadata = _parse_skill_metadata(default_name, content)
            name = metadata["name"]
            description = metadata["description"]
            display_name = name if name != default_name else None
            skills.append(
                SkillDefinition(
                    name=name,
                    description=description,
                    content=content,
                    source=source,
                    path=str(path),
                    base_dir=str(path.parent),
                    command_name=default_name,
                    display_name=display_name,
                    user_invocable=metadata["user_invocable"],
                    disable_model_invocation=metadata["disable_model_invocation"],
                    model=metadata["model"],
                    argument_hint=metadata["argument_hint"],
                )
            )
    return skills

"""解析技能 Markdown 文件的名称和描述。"""
def _parse_skill_markdown(default_name: str, content: str) -> tuple[str, str]:
    """Parse name and description from a skill markdown file with YAML frontmatter support."""
    return parse_skill_frontmatter(default_name, content, fallback_template="Skill: {name}")

"""解析技能文件的完整元数据（配置、参数、开关等）。"""
def _parse_skill_metadata(default_name: str, content: str) -> dict:
    parsed = parse_skill_metadata(default_name, content, fallback_template="Skill: {name}")
    frontmatter = parsed.get("frontmatter")
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    return {
        "name": str(parsed["name"]),
        "description": str(parsed["description"]),
        "user_invocable": parse_bool_frontmatter(frontmatter.get("user-invocable"), default=True),
        "disable_model_invocation": parse_bool_frontmatter(
            frontmatter.get("disable-model-invocation"),
            default=False,
        ),
        "model": optional_frontmatter_str(frontmatter.get("model")),
        "argument_hint": optional_frontmatter_str(frontmatter.get("argument-hint")),
    }
