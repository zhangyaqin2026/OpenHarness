"""CLAUDE.md discovery and loading."""

from __future__ import annotations

from pathlib import Path

""" 查找并加载项目里的 CLAUDE.md 配置文件，把项目自定义规则、指令读取后拼接成提示词，让 AI 遵守项目专属要求，是 AI 适配项目规则的核心模块。

load_claude_md_prompt：最核心，生成项目规则提示词
discover_claude_md_files：负责找到配置文件
功能：让 AI遵守项目里的自定义指令   """

def discover_claude_md_files(cwd: str | Path) -> list[Path]:
    """Discover relevant CLAUDE.md instruction files from the cwd upward."""
    current = Path(cwd).resolve()
    results: list[Path] = []
    seen: set[Path] = set()

    for directory in [current, *current.parents]:
        for candidate in (
            directory / "CLAUDE.md",
            directory / ".claude" / "CLAUDE.md",
        ):
            if candidate.exists() and candidate not in seen:
                results.append(candidate)
                seen.add(candidate)

        rules_dir = directory / ".claude" / "rules"
        if rules_dir.is_dir():
            for rule in sorted(rules_dir.glob("*.md")):
                if rule not in seen:
                    results.append(rule)
                    seen.add(rule)

        if directory.parent == directory:
            break

    return results

"""!!!读取找到的文件内容，拼接成一段完整的 AI 提示词，供 AI 使用。"""
def load_claude_md_prompt(cwd: str | Path, *, max_chars_per_file: int = 12000) -> str | None:
    """Load discovered instruction files into one prompt section."""
    files = discover_claude_md_files(cwd)
    if not files:
        return None

    lines = ["# Project Instructions"]
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n...[truncated]..."
        lines.extend(["", f"## {path}", "```md", content.strip(), "```"])
    return "\n".join(lines)
