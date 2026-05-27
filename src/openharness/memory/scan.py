"""Scan project memory files."""

from __future__ import annotations

from pathlib import Path

from openharness.memory.paths import get_project_memory_dir
from openharness.memory.types import MemoryHeader

"""项目记忆文件扫描工具，负责读取项目里的记忆文档，解析标题、描述、内容预览，按修改时间排序返回，
供 AI 快速检索和加载记忆，是记忆系统的核心读取模块。
 
 scan_memory_files：扫描记忆目录，读取所有.md 文件，过滤索引文件，解析并返回按时间排序的记忆列表。
_parse_memory_file：解析单个记忆文件，提取 YAML 头信息、标题、描述、内容预览，返回结构化数据。

scan_memory_files：AI 获取记忆的入口函数
_parse_memory_file：解析记忆文件的核心工具
总结:代码用于扫描、解析、整理项目记忆文件，让 AI 快速读取上下文。
 """
def scan_memory_files(cwd: str | Path, *, max_files: int = 50) -> list[MemoryHeader]:
    """Return memory headers sorted by newest first."""
    memory_dir = get_project_memory_dir(cwd)
    headers: list[MemoryHeader] = []
    for path in memory_dir.glob("*.md"):
        if path.name == "MEMORY.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        header = _parse_memory_file(path, text)
        headers.append(header)
    headers.sort(key=lambda item: item.modified_at, reverse=True)
    return headers[:max_files]


def _parse_memory_file(path: Path, content: str) -> MemoryHeader:
    """Parse a memory file, extracting YAML frontmatter when present."""
    lines = content.splitlines()
    title = path.stem
    description = ""
    memory_type = ""
    body_start = 0

    # Parse YAML frontmatter (--- ... ---)
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                for fm_line in lines[1:i]:
                    key, _, value = fm_line.partition(":")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if not value:
                        continue
                    if key == "name":
                        title = value
                    elif key == "description":
                        description = value
                    elif key == "type":
                        memory_type = value
                body_start = i + 1
                break

    # Fallback: first non-empty, non-frontmatter line as description
    desc_line_idx: int | None = None
    if not description:
        for idx, line in enumerate(lines[body_start:body_start + 10], body_start):
            stripped = line.strip()
            if stripped and stripped != "---" and not stripped.startswith("#"):
                description = stripped[:200]
                desc_line_idx = idx
                break

    # Build body preview from content after frontmatter, excluding the
    # line already used as description so search scoring stays consistent.
    body_lines = [
        line.strip()
        for idx, line in enumerate(lines[body_start:], body_start)
        if line.strip()
        and not line.strip().startswith("#")
        and idx != desc_line_idx
    ]
    body_preview = " ".join(body_lines)[:300]

    return MemoryHeader(
        path=path,
        title=title,
        description=description,
        modified_at=path.stat().st_mtime,
        memory_type=memory_type,
        body_preview=body_preview,
    )
