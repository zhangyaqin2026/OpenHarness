"""Scan project memory files."""

from __future__ import annotations

from pathlib import Path

from openharness.memory.paths import get_project_memory_dir
from openharness.memory.schema import (
    coerce_bool,
    coerce_int,
    coerce_optional_int,
    coerce_str_list,
    first_content_line,
    is_memory_expired,
    split_memory_file,
)
from openharness.memory.types import MemoryHeader

"""项目记忆文件扫描工具，负责读取项目里的记忆文档，解析标题、描述、内容预览，按修改时间排序返回，
供 AI 快速检索和加载记忆，是记忆系统的核心读取模块。

 scan_memory_files：扫描记忆目录，读取所有.md 文件，过滤索引文件，解析并返回按时间排序的记忆列表。
_parse_memory_file：解析单个记忆文件，提取 YAML 头信息、标题、描述、内容预览，返回结构化数据。

scan_memory_files：AI 获取记忆的入口函数
_parse_memory_file：解析记忆文件的核心工具
总结:代码用于扫描、解析、整理项目记忆文件，让 AI 快速读取上下文。
 """
def scan_memory_files(
    cwd: str | Path,
    *,
    max_files: int | None = 50,
    include_disabled: bool = False,
    include_expired: bool = False,
    memory_dir: str | Path | None = None,
) -> list[MemoryHeader]:
    """Return memory headers sorted by newest first."""
    memory_dir = Path(memory_dir) if memory_dir is not None else get_project_memory_dir(cwd)
    headers: list[MemoryHeader] = []
    for path in memory_dir.glob("*.md"):
        if path.name == "MEMORY.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        header = _parse_memory_file(path, text)
        if header.disabled and not include_disabled:
            continue
        if is_memory_expired(_metadata_from_header(header)) and not include_expired:
            continue
        headers.append(header)
    headers.sort(key=lambda item: item.modified_at, reverse=True)
    if max_files is None:
        return headers
    return headers[:max_files]


def _parse_memory_file(path: Path, content: str) -> MemoryHeader:
    """Parse a memory file, extracting YAML frontmatter when present."""
    metadata, body, _, _ = split_memory_file(content)
    lines = body.splitlines()
    title = path.stem
    description = ""
    memory_type = ""
    if metadata.get("name"):
        title = str(metadata["name"])
    if metadata.get("description"):
        description = str(metadata["description"])
    if metadata.get("type"):
        memory_type = str(metadata["type"])

    # Fallback: first non-empty, non-frontmatter line as description
    desc_line_idx: int | None = None
    if not description:
        fallback = first_content_line("\n".join(lines[:10]))
        if fallback:
            description = fallback
        for idx, line in enumerate(lines[:10]):
            stripped = line.strip()
            if stripped[:200] == description:
                desc_line_idx = idx
                break

    # Build body preview from content after frontmatter, excluding the
    # line already used as description so search scoring stays consistent.
    body_lines = [
        line.strip()
        for idx, line in enumerate(lines)
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
        id=str(metadata.get("id") or ""),
        schema_version=coerce_int(metadata.get("schema_version"), default=0),
        category=str(metadata.get("category") or ""),
        importance=coerce_int(metadata.get("importance"), default=0),
        source=str(metadata.get("source") or ""),
        signature=str(metadata.get("signature") or ""),
        created_at=str(metadata.get("created_at") or ""),
        updated_at=str(metadata.get("updated_at") or ""),
        ttl_days=coerce_optional_int(metadata.get("ttl_days")),
        disabled=coerce_bool(metadata.get("disabled"), default=False),
        supersedes=coerce_str_list(metadata.get("supersedes")),
        relative_path=path.name,
        tags=coerce_str_list(metadata.get("tags")),
        frontmatter=metadata,
    )


def _metadata_from_header(header: MemoryHeader) -> dict[str, object]:
    return {
        "created_at": header.created_at,
        "updated_at": header.updated_at,
        "ttl_days": header.ttl_days,
    }
