"""Helpers for managing memory files."""

from __future__ import annotations

from pathlib import Path
from re import sub

from openharness.memory.paths import get_memory_entrypoint, get_project_memory_dir
from openharness.utils.file_lock import exclusive_file_lock
from openharness.utils.fs import atomic_write_text

"""OpenHarness 的记忆文件管理工具，提供查看、新增、删除项目记忆文件的功能，自动维护索引和文件锁，保证读写安全，
是 AI 记忆系统的文件操作核心。

add_memory_entry：添加记忆的核心方法
remove_memory_entry：删除记忆的核心方法
文件锁：保证多进程安全读写
总结
代码负责记忆文件的增删查，是 AI 长期记忆的文件管理层。
"""

"""获取记忆文件锁路径，保证并发安全"""
def _memory_lock_path(cwd: str | Path) -> Path:
    return get_project_memory_dir(cwd) / ".memory.lock"

"""列出所有记忆文件，供查询使用"""
def list_memory_files(cwd: str | Path) -> list[Path]:
    """List memory markdown files for the project."""
    memory_dir = get_project_memory_dir(cwd)
    return sorted(path for path in memory_dir.glob("*.md"))

"""!!!重点：新增记忆条目，自动生成文件并更新索引"""
def add_memory_entry(cwd: str | Path, title: str, content: str) -> Path:
    """Create a memory file and append it to MEMORY.md."""
    memory_dir = get_project_memory_dir(cwd)
    slug = sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_") or "memory"
    path = memory_dir / f"{slug}.md"
    with exclusive_file_lock(_memory_lock_path(cwd)):
        atomic_write_text(path, content.strip() + "\n")

        entrypoint = get_memory_entrypoint(cwd)
        existing = entrypoint.read_text(encoding="utf-8") if entrypoint.exists() else "# Memory Index\n"
        if path.name not in existing:
            existing = existing.rstrip() + f"\n- [{title}]({path.name})\n"
            atomic_write_text(entrypoint, existing)
    return path

"""!!!重点：删除记忆文件，并清理索引"""
def remove_memory_entry(cwd: str | Path, name: str) -> bool:
    """Delete a memory file and remove its index entry."""
    memory_dir = get_project_memory_dir(cwd)
    matches = [path for path in memory_dir.glob("*.md") if path.stem == name or path.name == name]
    if not matches:
        return False
    path = matches[0]
    with exclusive_file_lock(_memory_lock_path(cwd)):
        if path.exists():
            path.unlink()

        entrypoint = get_memory_entrypoint(cwd)
        if entrypoint.exists():
            lines = [
                line
                for line in entrypoint.read_text(encoding="utf-8").splitlines()
                if path.name not in line
            ]
            atomic_write_text(entrypoint, "\n".join(lines).rstrip() + "\n")
    return True
