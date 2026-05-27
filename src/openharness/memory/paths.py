"""Paths for persistent project memory."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path

from openharness.config.paths import get_data_dir

"""用于计算并返回项目持久化记忆的存储路径，通过项目路径生成唯一目录，存放记忆文件，是整个 AI 长期记忆系统的路径管理核心。 

get_project_memory_dir：根据项目路径生成唯一记忆目录，自动创建文件夹并返回路径。
get_memory_entrypoint：返回项目记忆索引文件 MEMORY.md 的路径。

get_project_memory_dir：生成项目记忆根目录
get_memory_entrypoint：定位记忆索引文件
两个函数共同支撑长期记忆文件存储
总结
代码是记忆系统的路径管理器，负责定位记忆存在哪里。
"""

def get_project_memory_dir(cwd: str | Path) -> Path:
    """Return the persistent memory directory for a project."""
    path = Path(cwd).resolve()
    digest = sha1(str(path).encode("utf-8")).hexdigest()[:12]
    memory_dir = get_data_dir() / "memory" / f"{path.name}-{digest}"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


def get_memory_entrypoint(cwd: str | Path) -> Path:
    """Return the project memory entrypoint file."""
    return get_project_memory_dir(cwd) / "MEMORY.md"
