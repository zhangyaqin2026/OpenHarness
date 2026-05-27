"""Simple heuristic memory search."""

from __future__ import annotations

import re
from pathlib import Path

from openharness.memory.scan import scan_memory_files
from openharness.memory.types import MemoryHeader

"""简易记忆搜索工具，通过关键词匹配从项目记忆文件中找到相关内容。它会分词、计算匹配得分，优先展示标题 / 描述命中的结果，
是 AI 快速检索长期记忆的核心功能。
find_relevant_memories：根据查询词搜索记忆，标题描述权重更高，按得分 + 时间排序返回结果。
_tokenize：对搜索词分词，提取英文关键词和中文汉字，用于匹配检索。

find_relevant_memories：AI搜索记忆的唯一入口
_tokenize：分词工具，支撑搜索功能
总结:代码实现AI 长期记忆的关键词检索，让 Agent 能快速找到相关上下文。
 """

"""把查询词分词 → 遍历记忆文件 → 标题 / 描述匹配权重 ×2 → 正文匹配权重 ×1 → 算总分 → 按分数 + 时间排序 → 返回前 N 条。"""
def find_relevant_memories(
    query: str,# 输入：用户的搜索查询字符串（比如 "python 代码"）
    cwd: str | Path,# 输入：要搜索的文件夹路径（字符串 或 Path 对象）
    *,
    max_results: int = 5,# 最多返回几条结果，默认5条
) -> list[MemoryHeader]:
    """Return the memory files whose metadata and content overlap the query.

    Scoring weights frontmatter fields higher than body content so that
    well-annotated memories surface first.
    """
    tokens = _tokenize(query) # 第一步：把用户的查询词 拆分成 搜索关键词（token）
    if not tokens:
        return []

    # 准备一个列表，存放 (得分, 记忆头部) 的元组
    scored: list[tuple[float, MemoryHeader]] = []
    # 遍历文件夹里的记忆文件（最多扫描100个）
    # scan_memory_files 是外部函数：扫描目录，返回所有记忆文件的头部信息
    for header in scan_memory_files(cwd, max_files=100):

        # 把 标题 + 描述 拼接成字符串，并转小写（不区分大小写匹配）
        meta = f"{header.title} {header.description}".lower()
        body = header.body_preview.lower() # 正文预览内容，也转小写

        # Metadata matches are weighted 2x; body matches 1x.
        # 评分规则：元数据（标题/描述）匹配 权重 2 倍；正文匹配 权重 1 倍

        # 统计：多少个关键词出现在 标题/描述 里
        meta_hits = sum(1 for t in tokens if t in meta)
        body_hits = sum(1 for t in tokens if t in body)

        # 计算最终得分，只要有匹配（>0），就加入结果列表
        score = meta_hits * 2.0 + body_hits
        if score > 0:
            scored.append((score, header))

    # 排序：1. 按得分 从高到低 排序（-item[0]）
    # 2. 得分相同，按 修改时间 最新的 排前面（-item[1].modified_at）
    scored.sort(key=lambda item: (-item[0], -item[1].modified_at))

    # 取前 max_results 条，只返回 MemoryHeader，丢弃分数
    return [header for _, header in scored[:max_results]]

"""分词：把文本拆成搜索关键词:让搜索能同时匹配英文单词 + 单个汉字"""
def _tokenize(text: str) -> set[str]:
    """Extract search tokens from *text*, handling ASCII and Han ideographs."""
    # ASCII word tokens (3+ chars)
    """第一类：英文/数字/下划线 单词（长度≥3）; 转小写 → 正则提取所有字母数字组合 → 过滤掉长度<3的"""
    ascii_tokens = {t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) >= 3}

    # Han ideographs (each character carries independent meaning)
    # 第二类：中文汉字（每个汉字单独作为一个搜索词）
    # 正则匹配 Unicode 汉字范围
    han_chars = set(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    return ascii_tokens | han_chars






