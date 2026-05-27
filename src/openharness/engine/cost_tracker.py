"""Simple usage aggregation."""

from __future__ import annotations

from openharness.api.usage import UsageSnapshot

""" 会话期间的 AI 令牌用量统计与累计，简单轻量。 
  
  CostTracker累计整个会话的输入 / 输出 token 消耗。"""
class CostTracker:
    """Accumulate usage over the lifetime of a session."""

    def __init__(self) -> None:
        self._usage = UsageSnapshot()
    """累加新的 token 用量，更新总计数。"""
    def add(self, usage: UsageSnapshot) -> None:
        """Add a usage snapshot to the running total."""
        self._usage = UsageSnapshot(
            input_tokens=self._usage.input_tokens + usage.input_tokens,
            output_tokens=self._usage.output_tokens + usage.output_tokens,
        )

    @property
    def total(self) -> UsageSnapshot:
        """获取累计总消耗，返回汇总后的 token 数据。"""
        """Return the aggregated usage."""
        return self._usage
