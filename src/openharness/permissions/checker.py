"""Permission checking for tool execution."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass

from openharness.config.settings import PermissionSettings
from openharness.permissions.modes import PermissionMode

log = logging.getLogger(__name__)

# Paths that are always denied regardless of permission mode or user config.
# These protect high-value credential and key material from LLM-directed access
# (including via prompt injection).  Patterns use fnmatch syntax and are matched
# against the fully-resolved absolute path produced by the query engine.
"""最高优先级保护清单，永久禁止访问系统密钥、云凭证等敏感路径。"""
SENSITIVE_PATH_PATTERNS: tuple[str, ...] = (
    # SSH keys and config
    "*/.ssh/*",
    # AWS credentials
    "*/.aws/credentials",
    "*/.aws/config",
    # GCP credentials
    "*/.config/gcloud/*",
    # Azure credentials
    "*/.azure/*",
    # GPG keys
    "*/.gnupg/*",
    # Docker credentials
    "*/.docker/config.json",
    # Kubernetes credentials
    "*/.kube/config",
    # OpenHarness own credential stores
    "*/.openharness/credentials.json",
    "*/.openharness/copilot_auth.json",
)

"""LLM 工具调用的安全权限系统。永久禁止访问密钥等敏感路径；通过工具黑白名单、路径规则、命令黑名单控制执行；
根据权限模式决定是否允许或需用户确认。
PermissionChecker 是核心类，evaluate () 是核心判断方法，所有安全校验都由此完成，防止越权操作、提示注入风险，保障系统安全。

PermissionChecker.evaluate()：核心函数，所有权限判断都在这里执行
SENSITIVE_PATH_PATTERNS：最高优先级保护，强制禁止访问凭证
代码按敏感路径→工具黑名单→白名单→路径规则→权限模式顺序判断"""

"""权限检查结果类，记录是否允许、是否需要确认、拒绝原因。"""
@dataclass(frozen=True)
class PermissionDecision:
    """Result of checking whether a tool invocation may run."""

    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""

"""路径权限规则，存储路径匹配表达式 + 允许 / 拒绝。"""
@dataclass(frozen=True)
class PathRule:
    """A glob-based path permission rule."""

    pattern: str
    allow: bool  # True = allow, False = deny

"""核心权限检查器，初始化加载配置，evaluate是最重要方法，完成所有权限判断。"""
class PermissionChecker:
    """Evaluate tool usage against the configured permission mode and rules."""
    """加载配置，解析路径权限规则"""
    def __init__(self, settings: PermissionSettings) -> None:
        self._settings = settings
        # Parse path rules from settings
        self._path_rules: list[PathRule] = []
        for rule in getattr(settings, "path_rules", []):
            pattern = getattr(rule, "pattern", None) or (rule.get("pattern") if isinstance(rule, dict) else None)
            allow = getattr(rule, "allow", True) if not isinstance(rule, dict) else rule.get("allow", True)
            if isinstance(pattern, str) and pattern.strip():
                self._path_rules.append(PathRule(pattern=pattern.strip(), allow=allow))
            else:
                log.warning(
                    "Skipping path rule with missing, empty, or non-string 'pattern' field: %r",
                    rule,
                )
    """!!!evaluate()（总入口，所有权限判断都在这里）
     tool_name：工具名    is_read_only：是否只读操作     file_path：操作的文件路径
     command：执行的命令   逻辑：按敏感路径→工具黑名单→白名单→路径规则→命令黑名单→权限模式依次判断，返回结果。"""
    def evaluate(
        self,
        tool_name: str,
        *,
        is_read_only: bool,
        file_path: str | None = None,
        command: str | None = None,
    ) -> PermissionDecision:
        """Return whether the tool may run immediately."""
        # Built-in sensitive path protection — always active, cannot be
        # overridden by user settings or permission mode.  This is a
        # defence-in-depth measure against LLM-directed or prompt-injection
        # driven access to credential files.
        if file_path:
            for candidate_path in _policy_match_paths(file_path):
                for pattern in SENSITIVE_PATH_PATTERNS:
                    if fnmatch.fnmatch(candidate_path, pattern):
                        return PermissionDecision(
                            allowed=False,
                            reason=(
                                f"Access denied: {file_path} is a sensitive credential path "
                                f"(matched built-in pattern '{pattern}')"
                            ),
                        )

        # Explicit tool deny list
        if tool_name in self._settings.denied_tools:
            return PermissionDecision(allowed=False, reason=f"{tool_name} is explicitly denied")

        # Explicit tool allow list
        if tool_name in self._settings.allowed_tools:
            return PermissionDecision(allowed=True, reason=f"{tool_name} is explicitly allowed")

        # Check path-level rules
        if file_path and self._path_rules:
            for candidate_path in _policy_match_paths(file_path):
                for rule in self._path_rules:
                    if fnmatch.fnmatch(candidate_path, rule.pattern):
                        if not rule.allow:
                            return PermissionDecision(
                                allowed=False,
                                reason=f"Path {file_path} matches deny rule: {rule.pattern}",
                            )

        # Check command deny patterns (e.g. deny "rm -rf /")
        if command:
            for pattern in getattr(self._settings, "denied_commands", []):
                if isinstance(pattern, str) and fnmatch.fnmatch(command, pattern):
                    return PermissionDecision(
                        allowed=False,
                        reason=f"Command matches deny pattern: {pattern}",
                    )

        # Full auto: allow everything
        if self._settings.mode == PermissionMode.FULL_AUTO:
            return PermissionDecision(allowed=True, reason="Auto mode allows all tools")

        # Read-only tools always allowed
        if is_read_only:
            return PermissionDecision(allowed=True, reason="read-only tools are allowed")

        # Plan mode: block mutating tools
        if self._settings.mode == PermissionMode.PLAN:
            return PermissionDecision(
                allowed=False,
                reason="Plan mode blocks mutating tools until the user exits plan mode",
            )

        # Default mode: require confirmation for mutating tools
        bash_hint = _bash_permission_hint(command)
        reason = (
            "Mutating tools require user confirmation in default mode. "
            "Approve the prompt when asked, or run /permissions full_auto "
            "if you want to allow them for this session."
        )
        if bash_hint:
            reason = f"{reason} {bash_hint}"
        return PermissionDecision(
            allowed=False,
            requires_confirmation=True,
            reason=reason,
        )

"""处理文件路径格式，让权限匹配更准确。"""
def _policy_match_paths(file_path: str) -> tuple[str, ...]:
    """Return path forms that should participate in policy matching.

    Directory-scoped tools like ``grep`` and ``glob`` may operate on a root such
    as ``/home/user/.ssh``. Appending a trailing slash lets glob-style deny
    patterns like ``*/.ssh/*`` and ``/etc/*`` match the directory root itself.
    """
    normalized = file_path.rstrip("/")
    if not normalized:
        return (file_path,)
    return (normalized, normalized + "/")

"""对 npm/pip 安装等命令生成安全提示。"""
def _bash_permission_hint(command: str | None) -> str:
    if not command:
        return ""
    lowered = command.lower()
    install_markers = (
        "npm install",
        "pnpm install",
        "yarn install",
        "bun install",
        "pip install",
        "uv pip install",
        "poetry install",
        "cargo install",
        "create-next-app",
        "npm create ",
        "pnpm create ",
        "yarn create ",
        "bun create ",
        "npx create-",
        "npm init ",
        "pnpm init ",
        "yarn init ",
    )
    if any(marker in lowered for marker in install_markers):
        return (
            "Package installation and scaffolding commands change the workspace, "
            "so they will not run automatically in default mode."
        )
    return ""
