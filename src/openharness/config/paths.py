"""Path resolution for OpenHarness configuration and data directories.

Follows XDG-like conventions with ~/.openharness/ as the default base directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_BASE_DIR = ".openharness"
_CONFIG_FILE_NAME = "settings.json"

"""用于统一管理 OpenHarness 的配置、数据、日志、会话等文件路径，遵循固定目录规范，自动创建文件夹，提供全局 + 项目级的路径获取，
是系统读写文件的统一路径入口。

get_config_dir()：全局根目录，所有全局文件的基础
get_project_config_dir()：项目级目录核心，每个项目独立配置
所有函数自动创建文件夹，保证路径可用
该模块是系统读写文件的统一入口，所有配置 / 日志 / 会话都靠它找路径"""

"""获取全局配置目录，优先环境变量，默认～/.openharness/"""
def get_config_dir() -> Path:
    """Return the configuration directory, creating it if needed.

    Resolution order:
    1. OPENHARNESS_CONFIG_DIR environment variable
    2. ~/.openharness/
    """
    env_dir = os.environ.get("OPENHARNESS_CONFIG_DIR")
    if env_dir:
        config_dir = Path(env_dir)
    else:
        config_dir = Path.home() / _DEFAULT_BASE_DIR

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

"""获取主配置文件 settings.json 路径"""
def get_config_file_path() -> Path:
    """Return the path to the main settings file (~/.openharness/settings.json)."""
    return get_config_dir() / _CONFIG_FILE_NAME

"""获取数据存储目录（缓存、历史等）"""
def get_data_dir() -> Path:
    """Return the data directory for caches, history, etc.

    Resolution order:
    1. OPENHARNESS_DATA_DIR environment variable
    2. ~/.openharness/data/
    """
    env_dir = os.environ.get("OPENHARNESS_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        data_dir = get_config_dir() / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

"""获取日志目录"""
def get_logs_dir() -> Path:
    """Return the logs directory.

    Resolution order:
    1. OPENHARNESS_LOGS_DIR environment variable
    2. ~/.openharness/logs/
    """
    env_dir = os.environ.get("OPENHARNESS_LOGS_DIR")
    if env_dir:
        logs_dir = Path(env_dir)
    else:
        logs_dir = get_config_dir() / "logs"

    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

"""获取会话存储目录"""
def get_sessions_dir() -> Path:
    """Return the session storage directory."""
    sessions_dir = get_data_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir

"""获取后台任务输出目录"""
def get_tasks_dir() -> Path:
    """Return the background task output directory."""
    tasks_dir = get_data_dir() / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return tasks_dir

"""获取反馈存储目录"""
def get_feedback_dir() -> Path:
    """Return the feedback storage directory."""
    feedback_dir = get_data_dir() / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    return feedback_dir

"""获取反馈日志文件路径"""
def get_feedback_log_path() -> Path:
    """Return the feedback log file path."""
    return get_feedback_dir() / "feedback.log"


def get_cron_registry_path() -> Path:
    """Return the cron registry file path."""
    return get_data_dir() / "cron_jobs.json"

"""!!!获取当前项目专属的配置目录（项目内 /.openharness）"""
def get_project_config_dir(cwd: str | Path) -> Path:
    """Return the per-project .openharness directory."""
    project_dir = Path(cwd).resolve() / ".openharness"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_project_issue_file(cwd: str | Path) -> Path:
    """Return the per-project issue context file."""
    return get_project_config_dir(cwd) / "issue.md"


def get_project_pr_comments_file(cwd: str | Path) -> Path:
    """Return the per-project PR comments context file."""
    return get_project_config_dir(cwd) / "pr_comments.md"


def get_project_autopilot_dir(cwd: str | Path) -> Path:
    """Return the per-project autopilot state directory."""
    autopilot_dir = get_project_config_dir(cwd) / "autopilot"
    autopilot_dir.mkdir(parents=True, exist_ok=True)
    return autopilot_dir


def get_project_autopilot_registry_path(cwd: str | Path) -> Path:
    """Return the autopilot task registry path."""
    return get_project_autopilot_dir(cwd) / "registry.json"


def get_project_repo_journal_path(cwd: str | Path) -> Path:
    """Return the append-only repo journal path."""
    return get_project_autopilot_dir(cwd) / "repo_journal.jsonl"


def get_project_active_repo_context_path(cwd: str | Path) -> Path:
    """Return the synthesized active repo context path."""
    return get_project_autopilot_dir(cwd) / "active_repo_context.md"


def get_project_autopilot_policy_path(cwd: str | Path) -> Path:
    """Return the autopilot policy path."""
    return get_project_autopilot_dir(cwd) / "autopilot_policy.yaml"


def get_project_verification_policy_path(cwd: str | Path) -> Path:
    """Return the verification policy path."""
    return get_project_autopilot_dir(cwd) / "verification_policy.yaml"


def get_project_release_policy_path(cwd: str | Path) -> Path:
    """Return the release policy path."""
    return get_project_autopilot_dir(cwd) / "release_policy.yaml"


def get_project_autopilot_runs_dir(cwd: str | Path) -> Path:
    """Return the autopilot run artifacts directory."""
    runs_dir = get_project_autopilot_dir(cwd) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir
