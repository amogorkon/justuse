"""
Custom warnings and exceptions for justuse.
LLM/agent-friendly, RFC 7807-compatible error base class and hierarchy.
"""

import json
import os

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "error_registry.json")
try:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        ERROR_REGISTRY = json.load(f)
except Exception:
    ERROR_REGISTRY = {}


class JustUseError(Exception):
    def __init__(
        self,
        message=None,
        context=None,
        recovery_actions=None,
        error_id=None,
        severity=None,
        error_namespace=None,
        justuse_version=None,
        timestamp=None,
        **kwargs,
    ):
        from . import __version__
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recovery_actions = recovery_actions or []
        self.error_id = error_id or self.error_id
        self.severity = severity or self.severity
        self.error_namespace = error_namespace or self.error_namespace
        self.justuse_version = justuse_version if justuse_version is not None else __version__
        self.timestamp = timestamp
        for k, v in kwargs.items():
            setattr(self, k, v)

    """
    Base error for JustUse, RFC 7807 compatible, LLM/agent-friendly.
    """

    error_id = None
    type = None
    severity = "error"
    error_namespace = "JUSTUSE"

    def to_json(self):
        return json.dumps(
            {
                "error_id": self.error_id,
                "type": self.__class__.__name__,
                "severity": self.severity,
                "message": self.message,
                "context": self.context,
                "recovery_actions": self.recovery_actions,
                "error_namespace": self.error_namespace,
                "justuse_version": self.justuse_version,
                "timestamp": self.timestamp,
            },
            default=str,
        )

    def __str__(self):
        return f"[{self.error_id}]: {self.message} \nSuggestions how to recover: {self.recovery_actions}"

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.error_id} severity={self.severity} message={self.message!r}>"


class RepoPathNotFoundError(JustUseError):
    """
    Raised when a requested file/module path is not found in a remote repository clone.
    Agent-centric: includes diagnostics and recovery actions for branch, commit, and file presence issues.
    """

    error_id = "JU4102"
    type = "RepoPathError"
    severity = "error"
    error_namespace = "JUSTUSE_REPO"

    def __init__(
        self, repo=None, path=None, ref=None, agent_diagnostics=None, **kwargs
    ):
        message = f"Module path '{path}' not found in repository '{repo}' (ref='{ref}')"
        context = {
            "platform": "github",
            "repo": repo,
            "path": path,
            "ref": ref,
            "agent_diagnostics": agent_diagnostics or {},
        }
        recovery_actions = [
            {
                "type": "check_branch",
                "description": "Switch to the branch where the file exists (e.g., 'unstable').",
            },
            {
                "type": "browse_repo",
                "description": "Explore repository structure",
                "url": f"https://github.com/{repo}/tree/{ref}"
                if repo and ref
                else None,
            },
            {
                "type": "suggest_commit",
                "description": "Check if the file is present in the latest commit on the target branch.",
            },
            {
                "type": "suggest_path",
                "description": "Try alternative path or check diagnostics for correct file location.",
            },
        ]
        super().__init__(
            message=message,
            context=context,
            recovery_actions=recovery_actions,
            error_id=self.error_id,
            severity=self.severity,
            error_namespace=self.error_namespace,
            **kwargs,
        )


class NirvanaWarning(Warning, JustUseError): ...


class VersionWarning(Warning, JustUseError): ...


class NotReloadableWarning(Warning, JustUseError): ...


class NoValidationWarning(Warning, JustUseError): ...


class AmbiguityWarning(Warning, JustUseError): ...


class UnexpectedHash(ImportError, JustUseError):
    error_id = "JU2001"
    error_namespace = "JUSTUSE_SECURITY"
    ...


class InstallationError(ImportError, JustUseError):
    error_id = "JU3001"
    error_namespace = "JUSTUSE_INSTALL"
    ...


class VersionError(JustUseError):
    """Raised for version-related errors."""

    ...


class SecurityError(JustUseError):
    """Raised for security-related errors."""

    ...


class RegistryError(JustUseError):
    """Raised for registry/lookup errors."""

    ...


class ConfigError(JustUseError):
    """Raised for configuration errors."""

    ...


class DependencyError(JustUseError):
    """Raised for dependency resolution errors."""

    ...


class NetworkError(JustUseError):
    """Raised for network-related errors."""

    ...


class InternalError(JustUseError):
    """Raised for unexpected internal errors."""

    ...


class UserError(JustUseError):
    """Raised for user-caused errors."""

    ...
    error_id = "JU3001"
    error_namespace = "JUSTUSE_INSTALL"
    ...
