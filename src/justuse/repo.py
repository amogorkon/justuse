import asyncio
import inspect
import os
import sys
import threading
from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePosixPath
from time import sleep
from types import ModuleType

from git import Repo as GitPythonRepo
from pydantic import BaseModel, ConfigDict
from zvic import SignatureIncompatible, is_compatible

from .classes import ProxyModule
from .config import home
from .exceptions import RepoPathNotFoundError
from .modutils import _build_mod


class Repo(ABC):
    """
    Abstract base class representing a repository.

    Attributes:
        repo (str): The repository name or URL.
        path (str): The file path within the repository.
        ref (str): The branch, tag, or commit SHA. Defaults to None.
        subdir (str): Optional subdirectory within the repository. Defaults to None.
    """

    @abstractmethod
    def url(self):
        """
        Abstract method to generate a URL for the repository.

        Returns:
            str: The URL of the repository.
        """
        pass

    @staticmethod
    def github(
        repo_name: str,
        path: str,
        ref: str = "main",
        subdir: str = None,
        baseline_commit: str = None,
        **kwargs,
    ):
        """
        Factory method to create a GitHubRepo instance. Accepts additional keyword arguments for flexibility.

        Args:
            repo_name (str): GitHub repo, e.g. 'amogorkon/justuse'
            path (str): File path in repo
            ref (str): Branch, tag, or commit (default 'main')
            subdir (str|None): Optional subdir
            baseline_commit (str|None): If set, checkout this commit after clone (for contract evolution baseline)
        """
        return GitHubRepo(
            repo_name=repo_name,
            path=path,
            ref=ref,
            subdir=subdir,
            baseline_commit=baseline_commit,
            **kwargs,
        )

    @abstractmethod
    def sync(self):
        """
        Abstract method to pull the latest changes from the repository.
        """
        pass

    @abstractmethod
    def load_module(self) -> ModuleType | Exception:
        """
        Abstract method to load the module from the repository.

        Returns:
            ModuleType | Exception: The loaded module or an exception if loading fails.
        """
        pass


class GitHubRepo(Repo, BaseModel):
    """
    Represents a GitHub repository. Now a Pydantic model for validation and flexibility.

    baseline_commit (str|None): If set, after cloning, checkout this commit (for contract evolution baseline)
    """

    repo_name: str
    "Repository name, e.g., 'amogorkon/justuse'"
    path: str
    "File path within the repository, e.g., 'docs/demo.py'"
    ref: str = "main"
    "Branch, tag, or commit SHA, default is main"
    subdir: str | None = None
    "Optional subdirectory within the repository"

    local_path: Path | None = None
    "Local path for the cloned repository"
    baseline_commit: str | None = None
    "Optional commit SHA or ref to checkout after cloning (for contract evolution baseline)"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    proxy: ProxyModule | None = None

    def __repr__(self):
        return f"<Repo: {self.repo_name}/{self.path}@{self.ref or 'HEAD'}>"

    def __init__(self, **data):
        if "local_path" not in data or data["local_path"] is None:
            repos_dir = home / "repos"
            repos_dir.mkdir(parents=True, exist_ok=True)
            # Encode branch/ref in the local folder name for branch coexistence, using -- as separator
            ref_part = data.get("ref", "main")
            repo_part = data["repo_name"].replace("/", "_")
            data["local_path"] = repos_dir / f"{repo_part}--{ref_part}"
        else:
            Path(data["local_path"]).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(**data)
        assert self.local_path is not None, "local_path should not be None!"
        if not self.local_path.exists():
            self.local_path.mkdir(parents=True, exist_ok=True)
            url = f"https://github.com/{self.repo_name}.git"
            repo = GitPythonRepo.clone_from(
                url,
                self.local_path,
                branch=self.ref,
            )
            # If baseline_commit is specified, checkout that commit
            if self.baseline_commit:
                repo.git.checkout(self.baseline_commit)
            # record accepted baseline commit: either baseline_commit (if present)
            # or the current HEAD of the branch we cloned.
            if self.baseline_commit:
                self._accepted_commit = self.baseline_commit
            else:
                # repo.head points at local HEAD (cloned branch)
                self._accepted_commit = repo.head.commit.hexsha
        else:
            # If repo already existed locally, ensure we have an accepted commit
            # set: default to current HEAD if nothing specified
            if not hasattr(self, "_accepted_commit"):
                try:
                    repo = GitPythonRepo(self.local_path)
                    self._accepted_commit = (
                        self.baseline_commit or repo.head.commit.hexsha
                    )
                except Exception:
                    self._accepted_commit = self.baseline_commit

    @property
    def git(self):
        try:
            return GitPythonRepo(self.local_path)
        except Exception:
            # Not a git repo - that's okay for local packages
            return None

    @property
    def origin(self):
        git_repo = self.git
        if git_repo is None:
            return None
        try:
            return git_repo.remotes.origin
        except Exception:
            return None

    def sync(self):
        """Fetch latest changes from remote. Safe to call even if not a git repo."""
        try:
            origin = self.origin
            if origin is not None:
                origin.fetch()
        except Exception:
            # Not a git repo or no remote - that's fine
            pass

    def _git(self) -> GitPythonRepo:
        return GitPythonRepo(self.local_path)

    def _changed_py_files_between(self, old_commit: str, new_commit: str) -> list[str]:
        """Return list of changed file paths (relative) between two commits."""
        repo = self._git()
        try:
            diff_text = repo.git.diff("--name-only", f"{old_commit}..{new_commit}")
        except Exception:
            return []
        files = [ln.strip() for ln in diff_text.splitlines() if ln.strip()]
        return [f for f in files if f.endswith(".py")]

    def _module_from_commit(self, commit_hash: str, file_path: str):
        """Build a ModuleType from the contents of file_path at commit_hash without
        checking out the working tree.
        Returns a ModuleType or raises an exception on failure.
        """
        repo = self._git()
        try:
            # git show returns file contents at commit
            content = repo.git.show(f"{commit_hash}:{file_path}")
        except Exception as e:
            raise ImportError(f"Could not read {file_path} at {commit_hash}: {e}")
        code_bytes = content.encode("utf-8")
        # create a stable module name derived from repo and commit
        mod_name = f"{self.repo_name.replace('/', '_')}_{commit_hash[:7]}_{PurePosixPath(file_path).stem}"
        # module_path param is used for diagnostics in _build_mod
        module_path = Path(self.local_path) / file_path
        return _build_mod(
            mod_name=mod_name,
            code=code_bytes,
            initial_globals={},
            module_path=module_path,
        )

    def _check_compatibility_between_commits(
        self, old_commit: str, new_commit: str
    ) -> tuple[bool, dict]:
        """Check all changed .py files between old_commit and new_commit for compatibility.

        Returns (True, {}) on success or (False, details) on failure.
        """
        details = {"incompatible": [], "errors": []}
        changed = self._changed_py_files_between(old_commit, new_commit)
        if not changed:
            return True, {}
        for fpath in changed:
            try:
                a = self._module_from_commit(old_commit, fpath)
            except Exception as e:
                details["errors"].append({"file": fpath, "error": str(e)})
                return False, details
            try:
                b = self._module_from_commit(new_commit, fpath)
            except Exception as e:
                details["errors"].append({"file": fpath, "error": str(e)})
                return False, details
            try:
                # Use zvic's compatibility check. SignatureIncompatible means
                # the modules are incompatible; other exceptions are treated
                # as errors in the check itself and returned in `details["errors"]`.
                is_compatible(a, b)
            except SignatureIncompatible as e:
                details["incompatible"].append({
                    "file": fpath,
                    "details": {"error": str(e)},
                })
                return False, details
            except Exception as e:
                details["errors"].append({"file": fpath, "error": str(e)})
                return False, details
        return True, {}

    def check_compatibility(
        self, old_commit: str, new_commit: str
    ) -> tuple[bool, dict]:
        """Public wrapper for compatibility checking between two commits.

        This method builds modules from the two commits for every changed .py file
        and delegates to the project's compatibility oracle (the external
        `zvic` package). The `zvic` compatibility implementation enforces
        its own internal timeout/safety guards, so callers can rely on this
        method to return in bounded time for common cases.

        Returns the same (bool, details) tuple as `_check_compatibility_between_commits`.
        """
        return self._check_compatibility_between_commits(old_commit, new_commit)

    def load_module(self) -> ModuleType | Exception:
        """
        Returns the module at self.path from the local repo clone.
        """
        mod_path = self.local_path / self.path
        
        # Python 3.14 fix: Create proper module name from path relative to repo root
        # Instead of using absolute path, use the relative path from self.path
        # This allows relative imports to work properly
        mod_name = self.path.replace("/", ".").replace("\\", ".").rstrip(".py")
        
        if not mod_path.exists():
            agent_diagnostics = {
                "cwd": os.getcwd(),
                "sys_path": list(sys.path),
                "mod_path": str(mod_path),
                "local_path_exists": self.local_path.exists(),
                "local_path": str(self.local_path),
                "call_stack": inspect.stack()[1:4],
            }
            return RepoPathNotFoundError(
                repo=self.repo_name,
                path=self.path,
                ref=self.ref,
                agent_diagnostics=agent_diagnostics,
            )
        
        # Add local_path to sys.path so relative imports work
        if str(self.local_path) not in sys.path:
            sys.path.insert(0, str(self.local_path))
            
        spec = spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            return ImportError(f"Could not load spec for {mod_name} at {mod_path}")
        mod = module_from_spec(spec)
        
        # Set __package__ for relative imports
        if "." in mod_name:
            mod.__package__ = ".".join(mod_name.split(".")[:-1])
            
        spec.loader.exec_module(mod)
        return mod

    def reload_threaded(self):
        """
        Start a background thread to watch for remote changes and reload the module when updates are detected.
        """

        def _reload_loop():
            last_commit_hash = self.origin.refs[self.ref].commit.hexsha

            while True:
                try:
                    # fetch remote and compare
                    self.origin.fetch()
                    remote_commit_hash = self.origin.refs[self.ref].commit.hexsha
                    if last_commit_hash != remote_commit_hash:
                        # Check compatibility between our accepted commit and remote
                        ok, details = self._check_compatibility_between_commits(
                            getattr(self, "_accepted_commit", last_commit_hash),
                            remote_commit_hash,
                        )
                        if not ok:
                            print(f"[GitHubRepo] Compatibility check failed: {details}")
                            # do not update implementation; just move on
                            last_commit_hash = remote_commit_hash
                            continue
                        # Build module from remote commit and swap
                        mod = self._module_from_commit(remote_commit_hash, self.path)
                        self.proxy._ProxyModule__implementation = mod
                        # advance accepted baseline to this new commit
                        self._accepted_commit = remote_commit_hash
                        last_commit_hash = remote_commit_hash
                except Exception as e:
                    print(f"[GitHubRepo] Error during reload polling: {e}")
                sleep(60)

        thread = threading.Thread(target=_reload_loop, daemon=True)
        thread.start()

    async def reload_async(self):
        last_commit_hash = self.origin.refs[self.ref].commit.hexsha
        while True:
            try:
                self.origin.fetch()
                remote_commit_hash = self.origin.refs[self.ref].commit.hexsha
                if last_commit_hash != remote_commit_hash:
                    self.origin.pull()
                    mod = self.load_module()
                    self.proxy._ProxyModule__implementation = mod
                    last_commit_hash = remote_commit_hash
            except Exception as e:
                print(f"Error during async polling: {e}")
            await asyncio.sleep(60)

    def url(self):
        # Generate a GitHub URL for the file
        base_url = f"https://github.com/{self.repo_name}/blob/{self.ref}/{self.path}"
        return f"{base_url}/{self.subdir}" if self.subdir else base_url
