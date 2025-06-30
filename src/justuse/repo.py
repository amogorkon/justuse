import asyncio
import inspect
import os
import sys
import threading
from abc import ABC, abstractmethod
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import sleep
from types import ModuleType

from git import Repo as GitPythonRepo
from pydantic import BaseModel, ConfigDict

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
        repo_name: str, path: str, ref: str = "main", subdir: str = None, **kwargs
    ):
        """
        Factory method to create a GitHubRepo instance. Accepts additional keyword arguments for flexibility.
        """
        return GitHubRepo(
            repo_name=repo_name, path=path, ref=ref, subdir=subdir, **kwargs
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
            GitPythonRepo.clone_from(
                url,
                self.local_path,
                branch=self.ref,
            )

    @property
    def git(self):
        return GitPythonRepo(self.local_path)

    @property
    def origin(self):
        return self.git.remotes.origin

    def sync(self):
        self.origin.fetch()

    def load_module(self) -> ModuleType | Exception:
        """
        Returns the module at self.path from the local repo clone.
        """
        mod_path = self.local_path / self.path
        mod_name = str(mod_path).replace("/", ".").replace("\\", ".").rstrip(".py")
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
        spec = spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            return ImportError(f"Could not load spec for {mod_name} at {mod_path}")
        mod = module_from_spec(spec)
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
                    self.origin.pull()
                    remote_commit_hash = self.origin.refs[self.ref].commit.hexsha
                    if last_commit_hash != remote_commit_hash:
                        self.origin.pull()
                        mod = self.load_module()
                        self.proxy._ProxyModule__implementation = mod
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
