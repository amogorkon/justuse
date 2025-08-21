import tempfile
from pathlib import Path

from git import Repo as GitPythonRepo
import importlib
import gc

from justuse.repo import GitHubRepo
from justuse.classes import ProxyModule
from justuse.modutils import _build_mod


def write_file(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def commit_file(repo: GitPythonRepo, filepath: Path, message: str):
    repo.index.add([str(filepath.relative_to(repo.working_tree_dir))])
    return repo.index.commit(message)


def test_live_simulation_baseline_compatible_incompatible():
    # Create a temp repo and exercise GitHubRepo compatibility gating
    with tempfile.TemporaryDirectory(prefix="justuse_test_sim_") as td:
        tmp = Path(td)
        repo = GitPythonRepo.init(tmp)
        # set minimal config for commits
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "JustUse Test")
            cw.set_value("user", "email", "test@example.org")

        mod_path = tmp / "module.py"
        baseline_code = 'def greet(name: str) -> str:\n    return f"Hello {name}"\n'
        write_file(mod_path, baseline_code)
        c1 = commit_file(repo, mod_path, "baseline: add greet")
        # ensure branch name 'main' for compatibility with GitHubRepo default
        try:
            repo.git.branch("-M", "main")
        except Exception:
            pass

        # build initial module and create GitHubRepo wrapper pointing at local repo
        initial_mod = _build_mod(
            mod_name="live_module_base",
            code=baseline_code.encode("utf-8"),
            initial_globals={},
            module_path=mod_path,
        )

        gh = GitHubRepo(
            repo_name="local/sim",
            path="module.py",
            ref="main",
            local_path=tmp,
            baseline_commit=c1.hexsha,
            proxy=ProxyModule(initial_mod),
        )

        assert gh._accepted_commit == c1.hexsha

        # compatible change (same signature)
        compatible_code = 'def greet(name: str) -> str:\n    return f"Hi {name}, from v2"\n'
        write_file(mod_path, compatible_code)
        c2 = commit_file(repo, mod_path, "compatible: change greeting text")

        ok, details = gh._check_compatibility_between_commits(c1.hexsha, c2.hexsha)
        assert ok is True and details == {}

        # apply module from commit and verify proxy behavior
        mod_v2 = gh._module_from_commit(c2.hexsha, "module.py")
        gh.proxy._ProxyModule__implementation = mod_v2
        assert gh.proxy.greet("Alice") == "Hi Alice, from v2"

        # incompatible change (annotation changes from str -> int)
        incompatible_code = 'def greet(name: int) -> str:\n    return f"Hello {name}"\n'
        write_file(mod_path, incompatible_code)
        c3 = commit_file(repo, mod_path, "incompatible: change param annotation")

        ok2, details2 = gh._check_compatibility_between_commits(c2.hexsha, c3.hexsha)
        assert ok2 is False
        assert "incompatible" in details2
        # proxy should still be using last accepted implementation
        assert gh.proxy.greet("Bob") == "Hi Bob, from v2"

        # cleanup: close GitPython repo and drop references so Windows can remove tempdir
        try:
            repo.close()
        except Exception:
            pass
        # remove references to modules and repo objects
        try:
            del gh
            del mod_v2
            del initial_mod
        except Exception:
            pass
        importlib.invalidate_caches()
        gc.collect()
