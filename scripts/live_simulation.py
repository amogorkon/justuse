import tempfile
from pathlib import Path
import shutil
from git import Repo as GitPythonRepo

from justuse.repo import GitHubRepo
from justuse.classes import ProxyModule
from justuse.modutils import _build_mod


def write_file(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def commit_file(repo: GitPythonRepo, filepath: Path, message: str):
    repo.index.add([str(filepath.relative_to(repo.working_tree_dir))])
    return repo.index.commit(message)


def simulate():
    tmp = Path(tempfile.mkdtemp(prefix="justuse_live_sim_"))
    try:
        repo = GitPythonRepo.init(tmp)
        # minimal config so commits don't fail
        with repo.config_writer() as cw:
            cw.set_value("user", "name", "JustUse Test")
            cw.set_value("user", "email", "test@example.org")

        mod_path = tmp / "module.py"
        # baseline commit: simple greet with annotation str
        baseline_code = 'def greet(name: str) -> str:\n    return f"Hello {name}"\n'
        write_file(mod_path, baseline_code)
        c1 = commit_file(repo, mod_path, "baseline: add greet")
        # ensure branch name 'main'
        try:
            repo.git.branch("-M", "main")
        except Exception:
            pass

        print("baseline commit:", c1.hexsha)

        # Build initial module for ProxyModule
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

        print("GitHubRepo accepted baseline:", gh._accepted_commit)

        # Compatible change: implementation change only (same signature)
        compatible_code = 'def greet(name: str) -> str:\n    return f"Hi {name}, from v2"\n'
        write_file(mod_path, compatible_code)
        c2 = commit_file(repo, mod_path, "compatible: change greeting text")
        print("compatible commit:", c2.hexsha)

        ok, details = gh._check_compatibility_between_commits(c1.hexsha, c2.hexsha)
        print("compatibility check (baseline->compatible):", ok, details)
        if ok:
            mod_v2 = gh._module_from_commit(c2.hexsha, "module.py")
            gh.proxy._ProxyModule__implementation = mod_v2
            print("proxy.greet result:", gh.proxy.greet("Alice"))

        # Incompatible change: change annotation from str to int (should be incompatible)
        incompatible_code = 'def greet(name: int) -> str:\n    return f"Hello {name}"\n'
        write_file(mod_path, incompatible_code)
        c3 = commit_file(repo, mod_path, "incompatible: change param annotation")
        print("incompatible commit:", c3.hexsha)

        ok2, details2 = gh._check_compatibility_between_commits(c2.hexsha, c3.hexsha)
        print("compatibility check (v2->incompatible):", ok2, details2)
        print("proxy.greet still returns:", gh.proxy.greet("Bob"))

    finally:
        # clean up
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


if __name__ == "__main__":
    simulate()
