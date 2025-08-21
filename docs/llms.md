
# JustUse: LLM/Agent Reference

## Core Concepts

- `use()` loads modules from PyPI, Git, URL, Path.
- Modes: `reloading` (auto-reload), `reckless` (explicitely no hash/signature checks - checks are default).
- ZVIC: Zero-Version Interface Contract. Enforced for all auto-reloading modules.

## Security Matrix
| Source | Hash Required | Algorithms | On Failure |
|--------|--------------|------------|------------|
| URL    | Yes          | SHA256, BLAKE2s, JACK | SecurityError |
| Git    | Recommended  | SHA256, BLAKE2s | Warning |
| Path   | Optional     | SHA256, BLAKE2s | ValidationError |
| PyPI   | Recommended  | SHA256 (metadata) | Warning |

## Error Envelope (JSON)
```jsonc
{
  "error_id": "JU1003",
  "type": "HashMismatchError",
  "context": {"package": "secure_pkg", "requested_hash": "...", "actual_hash": "..."},
  "recovery_actions": [
    {"type": "command", "command": "use('secure_pkg', version='1.2.4')"}
  ]
}
```

## Recovery Actions
| Type | Description |
|------|-------------|
| `command` | Run a command (e.g., `use()` with params) |
| `suggest_contract_update` | Propose contract update |
| `fallback` | Use compatible version or stub |

## ZVIC (Zero-Version Interface Contract)

- Enforced for all auto-reloading modules (`modes=reloading`).
- Contract: Python dict, Pydantic model, or JSON/YAML describing API (function names, args, return types).
- Example:
  ```python
  contract = {
      'test_function': {'args': [], 'returns': str}
  }
  mod = use(Path('tests/test_module.py'), contract=contract, modes=reloading)
  mod2 = use('some_pypi_package', contract=contract, modes=reloading)
  mod3 = use(Repo.github('amogorkon/justuse', 'docs/demo.py'), contract=contract, modes=reloading)
  ```
- On import/reload: compare module interface (`inspect.signature`) to contract.
- If mismatch: raise `ContractMismatchError`.

### ContractMismatchError Example
```json
{
  "error_id": "JU4201",
  "type": "ContractMismatchError",
  "context": {
    "missing": ["test_function"],
    "signature_mismatches": {"foo": {"expected": "(x: int)", "actual": "(x)"}}
  },
  "recovery_actions": [
    {"type": "suggest_contract_update"},
    {"type": "fallback"}
  ]
}
````markdown

# JustUse: LLM / Agent Reference

This document documents the current state of the repository (Aug 2025) and how LLM-driven workflows, live reloading, and ZVIC (Zero-Version Interface Contracts) are applied in justuse.

## Short status
- The project now integrates the external `zvic` project to perform compatibility checks between module versions. The repository no longer includes a local `zvic` adapter — the published `zvic` package is used directly.
- `zvic`'s timeout/safety behavior for expensive checks is handled by the `zvic` release itself (the project owner added a baked-in timeout). justuse calls `zvic.compatibility.is_compatible(pre_mod, post_mod)` and relies on zvic for timeouts.
- A live simulation script and a pytest integration test were added to demonstrate and verify the gating behavior:
  - `scripts/live_simulation.py` — manual runnable demo (temporary git repo) that shows baseline → compatible → incompatible scenario.
  - `tests/integration/test_live_simulation.py` — integration test that reproduces the demo in CI-safe form.

## Key files changed / new
- `src/justuse/repo.py` — compatibility gating logic: `GitHubRepo._check_compatibility_between_commits` builds modules from commit blobs and calls `zvic.compatibility.is_compatible` for each changed `.py` file. If any file is incompatible the remote change is rejected.
- `src/justuse/main.py` — now contains `from __future__ import annotations` so zvic's import-time checks succeed on import.
- `src/justuse/zvic.py` — removed (do not rely on local adapter; use the published package).
- `scripts/live_simulation.py` — demo script that creates a temp git repo, commits baseline, compatible and incompatible changes, and shows proxy swap/reject behavior.
- `tests/integration/test_live_simulation.py` — pytest integration test covering the same flow.
- `pyproject.toml` — updated to accept the `zvic` release range and to pin the `crosshair-tool` package (avoid ambiguous upstream package named `crosshair`).

## How compatibility gating works now
- When a remote commit is detected, `GitHubRepo` computes changed `.py` files between the accepted baseline commit and the new commit.
- For every changed file it builds two `ModuleType` instances from the file contents at the two commits (no checkout required) and calls `zvic.compatibility.is_compatible(pre_mod, post_mod)`.
- `zvic` performs the signature and contract checks and enforces an internal timeout; if all changed modules are compatible, the project's `ProxyModule` implementation is swapped to the new module.

## Running the demo locally
With your project's virtualenv activated (PowerShell example):

```powershell
& .\venv\Scripts\Activate.ps1
py -3.12 -u scripts\live_simulation.py
```

This prints the baseline commit, the compatible commit acceptance, the incompatible commit rejection (with `zvic` diagnostics), and shows that the `ProxyModule` remains on the last accepted implementation.

## Running the integration test
Run the single integration test (it uses a temporary git repo and cleans up properly on Windows):

```powershell
py -3.12 -m pytest tests/integration/test_live_simulation.py -q
```

Or run the whole test suite:

```powershell
py -3.12 -m pytest -q
```

## Security and operational notes
- Building and executing module code from repository blobs means some code is executed in-process during compatibility checks. This has two risk vectors:
  1. Execution-time side-effects in module top-level code.
 2. Resource exhaustion or long-running analyses (mitigated by `zvic`'s internal timeout).

Recommendations:
- Prefer to keep repository code that will be used for compatibility checks free of dangerous top-level side-effects.
- Consider running `zvic` checks inside a separate process / sandbox if you need stronger isolation than the built-in timeout (future work; not currently required because `zvic` enforces a timeout).
- Make heavy analysis optional for CI (provide a `--heavy-analysis` flag or run heavy checks only in scheduled jobs) to keep developer turnaround fast.

## Troubleshooting
- If `pip` pulls an unrelated package named `crosshair`, install the intended package `crosshair-tool` and pin it in `pyproject.toml`:

```powershell
py -3.12 -m pip install --upgrade --force-reinstall crosshair-tool
```

- If `scripts/live_simulation.py` fails to remove temp files on Windows, ensure no external sync programs (Dropbox, OneDrive) hold locks on the temp directory; stop the sync client and re-run the test.

## Development notes and next improvements
- Convert internal helpers to a small public API (e.g., `GitHubRepo.check_compatibility(old, new) -> (bool, details)`) so external consumers don't rely on private methods.
- Add optional subprocess isolation for module loading and `zvic` invocation for stronger security guarantees.
- Persist compatibility diagnostics for rejected commits under `tests/artifacts/` (useful for debugging in CI).

## Example: minimal workflow (pseudo)
```python
from justuse.repo import GitHubRepo

gh = GitHubRepo(repo_name='owner/repo', path='pkg/module.py', ref='main', baseline_commit='...')
ok, details = gh._check_compatibility_between_commits(gh._accepted_commit, 'new_commit_sha')
if ok:
    new_mod = gh._module_from_commit('new_commit_sha', gh.path)
    gh.proxy._ProxyModule__implementation = new_mod
else:
    print('Rejected changes', details)
```

Replace private calls with a public wrapper if you want to use this from external tooling.

---

This file will be kept minimal and practical — if you want I can:
- Add a short CI workflow snippet that runs only the integration test on push to `unstable`.
- Promote `_check_compatibility_between_commits` to a public method and update tests to call it.

````
