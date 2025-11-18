
# JustUse: LLM / Agent Reference

> **Version**: 2025.34 (Week 34, November 2025)
> **Python**: >=3.12
> **Repository**: [github.com/amogorkon/justuse](https://github.com/amogorkon/justuse)

## Quick Reference

### Core Concepts
- **`use()`** - Universal import function supporting PyPI, Git, URL, and Path sources
- **ProxyModule** - Transparent wrapper enabling hot reloading and aspect-oriented programming
- **ZVIC** - Zero-Version Interface Contracts for runtime compatibility checking
- **Modes** - Bitwise flags controlling behavior (auto_install, reloading, reckless, etc.)
- **CalVer** - Versioning scheme: `YYYY.0W[.patchx/devx/rcx]`

### Loading Sources
```python
from justuse import use, Repo, URL
from pathlib import Path

# PyPI with version check
use("numpy", version="1.21.0")

# Local path with hot reloading
use(Path("./mymodule.py"), modes=use.reloading)

# GitHub repository
use(Repo.github("owner/repo", "path/file.py"))

# Raw URL with hash
use(URL("https://example.com/module.py"), hashes={"abc123..."})
```

### Modes (Flag Enum)
Combine with bitwise OR (`|`):
- `auto_install` - Automatically install missing packages
- `reloading` - Enable hot reloading with ZVIC checks
- `recklessness` - Skip hash/signature validation (unsafe!)
- `fatal_exceptions` - Raise exceptions instead of warnings
- `verbose` - Enable detailed logging
- `DEFAULT` - Safe defaults

```python
mod = use("pkg", modes=use.auto_install | use.verbose)
```

---

## Security & Hash Verification

### Security Matrix
| Source | Hash Required | Algorithms | On Failure |
|--------|--------------|------------|------------|
| URL    | Yes          | SHA256, BLAKE2s, JACK | SecurityError |
| Git    | Recommended  | SHA256, BLAKE2s | Warning |
| Path   | Optional     | SHA256, BLAKE2s | ValidationError |
| PyPI   | Recommended  | SHA256 (metadata) | Warning |

### JACK Encoding
Compresses 64-char hexdigests to 18 characters using Japanese/ASCII/Chinese/Korean alphabets:
```python
from justuse.hash_alphabet import hexdigest_as_JACK
compact = hexdigest_as_JACK("abc123...")  # 64 chars → 18 chars
```

## ZVIC (Zero-Version Interface Contract)

ZVIC provides runtime compatibility verification based on callable signatures rather than version numbers. Implementation via external `zvic` package (>=2025.34).

### Compatibility Rules
```python
# ✅ COMPATIBLE changes:
def func(x: int) -> str: ...                      # baseline
def func(x: int, y: str = "default") -> str: ...  # added optional param
def func(x: int | float) -> str: ...              # widened type

# ❌ INCOMPATIBLE changes:
def func(x: int, y: str) -> str: ...  # baseline
def func(x: int) -> str: ...          # removed parameter
def func(username: str) -> str: ...   # changed parameter name
def func(x: int) -> str: ...          # narrowed type (from int|float)
```

### Usage
```python
from justuse import use, Repo
from pathlib import Path

# Local file with automatic reloading + ZVIC checks
mod = use(Path('mymodule.py'), modes=use.reloading)

# GitHub with hot reloading + ZVIC checks
repo = Repo.github("owner/repo", "module.py", ref="main")
mod = use(repo)
repo.reload_threaded()  # Background thread watches for compatible commits
```

When incompatibility detected:
```python
ok, details = gh.check_compatibility('old_sha', 'new_sha')
# ok = False
# details = {
#   "incompatible": [{
#     "file": "module.py",
#     "details": {"error": "Parameter 'name' removed from function 'greet'"}
#   }]
# }
```

---

## Project Status (Week 2025-34)
- **Version**: 2025.34 (CalVer: `YYYY.0W[.patchx/devx/rcx]`)
- **Python**: Requires Python >=3.12
- **Core Dependencies**: `zvic>=2025.34`, `GitPython>=3.1.30`, `pydantic>=2.8.2`, `crosshair-tool>=0.0.95`

## Key Features
- The project integrates the external `zvic` package (Zero-Version Interface Contracts) for runtime compatibility checking between module versions
- Hot reloading for both local and remote (GitHub) modules with signature compatibility validation
- Multi-source module loading: PyPI, Git, URL, local Path
- Inline version checking and secure hash verification
- Auto-installation with version isolation (multiple versions of same package)
- Aspect-oriented programming via ProxyModule pattern matching
- Structured error handling with recovery actions (RFC 7807-compatible)

## Repository Structure
```
src/justuse/
├── main.py              # Core use() implementation, Use class
├── repo.py              # GitHubRepo, hot reloading, compatibility gating
├── classes.py           # ProxyModule, ModuleReloader
├── pimp.py              # Package installation, version handling, signature checks
├── modutils.py          # Module building utilities
├── exceptions.py        # JustUseError hierarchy, structured errors
├── pydantics.py         # Pydantic models for validation
├── constants.py         # Modes, Hash enums
├── config.py            # Configuration management
├── messages.py          # User-facing messages
├── utils.py             # General utilities
├── hash_alphabet.py     # JACK encoding for compact hashes
└── templates/           # Jinja2 templates

scripts/
└── live_simulation.py   # Demo: baseline → compatible → incompatible changes

tests/
├── integration/
│   ├── test_live_simulation.py  # GitHubRepo compatibility gating tests
│   └── integration_test.py      # General integration tests
└── unit/                         # Unit tests for all components
```

## Key Files & Components
- `src/justuse/repo.py` — `GitHubRepo` class with compatibility gating: `_check_compatibility_between_commits()` builds modules from commit blobs and calls `zvic.is_compatible()` for each changed `.py` file. Incompatible changes are rejected.
- `src/justuse/main.py` — Core `use()` function and `Use` class handling all import sources
- `scripts/live_simulation.py` — Demo script: creates temp git repo, commits baseline/compatible/incompatible changes, shows proxy swap/reject behavior
- `tests/integration/test_live_simulation.py` — Integration test reproducing the demo flow
- `pyproject.toml` — Dependencies including `zvic>=2025.34` and `crosshair-tool>=0.0.95`

## ProxyModule & Aspect-Oriented Programming

### ProxyModule
All modules loaded via `use()` are wrapped in a `ProxyModule` that enables:
- Transparent hot reloading without changing references
- Aspect-oriented decoration via `@` operator
- Automatic forwarding of attribute access to implementation

```python
from justuse import use
from pathlib import Path

# ProxyModule wraps the actual module
mod = use(Path("mymodule.py"))
# mod.__class__ == ProxyModule
# mod.some_function() → forwards to actual implementation
```

### Aspectizing (Decoration)
Apply decorators to functions/methods matching patterns:

```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Decorate all functions
mod = use(Path("utils.py")) @ (use.isfunction, "*", log_calls)

# Decorate specific function by name
mod = use(Path("utils.py")) @ (use.isfunction, "process", log_calls)

# Decorate all methods in classes
mod = use(Path("classes.py")) @ (use.ismethod, "*", log_calls)
```

Pattern matching:
- `"*"` - matches all
- `"func_name"` - exact name match
- Predicate: `use.isfunction`, `use.ismethod`, `use.isclass`

## How Compatibility Gating Works
When a remote commit is detected:
1. `GitHubRepo._check_compatibility_between_commits()` computes changed `.py` files between the accepted baseline commit and the new commit
2. For each changed file, builds two `ModuleType` instances from file contents at both commits (no checkout required)
3. Calls `zvic.is_compatible(pre_mod, post_mod)` which performs signature and contract checks
4. If `zvic` raises `SignatureIncompatible`, the change is rejected with details
5. If all changed modules are compatible, the `ProxyModule` implementation is swapped to the new module
6. The `_accepted_commit` is updated to track the new baseline

**Public API**: `GitHubRepo.check_compatibility(old_commit, new_commit) -> (bool, details_dict)`

## Running Examples

### Live Simulation Demo
With your virtualenv activated (PowerShell):
```powershell
& .\venv\Scripts\Activate.ps1
python -m scripts.live_simulation
```

This demonstrates:
- Baseline commit creation
- Compatible change acceptance
- Incompatible change rejection (with `zvic` diagnostics)
- ProxyModule behavior (remains on last accepted implementation)

### Integration Tests
Run the compatibility gating test:
```powershell
python -m pytest tests/integration/test_live_simulation.py -v
```

Run all tests with coverage:
```powershell
python -m pytest --cov=justuse --cov-report=html
```

### Basic Usage Examples
```python
from justuse import use, Repo
from pathlib import Path

# Load from PyPI with version check
np = use("numpy", version="1.21.0")

# Load from local path with hot reloading
mod = use(Path("./mymodule.py"), modes=use.reloading)

# Load from GitHub with compatibility checking
repo = Repo.github("amogorkon/justuse", "tests/.tests/test_module.py")
mod = use(repo)
repo.reload_threaded()  # Enable background hot reloading

# Aspect-oriented decoration
mod = use(Path("./funcs.py")) @ (use.isfunction, "my_func", decorator)
```

## Security & Safety

### Compatibility Checking Risks
Building and executing module code from repository blobs during compatibility checks has two risk vectors:
1. **Execution-time side-effects**: Module top-level code runs during import
2. **Resource exhaustion**: Long-running or expensive analyses (mitigated by `zvic`'s internal timeout)

### Recommendations
- Keep repository code used for compatibility checks free of dangerous top-level side-effects
- For stronger isolation, consider running `zvic` checks in separate process/sandbox (future enhancement)
- Make heavy analysis optional for CI (use `--heavy-analysis` flag or scheduled jobs)

### Hash Verification
JustUse supports multiple hash algorithms:
- **URL sources**: Hash required (SHA256, BLAKE2s, JACK encoding)
- **Git sources**: Hash recommended (warning if missing)
- **Path sources**: Hash optional (ValidationError on mismatch)
- **PyPI sources**: Hash recommended (metadata SHA256, warning if missing)

JACK encoding compresses 64-char hexdigests to 18 characters using Japanese/ASCII/Chinese/Korean alphabets:
```python
from justuse.hash_alphabet import hexdigest_as_JACK
compact = hexdigest_as_JACK("abc123...")  # 64 chars → 18 chars
```

## Error Handling

All errors inherit from `JustUseError` (RFC 7807-compatible):
- `RepoPathNotFoundError` - Repository path not found
- `InstallationError` - Package installation failed
- `VersionError` - Version constraint violation
- `SecurityError` - Hash mismatch or validation failure
- `UnexpectedHash` - Hash verification failed
- Warnings: `VersionWarning`, `NoValidationWarning`, `AmbiguityWarning`

Errors include:
- `error_id`: Unique identifier (e.g., "JU4201")
- `context`: Detailed diagnostic information
- `recovery_actions`: Suggested fixes (commands, fallbacks)
- JSON serialization for agent/automation

Example:
```python
from justuse import use, JustUseError
try:
    mod = use("some_module", version="1.2.3")
except JustUseError as e:
    print(f"Error: {e.message}")
    print(f"Context: {e.context}")
    for action in e.recovery_actions:
        print(f"Recovery: {action}")
```

## Troubleshooting

### Dependency Issues
If `pip` installs wrong `crosshair` package:
```powershell
python -m pip install --upgrade --force-reinstall crosshair-tool
```

### Windows File Locks
If `scripts/live_simulation.py` fails to remove temp files, stop sync programs (Dropbox, OneDrive) that may hold locks on temp directories.

### Module Not Found
Check that Python version is >=3.12 and all dependencies are installed:
```powershell
python --version  # Should be 3.12+
python -m pip install -e ".[test]"
```

## Development Roadmap

### Completed (Week 2025-27)
- ✅ GitHub hot reloading for remote modules
- ✅ Compatibility gating with `zvic` integration
- ✅ Public API: `GitHubRepo.check_compatibility()`
- ✅ Live simulation demo and integration tests
- ✅ CalVer versioning (YYYY.0W)

### In Progress / Planned
- [ ] Subprocess isolation for module loading and `zvic` invocation (stronger security)
- [ ] Persist compatibility diagnostics for rejected commits (`tests/artifacts/`)
- [ ] P2P network for package distribution (before PyPI/conda)
- [ ] Birdseye debugger integration as mode
- [ ] Visual dependency graph representation
- [ ] Module-level variable guards ("module-properties")
- [ ] Slot-based plugin architecture
- [ ] Optional Cython compilation for annotated code

### Implementation Example
```python
from justuse.repo import GitHubRepo
from justuse.classes import ProxyModule

# Create GitHubRepo with baseline
gh = GitHubRepo(
    repo_name='owner/repo',
    path='pkg/module.py',
    ref='main',
    baseline_commit='abc123',
    proxy=ProxyModule(initial_module)
)

# Check compatibility between commits (public API)
ok, details = gh.check_compatibility('old_sha', 'new_sha')

if ok:
    # Load and swap implementation
    new_mod = gh._module_from_commit('new_sha', gh.path)
    gh.proxy._ProxyModule__implementation = new_mod
    gh._accepted_commit = 'new_sha'
else:
    print(f"Rejected changes: {details}")
    # details contains {"incompatible": [...], "errors": [...]}
```

## Contributing
See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. Join discussions on [Slack](https://join.slack.com/t/justuse/shared_invite/zt-tot4bhq9-_qIXBdeiRIfhoMjxu0EhFw).

### Testing
```powershell
# Run all tests
python -m pytest

# With coverage
python -m pytest --cov=justuse --cov-report=html

# Specific test file
python -m pytest tests/unit/test_signature_compatibility.py -v
```

### Code Style
- Black formatter
- Type hints (Python 3.12+)
- Docstrings for public APIs
- Pydantic models for validation

---

**Note**: This document is synchronized with repository state as of November 2025, version 2025.34. See `docs/scrum/` for weekly development logs and `CHANGELOG.md` for version history.