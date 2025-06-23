
# JustUse System Overview
# Features & Capabilities

| Category | Features |
|----------|----------|
| **Core** | Unified import API (`use()`), inline version checking, hash pinning & verification (SHA256/BLAKE2s/JACK), auto-installation (PyPI, conda, C-extensions), multi-version support, hot auto-reloading, initial module globals, aspect-oriented programming (recursive decoration and module-level wrapping), default fallbacks, ProxyModule abstraction, registry & audit (SQLite), modes & flags (auto_install, fastfail, fatal_exceptions, no_browser, etc.), no-browser mode |
| **Security** | HTTPS enforcement, audit logging, configurable security levels, no public installation mode, hash verification, signature compatibility (planned), isolation (planned), module-level variable guards (planned) |
| **Configuration** | Layered config: env vars, config file, runtime flags, per-import options; testing & debugging support |
| **Error Handling** | Hierarchical error types, recovery strategies (fallbacks, isolated envs, registry rebuild, revert on reload failure), warning escalation, Brython-based browser UI for complex errors |
| **Observability** | Usage metrics, immutable audit logs, compliance support |
| **Testing & Quality** | Comprehensive test suite (unit, integration, security, performance), CI/CD integration, mock infrastructure |
| **Planned/Advanced** | Plugin/slot architecture, visual dependency graph, P2P sourcing, on-site compilation (Cython), sub-interpreter isolation, signature verification, module-level guards |

---

## High Level

- Unified, secure, and modular Python module management for distributed and local applications
- Supports multi-source imports (PyPI, Git, URL, local), version pinning, hash verification, hot-reloading, and auditability
- Designed for extensibility, security, and observability in modern Python workflows

## Concepts

| Concept                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Unified Import         | Single `use()` entrypoint for all sources and modes                         |
| Registry               | SQLite-backed metadata store for installations, artifacts, hashes, and usage |
| ProxyModule            | Module wrapper enabling hot-reload, aspectizing, and advanced features      |
| Aspect-Oriented Wrapping | Module-level and callable-level aspect/decorator application, including dry-run and browser UI |
| Security Model         | Enforced hash verification, HTTPS, and audit logging                        |
| Configuration          | Layered: env vars, config file, runtime, per-import                         |
| Observability          | Immutable audit logs, usage tracking, and monitoring                        |
| Error Handling         | Hierarchical error types, warnings, and recovery mechanisms                 |
| Extensibility          | Pluggable sources, aspect-oriented programming, and plugin architecture     |

> **Note:** Complex errors and tracebacks are displayed interactively in the browser using Brython for a better human debugging experience. Aspect-oriented features include module-level wrapping and browser-based dry-run/decorator selection UIs, while agents and LLMs can parse structured JSON diagnostics for real-time debugging and compliance.

## Architecture

```mermaid
graph TB
    subgraph "User"
        U1[use() API]
    end
    subgraph "Core System"
        D1[Dispatcher]
        R1[Registry]
        P1[ProxyModule]
        S1[Security]
        C1[Configuration]
        E1[Error Handling]
    end
    subgraph "Sources"
        PYPI[PyPI]
        GIT[Git]
        URL[URL]
        PATH[Local Path]
    end
    U1 --> D1
    D1 --> R1
    D1 --> P1
    D1 --> S1
    D1 --> C1
    D1 --> E1
    D1 --> PYPI
    D1 --> GIT
    D1 --> URL
    D1 --> PATH
    P1 --> U1
```

## Security & Compliance

- All remote content must be hash-verified (SHA256/BLAKE2s/JACK)
- HTTPS enforced for all downloads; HTTP rejected by default
- Audit logs for all imports, hash mismatches, and version conflicts
- Configurable security levels (minimal, standard, paranoid)

## Configuration

| Layer         | Mechanism                | Example/Key |
|---------------|--------------------------|-------------|
| Environment   | Env vars (`JUSTUSE_HOME`) | `/custom/path` |
| File          | `config.toml`            | `[general]` section |
| Runtime       | (runtime config API)     | (see documentation)  |
| Per-import    | Function params          | `use('pkg', modes=reloading)` |

## Integration Patterns

```python
# Development: hot-reload, auto-install
mod = use(Path('my_module.py'), modes=reloading | auto_install)

# Production: strict, secure
secure_mod = use('package', version='1.0.0', hash_value='abc...', modes=fastfail)

# Security-conscious import
mod = use('secure_package', hash_value='verified_hash', modes=fastfail)
```

## Observability & Audit

- All actions (import, install, reload, error) are logged
- Registry tracks usage, versions, and hash history
- Audit trail supports compliance and troubleshooting

## Error Handling & Recovery

| Error Type         | Recovery Strategy                |
|--------------------|----------------------------------|
| Version Conflict   | Isolated environments, fallback  |
| Hash Mismatch      | Alternative sources, fail fast   |
| Reload Failure     | Revert to previous, warn user    |
| Registry Corruption| Rebuild from artifacts, backup   |

## JSON Error Handling for Agents & LLMs

JustUse supports structured, agent-friendly error output for real-time debugging, CI, and LLM workflows. JSON diagnostics are emitted via a dedicated logger, and the logger's configuration determines the output destination (stdout, file, etc.).

### Core Strategy

- Errors are output as structured JSON envelopes via a dedicated logger (e.g., `justuse.json` logger). The logger's configuration determines the output destination (stdout, file, etc.).
- Human-readable diagnostics are sent to `stderr` (for developers).
- Output mode is auto-detected via environment/config/TTY.

### JSON Error Envelope Example

```jsonc
{
  "error_id": "JU1003",
  "type": "HashMismatchError",
  "severity": "fatal",
  "message": "Downloaded artifact SHA256 ≠ expected hash.",
  "context": {
    "package": "secure_pkg",
    "requested_hash": "abc123...",
    "actual_hash": "def456...",
    "source": "PyPI"
  },
  "recovery_actions": [
    { "type": "command", "description": "Fallback to v1.2.4", "command": "use('secure_pkg', version='1.2.4')" },
    { "type": "config_patch", "description": "Update lockfile hash", "file": "justuse.lock", "path": "dependencies.secure_pkg.hash", "value": "def456..." }
  ],
  "error_namespace": "JUSTUSE_SECURITY",
  "justuse_version": "0.8.2",
  "timestamp": "2025-06-21T22:10:35Z"
}
```

### Detection Logic for JSON Mode

```python
def json_mode():
    if os.getenv("JUSTUSE_OUTPUT", "").lower() == "json":
        return True
    if "--output=json" in sys.argv:
        return True
    if "VSCODE_PID" in os.environ or "GITHUB_COPILOT" in os.environ:
        return True
    if not sys.stdout.isatty():
        return True
    return False
```

### Output Routing

- Human-readable diagnostics → `stderr`
- Machine-readable JSON → dedicated logger (default: `justuse.json` logger, configurable)
- Notify via `stderr` if JSON diagnostics are emitted

#### Example Output

```bash
>&2 echo "JustUse: Machine-readable error available in stdout"
echo '{"error_id":"JU1003","type":"HashMismatchError",...}'
```

### RFC 7807 Compatibility

```json
{
  "type": "https://justuse.dev/errors/hash-mismatch",
  "title": "Hash Mismatch",
  "status": 422,
  "detail": "Expected SHA256 abc… but got def…",
  "instance": "/imports/myapp#JU1003",
  "extensions": {
    "package": "secure_pkg",
    "retryable": false,
    "justuse_version": "0.8.2"
  }
}
```

### Architecture Overview

```mermaid
graph TD
    A[JustUse import attempt] --> B[Error raised]
    B --> C{json_mode()?}
    C -- true --> D[Emit structured JSON to stdout]
    C -- false --> E[Emit human message to stderr]
    D --> F[Agent (e.g. Copilot) parses and reacts]
    E --> G[Developer sees traceback or tip]
```

### Agent-Driven Error Lifecycle

```mermaid
graph LR
    I1[use("secure_pkg")] --> E1[attempt_download]
    E1 --> V1[verify_hash]
    V1 -->|Mismatch| ER1[HashMismatchError JSON emitted]
    ER1 --> AI1[Copilot Agent]
    AI1 --> A1[Suggest fallback version or alt mirror]
    AI1 --> A2[Rewrite import line inline]
    AI1 --> A3[Modify config file (justuse.lock)]
```

### Benefits for LLMs and Tools

- Errors are parsable and actionable
- Compatible with CI, editors, telemetry, and Copilot agents
- Supports agent planning: retries, rewrites, alternate strategies

---


**Current Status (June 2025):**
- `JustUseError` base class implemented with `.to_dict()` / `.to_json()` methods
- `emit_error()` now routes output based on `json_mode()` detection
- `suggested_actions` and `recovery_primitives` merged into `recovery_actions`
- All errors now include `error_namespace` and `justuse_version`
- Ambiguity warning detection and reporting is active and tested
- JSON diagnostics can be emitted to stdout or file (configurable)
- Formal error code registry (JU1000–JU5999) in use
- Copilot/agent workflows simulated in test suite
- CI pipeline runs lint, test, security scan, and coverage
- Mock infrastructure covers PyPI, file system, and network
- VS Code extension for diagnostics/quick-fix is in prototyping

**Next Steps:**
- Expand agent-driven recovery actions and quick-fix suggestions
- Broaden RFC 7807 compatibility and diagnostics export
- Enhance test coverage for edge cases and error recovery
- Finalize VS Code extension for public release


## Testing & Quality

- Unit, integration, security, and performance tests in place
- CI pipeline: lint, test, security scan, coverage (all active)
- Mock infrastructure for PyPI, file system, and network
- Ambiguity and error handling scenarios covered in tests


## See Also

- [Introduction](spec 1 - Introduction.md)
- [Architecture](spec 2 - Architecture.md)
- [Data Types](spec 3 - Data Types and Structures.md)
- [Workflows and Use Cases](spec 4 - Workflows and Use Cases.md)
- [Security](spec 5 - Security.md)
- [Configuration](spec 6 - Configuration.md)
- [Error Handling](spec 7 - Error Handling.md)
- [Testing](spec 8 - Testing.md)
- [Conclusion](spec 10 - Conclusion.md)
