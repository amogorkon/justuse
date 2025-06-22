# Error Handling

## JSON Error Handling for Agents & LLMs

JustUse supports structured, agent-friendly error output for real-time debugging, CI, and LLM workflows. JSON diagnostics are emitted via a dedicated logger, and the logger's configuration determines the output destination (stdout, file, etc.).

- Errors are output as structured JSON envelopes via a dedicated logger (e.g., `justuse.json` logger). The logger's configuration determines the output destination (stdout, file, etc.).
- Human-readable diagnostics are sent to `stderr` (for developers).
- Output mode is auto-detected (env/config/TTY). See llms.md for schema and details.

> **Note:**
> - Complex errors and tracebacks are displayed interactively in the browser using Brython for a better debugging experience.
> - Aspect-oriented programming supports module-level wrapping and browser-based dry-run/decorator selection UIs.

## 1. Error Hierarchy

```mermaid
classDiagram
    class JustUseError {
        +message: str
        +context: dict
        +timestamp: datetime
    }

    class ImportError {
        +source: str
        +attempted_paths: list
    }

    class VersionError {
        +expected: str
        +actual: str
        +package: str
    }

    class SecurityError {
        +source: str
        +threat_type: str
        +details: dict
    }

    class ConfigError {
        +setting: str
        +value: any
        +expected: str
    }

    class RegistryError {
        +operation: str
        +table: str
        +sql_error: str
    }

    class ReloadError {
        +module_path: str
        +reason: str
        +signature_diff: dict
    }

    JustUseError <|-- ImportError
    JustUseError <|-- VersionError
    JustUseError <|-- SecurityError
    JustUseError <|-- ConfigError
    JustUseError <|-- RegistryError
    JustUseError <|-- ReloadError

    VersionError <|-- VersionMismatchError
    VersionError <|-- VersionNotFoundError

    SecurityError <|-- HashMismatchError
    SecurityError <|-- UntrustedSourceError
    SecurityError <|-- CertificateError

    RegistryError <|-- DatabaseError
    RegistryError <|-- CorruptedRegistryError

    ReloadError <|-- SignatureMismatchError
    ReloadError <|-- NotReloadableError
```

## 2. Error Handling Strategies

| Strategy | Mode | Behavior | Use Case |
|----------|------|----------|----------|
| **Graceful Degradation** | Default | Return stub, warn | Development, optional deps |
| **Fast Fail** | `fastfail=True` | Immediate exception | CI/CD, strict environments |
| **Retry with Backoff** | Network errors | Exponential backoff | Unreliable networks |
| **Fallback Sources** | Multi-source | Try alternatives | High availability |

## 3. Warning System

### Warning Categories

| Category | Trigger | Default Action | Configuration |
|----------|---------|----------------|---------------|
| `VersionWarning` | Version mismatch | Log warning | Can escalate to error |
| `SecurityWarning` | Security concerns | Log warning | Should escalate to error |
| `DeprecationWarning` | Deprecated features | Log warning | Future error |
| `NotReloadableWarning` | Reload failure | Log warning | Can ignore |
| `PerformanceWarning` | Slow operations | Log warning | Can ignore |

### Warning Configuration

```python
import warnings

# Convert security warnings to errors
warnings.filterwarnings('error', category=use.SecurityWarning)

# Ignore performance warnings
warnings.filterwarnings('ignore', category=use.PerformanceWarning)

# Custom warning handler
def custom_warning_handler(message, category, filename, lineno):
    if category == use.NotReloadableWarning:
        print(f"Reload failed: {message}")
        # Optionally restart or fallback

warnings.showwarning = custom_warning_handler
```

## 4. Recovery Mechanisms

### Automatic Recovery

| Error Type | Recovery Strategy | Implementation |
|------------|------------------|----------------|
| **Registry Corruption** | Rebuild from artifacts | `registry.recreate()` |
| **Network Timeout** | Retry with backoff | `retry_count=3, backoff=2^n` |
| **Module Reload Failure** | Revert to previous | Keep function backup |
| **Hash Mismatch** | Try alternative source | Fallback URL list |

### Recovery APIs

```python
# Registry recovery
try:
    use.registry.validate()
except use.CorruptedRegistryError:
    backup_path = use.registry.backup()
    use.registry.recreate()
    print(f"Registry rebuilt, backup at {backup_path}")

# Module recovery
mod = use(use.Path('changing_module.py'), modes=use.reloading)
if not mod._reload_status.success:
    print("Reload failed, manual restart required")
    mod._revert_to_previous()

# Network recovery with fallback
sources = [
    use.URL('https://primary.com/mod.py'),
    use.URL('https://backup.com/mod.py'),
    use.Path('local_fallback/mod.py')
]

for source in sources:
    try:
        mod = use(source, timeout=10)
        break
    except (use.TimeoutError, use.SecurityError):
        continue
else:
    raise ImportError("All sources failed")
```

## 5. Debugging Support

### Debug Configuration

| Debug Level | Information | Performance Impact |
|-------------|-------------|-------------------|
| `ERROR` | Errors only | None |
| `WARNING` | Errors + warnings | Minimal |
| `INFO` | Basic operations | Low |
| `DEBUG` | Detailed tracing | Medium |
| `TRACE` | Full execution path | High |

### Diagnostic Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `use.diagnose()` | System health check | `use.diagnose().report()` |
| `use.registry.inspect()` | Registry status | `use.registry.inspect().summary()` |
| `use.status(pkg)` | Package information | `use.status('numpy')` |
| `use.trace_import(pkg)` | Import path tracing | `use.trace_import('scipy')` |

### Error Context Enhancement

```python
# Enhanced error reporting
try:
    mod = use('package', version='1.0.0')
except use.VersionMismatchError as e:
    print(f"""
    Version Conflict:
      Package: {e.package}
      Expected: {e.expected}
      Found: {e.actual}
      Source: {e.source}
      Registry State: {e.context['registry_state']}
      Suggested Fix: {e.context['suggestion']}
    """)

# Debug context manager
with use.debug_context(level='TRACE'):
    mod = use('complex_package')
    # Detailed tracing logged
```


# Features & Capabilities

| Category | Features |
|----------|----------|
| **Core** | Unified import API (`use()`), inline version checking, hash pinning & verification (SHA256/BLAKE2s/JACK), auto-installation (PyPI, conda, C-extensions), multi-version support, hot auto-reloading, initial module globals, aspect-oriented programming (recursive decoration), default fallbacks, ProxyModule abstraction, registry & audit (SQLite), modes & flags (auto_install, fastfail, fatal_exceptions, no_browser, etc.), no-browser mode |
| **Security** | HTTPS enforcement, audit logging, configurable security levels, no public installation mode, hash verification, signature compatibility (planned), isolation (planned), module-level variable guards (planned) |
| **Configuration** | Layered config: env vars, config file, runtime flags, per-import options; testing & debugging support |
| **Error Handling** | Hierarchical error types, recovery strategies (fallbacks, isolated envs, registry rebuild, revert on reload failure), warning escalation |
| **Observability** | Usage metrics, immutable audit logs, compliance support |
| **Testing & Quality** | Comprehensive test suite (unit, integration, security, performance), CI/CD integration, mock infrastructure |
| **Planned/Advanced** | Plugin/slot architecture, visual dependency graph, P2P sourcing, on-site compilation (Cython), sub-interpreter isolation, signature verification, module-level guards |

---
