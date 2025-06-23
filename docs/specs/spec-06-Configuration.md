# Configuration

## 1. Configuration Hierarchy

```mermaid
graph TB
    ENV[Environment Variables] --> CONFIG[config.toml]
    CONFIG --> RUNTIME[Runtime Parameters]
    RUNTIME --> IMPORT[Per-Import Settings]

    ENV --> E1[JUSTUSE_HOME]
    ENV --> E2[JUSTUSE_DEBUG]
    ENV --> E3[USE_VERSION]

    CONFIG --> C1[Global Defaults]
    CONFIG --> C2[Security Settings]
    CONFIG --> C3[Path Configurations]

    RUNTIME --> R1[use.configure()]
    RUNTIME --> R2[Context Managers]

    IMPORT --> I1[Function Parameters]
    IMPORT --> I2[Mode Flags]
```

## 2. Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `JUSTUSE_HOME` | Config directory | `~/.justuse` | `/custom/path` |
| `USE_VERSION` | Pin library version | Latest | `1.0.0` |
| `JUSTUSE_DEBUG` | Enable debug logging | `False` | `True` |
| `JUSTUSE_REGISTRY_PATH` | Registry location | `{HOME}/registry.db` | `/tmp/reg.db` |
| `JUSTUSE_NO_BROWSER` | Disable browser | `True` | `False` |
| `JUSTUSE_VERIFY_SSL` | SSL verification | `True` | `False` |

## 3. Configuration File

**Location**: `{JUSTUSE_HOME}/config.toml`

```toml
[general]
debug_level = "INFO"          # DEBUG, INFO, WARNING, ERROR
auto_install = false          # Default auto-installation
fastfail = true              # Fail fast on errors
timeout = 30                 # Network timeout (seconds)

[paths]
venv_path = "~/.justuse/venvs"      # Virtual environments
registry_path = "~/.justuse/registry.db"  # Registry database
cache_path = "~/.justuse/cache"     # Download cache
temp_path = "/tmp/justuse"          # Temporary files

[security]
verify_ssl = true            # SSL certificate validation
allow_http = false          # Allow HTTP sources
require_hashes = true       # Mandatory hash verification
audit_level = "standard"    # none, basic, standard, full

[reloading]
enabled = true              # Enable hot-reloading
watch_mode = "auto"         # auto, threaded, async, disabled
debounce_ms = 100          # File change debounce
```

## 4. Runtime Configuration

### Global Configuration

| Method | Scope | Persistence | Example |
|--------|-------|-------------|---------|
| `use.configure()` | Session | Until changed | `use.configure(modes=auto_install)` |
| Context manager | Block | Temporary | `with use.configure(modes=fastfail): ...` |
| Function params | Single call | None | `use('pkg', modes=reloading)` |

### Configuration Examples

```python
# Development setup
use.configure(
    modes=reloading | auto_install,
    hash_algo=Hash.BLAKE2s,
    timeout=60,
    debug_level='DEBUG'
)

# Production setup
use.configure(
    modes=fastfail,
    require_hashes=True,
    auto_install=False,
    timeout=10,
    audit_level='full'
)

# Testing setup
use.configure(
    registry=":memory:",
    modes=fastfail,
    timeout=5,
    cache_duration=0
)
```

## 5. Deployment Scenarios

| Environment | Configuration Profile | Key Settings |
|-------------|----------------------|--------------|
| **Development** | Permissive, fast iteration | `reloading=True, auto_install=True, debug=True` |
| **Staging** | Moderate security | `require_hashes=True, auto_install=False, audit=standard` |
| **Production** | Maximum security | `fastfail=True, require_hashes=True, audit=full` |
| **CI/CD** | Reproducible, fast | `fastfail=True, timeout=5, cache=aggressive` |

### Deployment Configuration Scripts

```python
# Environment detection and auto-configuration
import os
import use

def configure_for_environment():
    env = os.getenv('DEPLOYMENT_ENV', 'development')

    configs = {
        'development': {
            'modes': reloading | auto_install,
            'debug_level': 'DEBUG',
            'timeout': 60
        },
        'production': {
            'modes': fastfail,
            'require_hashes': True,
            'auto_install': False,
            'audit_level': 'full'
        },
        'ci': {
            'modes': fastfail,
            'timeout': 5,
            'auto_install': False,
            'registry': ':memory:'
        }
    }

    use.configure(**configs.get(env, configs['development']))

# Auto-configure on import
configure_for_environment()
```

## 6. Configuration Validation

| Setting | Validation Rule | Error Type |
|---------|----------------|------------|
| `timeout` | `> 0 and < 3600` | ConfigurationError |
| `hash_algo` | Valid Hash enum | ValueError |
| `registry_path` | Writable directory | PermissionError |
| `modes` | Valid Mode flags | TypeError |

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
