# JustUse: LLM/Agent System Summary

## Features
| Category | Key Features |
|----------|-------------|
| Core | `use()` import API, multi-source (PyPI, Git, URL, Path), version pinning, hot-reload, audit |
| Security | Hash verification (SHA256/BLAKE2s/JACK), HTTPS only, signature pinning (planned/experimental), audit logging |
| Error Handling | Hierarchical error types, JSON error envelopes, agent-recoverable actions |
| Extensibility | Plugin/slot architecture, aspect-oriented wrapping, multi-version |

## Architecture
```mermaid
flowchart TD
  U[use()] --> D[Dispatcher]
  D -->|source| S[Security]
  D -->|source| R[Registry]
  D -->|source| P[ProxyModule]
  D -->|source| C[Config]
  D -->|source| E[ErrorHandling]
  D --> PYPI[PyPI]
  D --> GIT[Git]
  D --> URL[URL]
  D --> PATH[Path]
  P --> U
```

## Security Matrix
| Source   | Hash Required | Algorithms           | On Failure        |
|----------|--------------|----------------------|-------------------|
| URL      | Yes          | SHA256, BLAKE2s, JACK| SecurityError     |
| Git      | Recommended  | SHA256, BLAKE2s      | Warning           |
| Path     | Optional     | SHA256, BLAKE2s      | ValidationError   |
| PyPI     | Recommended  | SHA256 (metadata)    | Warning           |

## Error Envelope (JSON)
```jsonc
{
  "error_id": "JU1003",
  "type": "HashMismatchError",
  "context": {"package": "secure_pkg", "requested_hash": "...", "actual_hash": "..."},
  "recovery_actions": [
    {"type": "command", "command": "use('secure_pkg', version='1.2.4')"}
  ],
  "error_namespace": "JUSTUSE_SECURITY"
}
```
Schema: [Error Envelope JSON Schema](https://justuse.dev/schema/error-envelope.json)
Registry: [Error Code Registry](https://justuse.dev/errors/registry.json)

## Recovery Action Types
| Type                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `command`             | Execute a command (e.g., `use()` with params)   |
| `suggest_contract_update` | Propose updating a contract to match implementation |
| `fallback`            | Use a compatible version or stub               |

> **Note:** All error envelopes are sanitized to redact sensitive data. LLMs/agents can trust the context fields for automation and recovery actions.

## Git-Based Imports

### Proposed Syntax
```python
from justuse import Repo, use

# GitHub import
use(Repo.github("amogorkon/justuse", "docs/demo.py", ref="unstable"))

# GitLab with specific commit
use(Repo.gitlab("group/project", "module.py", ref="d3adb33f"))

# Generic Git repository
use(Repo.git("https://git.example.com/repo", "src/main.py", ref="main"))

# With security features
use(Repo.github("amogorkon/justuse", "src/core.py", ref="v1.0"),
    hash_value="sha256:abc123...",
    modes=security.paranoid)
```

### Implementation Workflow
```mermaid
graph TD
    U[use(Repo.github(...))] --> H[Git Handler]
    H --> P[Construct Clone URL]
    P --> C[Clone Repository]
    C --> V[Verify Ref]
    V -->|Valid| R[Resolve Path]
    R --> H[Hash Verification]
    H -->|Match| L[Load Module]
    H -->|Mismatch| E[JU1003 Error]
    V -->|Invalid| E2[JU4001 Error]
    R -->|Missing| E3[JU4102 Error]
```

### Key Benefits
| Benefit               | Description                                      |
|-----------------------|--------------------------------------------------|
| **Discoverability**   | IDE autocomplete shows all options (Repo.github, Repo.gitlab, etc.) |
| **Conciseness**       | Cleaner than enum-based approach                |
| **Validation**        | Platform-specific parameter validation          |
| **Consistency**       | Matches JustUse's philosophy of explicit, type-based dispatch |
| **Extensibility**     | Easy to add new platforms (Repo.azure, Repo.gitea, etc.) |

### Error Handling Example
```json
{
  "error_id": "JU4102",
  "type": "RepoPathError",
  "severity": "error",
  "message": "Module path not found in repository",
  "context": {
    "platform": "github",
    "repo": "amogorkon/justuse",
    "path": "missing.py",
    "ref": "unstable"
  },
  "recovery_actions": [
    {
      "type": "browse_repo",
      "description": "Explore repository structure",
      "url": "https://github.com/amogorkon/justuse/tree/unstable"
    },
    {
      "type": "suggest_path",
      "description": "Try alternative path",
      "command": "Repo.github('amogorkon/justuse', 'docs/demo.py')"
    }
  ]
}
```

## Signature-Driven Functional Programming (SDFP)

### Definition

> A runtime programming model where the behavior and lifecycle of functional code is governed by the structure of its function signatures, not external configuration or versions.

### Core Principles

* **First-class functions only**: modules should expose functions without internal state.
* **Signature-centric validation**: compatibility is enforced by comparing `inspect.signature()` results.
* **Hot reload enforcement**: modules can be reloaded safely if all signatures remain compatible.
* **Declarative, observable**: runtime code behavior is inspectable and predictable.

### Use Cases

* Shared logic across microservices
* Safe plugin/module reloads in dev & prod
* Real-time interface adaptation in agent-assisted systems

---

## Zero-Version Interface Contracts (ZVIC)

### Definition

> A strategy for managing code compatibility without version numbers, relying instead on runtime verification of callable structure.

### Core Principles

* **No semantic versioning required**
* **Interface stability through signature hashes**
* **Runtime fail-fast checks on API shape**
* **Inter-module contracts verified dynamically**

### Benefits

* No more dependency version drift
* Hashable, inspectable contracts
* Ideal for fast-moving, high-trust teams
