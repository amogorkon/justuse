# Introduction

## 1. Purpose & Scope

**Purpose**: Technical specification for JustUse Python library - unified `use()` interface for flexible, secure, versioned module imports.

**Scope**: Core design, architecture, components, interfaces, data stores, error handling, extension points. For contributors, maintainers, advanced users.

## 2. Specification Overview

| Chapter | Title | Content |
|---------|-------|---------|
| **Spec 1** | **Introduction** | Purpose, scope, goals, requirements |
| **Spec 2** | **Architecture** | System design, components, workflows |
| **Spec 3** | **Data Types** | Models, schemas, APIs (authoritative source for all data types, schemas, and relationships) |
| **Spec 4** | **Integration** | Usage patterns, narrative examples, workflows, troubleshooting (all use-cases and examples are here) |
| **Spec 5** | **Security** | Security model, verification, threats |
| **Spec 6** | **Configuration** | Environment, config files, deployment |
| **Spec 7** | **Error Handling** | Hierarchy, warnings, recovery |
| **Spec 8** | **Testing** | Strategy, frameworks, CI |
| **Spec 10** | **Conclusion** | Summary, benefits, future work |

## 3. Goals & Requirements

### Core Goals
- Single `use()` function for multiple sources (PyPI, Git, URL, local)
- Version pinning + hash verification (SHA256/BLAKE2s/JACK)
- Signature pinning: in-code public key or fingerprint for signature verification
- Inline auto-installation (C-extensions, conda)
- Hot-reloading with signature checks
- Aspect-oriented programming (including module-level wrapping and browser-based dry-run/decorator selection)
- Per-session registry with usage metrics

### Requirements Matrix

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR1 | URL import with hash pinning | High | README |
| FR1a | URL import with signature pinning (in-code keys/fingerprints) | High | Security Model |
| FR2 | Auto-install with flags | High | Modes |
| FR3 | Multi-version isolation | Medium | README |
| FR4 | Hot-reload on file save | Medium | ModuleReloader |
| FR5 | Aspect decoration | Low | ProxyModule |
| FR6 | Global variable injection | Low | _use_path |

### Non-Functional Requirements

| Category | Requirements |
|----------|-------------|
| **Security** | HTTPS fetch, hash validation, signature pinning (in-code keys/fingerprints), fatal deprecation warnings |
| **Performance** | Minimal overhead, registry caching |
| **Usability** | Single entrypoint, familiar syntax |
| **Maintainability** | Modular design, centralized policy (buffet_table) |
| **Extensibility** | Pluggable sources (P2P, GitHub ZIP) |

## 4. Core Definitions

| Term | Definition |
|------|------------|
| **Use** | Main callable module entrypoint |
| **Signature Pinning** | In-code declaration of trusted public keys/fingerprints for signature verification |
| **ProxyModule** | Module wrapper for reload/aspectizing |
| **Artifact** | Single file imported via JustUse |
| **Installation** | Package installed in virtual environment |
| **Registry** | SQLite metadata store |
| **Modes/Flags** | Bitflags controlling behavior |
| **JACK** | Emoji/Unicode hash encoding |

## 5. Quick Reference

| Need | Location |
|------|----------|
| **Getting Started** | Sections 3-4, Workflows and Use Cases (Spec 4) |
| **System Design** | Architecture (Spec 2) |
| **Public APIs** | Data Types (Spec 3, authoritative) |
| **Usage Examples** | Workflows and Use Cases (Spec 4, all use-cases/examples) |
| **Security Model** | Security (Spec 5) |
| **Configuration** | Configuration (Spec 6) |
| **Error Handling** | Error Handling (Spec 7) |
| **Testing** | Testing (Spec 8) |

## 6. Error & Security Overview


**Error Handling**: UseError → VersionConflictError, HashMismatchError, etc. Complex errors and tracebacks are displayed interactively in the browser using Brython for a better debugging experience. *See Spec 7 for details.*

> **Note:** Complex errors and tracebacks are displayed interactively in the browser using Brython for a better debugging experience.


**Security**: Hash validation and signature pinning (in-code keys/fingerprints) are both supported. Signature pinning is performed by embedding trusted public keys or fingerprints directly in the `use()` call, providing strong, portable, and reproducible security. HTTPS enforcement and no browser launch by default. *See Spec 5 for details.*


**Configuration**: In-code pinning (hashes, public keys, or fingerprints) is the default for ad-hoc and portable code. External trust stores (e.g., config files or environment variables) are supported as an option for managed environments. `JUSTUSE_HOME`, `USE_VERSION` env vars, and `config.toml` are available for advanced configuration. *See Spec 6 for details.*


> **Note:**
> - All use-case patterns, narrative examples, and workflow descriptions are now found in **Spec 4 - Workflows and Use Cases**.
> - **Spec 3 - Data Types and Structures** is the single authoritative source for all data types, schemas, and relationships.
**Agent/LLM Support:**
- Errors can be output as structured JSON envelopes via a dedicated logger (e.g., `justuse.json` logger). The logger's configuration determines the output destination (stdout, file, etc.).
- Human-readable diagnostics are sent to `stderr` (for developers).
- Output mode is auto-detected (env/config/TTY). See llms.md for schema and details.

*End of Introduction*
## 10. Security Overview

* Validate all downloaded content via hash or signature before execution. Signature pinning uses in-code public keys or fingerprints for verification.

* Disable browser launching by default; configurable via `no_browser`.eloadableWarning`.


*For comprehensive security details, see Security (Spec 5).*

## 11. Configuration Overview.
* via `no_browser`.
* Environment variables: `JUSTUSE_HOME`, `USE_VERSION`
* `config.toml` in home directory controls debug levels, venv path, registry path.## 11. Deployment & Configuration


*For complete configuration details, see Configuration (Spec 6).*

**Environment variables:**

- `JUSTUSE_HOME`: override config home directory.
- `USE_VERSION`: pin library version for testing.
- `config.toml` in home directory controls debug levels, venv path, registry path.

---

*End of Introduction*

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

### Key Benefits
- **Discoverability**: IDE autocomplete shows all options (Repo.github, Repo.gitlab, etc.)
- **Conciseness**: Cleaner than enum-based approach
- **Platform-Specific Validation**: Each factory method can validate parameters
- **Consistency**: Matches JustUse's philosophy of explicit, type-based dispatch
- **Extensibility**: Easy to add new platforms (Repo.azure, Repo.gitea, etc.)
