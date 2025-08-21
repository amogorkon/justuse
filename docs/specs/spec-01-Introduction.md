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
