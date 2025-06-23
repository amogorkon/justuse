# Testing Strategy for the Agent

This document outlines the testing strategy for the agent and the justuse project.


## Test Organization

- **Unit tests** are located in `tests/unit/` and are designed to test individual functions and components in isolation.
- **Integration tests** (if any) should be placed in `tests/integration/` and are intended to test the interaction between multiple components.
- **Work-in-progress tests** are in `tests/test.py` and are for features or fixes currently being developed. These tests are actively worked on and may be unstable.
- **Core quick tests** are in `tests/test_.py` and are a collection of very lightweight, fast tests that should be run on every save to quickly check basic functionality.
- **TDD (Test-Driven Development) tests** are in `tests/tdd_test.py` and are a collection of tests that are known to fail, representing features or behaviors not yet implemented. This file is also used to check how fast a test runs: if a test is too slow, it should be moved to `tests/integration`.

## Pytest Configuration

- The `pytest.ini` file specifies which directories and files pytest should collect tests from.
- By default, `norecursedirs = tests/*` prevents pytest from automatically discovering tests in subdirectories. To run tests in subfolders (like `tests/unit`), you must specify them explicitly in `testpaths` or via the command line.


## Fixtures

- Shared fixtures should be defined in `/conftest.py` at the project root. Pytest will automatically discover and use these fixtures in your tests.

## Running Tests
- To run all tests (including subdirectories): `pytest tests tests/unit --maxfail=3 --disable-warnings`
- To run only unit tests: `pytest tests/unit`
- To run TDD tests: `pytest tests/tdd_test.py`
- after running, always try to fix the simplest failing test first, then run all tests again.

## Best Practices

- Add new tests for every new feature or bugfix.
- Move tests from TDD to unit/integration as features are completed.
- Keep tests fast and isolated when possible.
- Use descriptive names for test functions and files.

## Continuous Integration

- Ensure all tests pass before merging changes.
- Address any warnings or errors reported by pytest.

---

For more details, see the `pytest.ini` configuration and the test files themselves.
