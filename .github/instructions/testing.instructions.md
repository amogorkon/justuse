# Testing Strategy for the Agent

This document outlines the testing strategy for the agent and the justuse project.


## Test Organization

- **Unit tests** are located in `tests/unit/` and are designed to test individual functions and components in isolation.
- **Integration tests** should be placed in `tests/integration/` and are intended to test the interaction between multiple components.
- **Work-in-progress tests** are in `tests/test.py` and are for features or fixes currently being developed. These tests are actively worked on and may be unstable.
- **Core quick tests** are in `tests/test_.py` and are a collection of very lightweight, fast tests (execution time < 1 sec) that should be run on every save to quickly check basic functionality.
- **TDD (Test-Driven Development) tests** are in `tests/tdd_test.py` and are a collection of tests that are known to fail, representing features or behaviors not yet implemented. This file is also used to check how fast a test runs.

## Pytest Configuration

- The `pytest.ini` file specifies which directories and files pytest should collect tests from.
- By default, `norecursedirs = tests/*` prevents pytest from automatically discovering tests in subdirectories. To run tests in subfolders (like `tests/unit`), you must specify them explicitly in `testpaths` or via the command line.


## Fixtures

- Shared fixtures should be defined in `/conftest.py` at the project root. Pytest will automatically discover and use these fixtures in your tests.

## Running Tests
- To run the core test: `pytest tests/test_.py`
- To run all tests (including subdirectories): `pytest tests tests/unit --maxfail=3 --disable-warnings`
- To run only unit tests: `pytest tests/unit`
- To run TDD tests: `pytest tests/tdd_test.py`

## What "do the tests" means
1. run the appropriate tests, usually the ones that are affected by the changes made, otherwise run all tests, with no warnings, stop after 3 fails.
2. after running, always try to fix the simplest failure first. do not ask for approval for simple fixes. If multiple failures occur, address them one at a time, starting with the simplest. if you are sure, fix it without asking for input or approval unless you already attempted to fix the same issue once without success. never try to fix multiple issues at once. if you fail to fix an issue on your second attempt, try to fix the next issue in the list.
3. back to 1 until all tests pass or you reach a point where you cannot fix any more failures.

## Best Practices

- Add new tests for every new feature or bugfix.
- Move tests from TDD to unit/integration as features are completed.
- Keep tests fast and isolated when possible.
- Use descriptive names for test functions and files.

## Continuous Integration

- Ensure all tests pass before merging changes.
- Address any warnings or errors reported by pytest.
