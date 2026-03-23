---
name: Use uv for Python commands
description: Always use uv (not .venv/bin/python) for running Python and managing packages
type: feedback
---

Use `uv run python` for running Python commands and `uv add` for adding packages. Never use `.venv/bin/python` directly.

**Why:** User preference for consistent tooling with uv as the project's package manager.
**How to apply:** All Python invocations should use `uv run python ...`, all package installs should use `uv add ...`.
