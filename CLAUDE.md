# Mağaza Asistanı

Multi-agent assistant for retail store teams. Portfolio project; all data is synthetic.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`

## Layout
- `src/magaza_asistani/`: package code
- `tests/`: pytest tests

## Rules
- Python is pinned to 3.13. Manage dependencies with uv only.
- Code, identifiers, commit messages and PR text are in English. Comments and docstrings are in Turkish.
- New behaviour comes with tests. Lint, format and tests must pass before work is called done.
- Use `turkish_lower()` instead of `str.lower()` for Turkish text.
- Secrets live only in `.env`. Never read, print or commit its values.
- Never use real personal data.
- Commit, push, PR and merge decisions belong to the user: do them only on explicit approval.
