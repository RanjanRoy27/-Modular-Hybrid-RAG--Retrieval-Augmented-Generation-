# Contributing

Thanks for improving this project. Keep changes focused, easy to review, and documented when they affect setup or behavior.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. Set `RAG_API_KEY` when testing protected routes.

## Before Opening a Pull Request

Run:

```bash
python -m py_compile api.py core.py ingest.py main.py agent.py monitor.py reranker.py pipeline.py tests/test_api.py tests/test_llm.py
```

If the API is running, also run:

```bash
python tests/test_api.py
```

## Commit and PR Rules

- One logical change per commit.
- Use imperative commit messages.
- Keep `main` deployable.
- Update `CHANGELOG.md` for user-visible changes.
- Prefer squash or rebase merges to keep the GitHub history linear.

See [docs/git_workflow.md](docs/git_workflow.md) for the full workflow.

