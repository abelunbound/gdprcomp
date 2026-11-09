# GDPR gap evaluator

Research tool for scoring privacy-policy text against 22 UK GDPR criteria.

The main method is a whole-policy LLM judge. Hybrid retrieval and Legal-BERT are comparators. Work is tracked in `tickets.md`.

LLM output is not legal advice and is not ground truth.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

GitHub Actions runs these checks on every push and pull request:

```bash
ruff check src tests
ruff format --check src tests
mypy src tests
pytest
```

## Layout

- `src/gdpr_gap/` — installable package
- `app/` — Dash application
- `configs/` — criterion and run configuration
- `prompts/` — versioned prompt files
- `data/` — local cache, not committed
- `experiments/` — notebooks and one-off runs
- `tests/` — automated tests
- `docs/` — project documentation
