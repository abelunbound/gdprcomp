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

## Configuration

Non-secret settings are in `configs/config.yaml`. The API key is read from `ANTHROPIC_API_KEY` in `.env` when `model.provider` is `claude`. A missing key or an invalid file raises an error before any model call.

Logs are JSON. API keys are redacted. Policy text is omitted unless `log_policy_text` is true.

Legal-BERT comparators need the embeddings extra:

```bash
pip install -e ".[embeddings]"
```

## Using the page

Start the single-policy page from the repository root:

```bash
python -m app.single_policy
```

Open http://127.0.0.1:8050/ and paste a Markdown policy. There is no file picker. `data/corpus/markdown/745.md` is one policy you can paste.

**Judge** runs the whole-policy LLM, one criterion at a time, then writes feedback for every Somewhat or No. The status line above the results names the criterion in progress. The finished view shows the score out of 100, the 220-point score, the band, and, for each criterion, whether the condition was met, the quoted text, and feedback for gaps.

**Show sample** renders a fixed review and does not call the model.

**Show retrieval signal** runs Legal-BERT only. The result is labelled `baseline/retrieval signal` and lists a best similarity and ranked passages for each criterion. The first load reads the pinned model from `data/models`. Similarity is topical relevance. It is recorded separately from a Yes, Somewhat, or No verdict.

Hybrid, saved runs, and repeat agreement are the Python calls below.

## Phase 6 — Comparators

The orchestrator accepts three methods. `whole_policy_llm` is the default. `hybrid` and `legal_bert_only` are comparators. Every result records which method produced it. T26, a config switch for a second embedding model, is still open.

```python
from pathlib import Path

from gdpr_gap import (
    EmbeddingClient,
    LLMClient,
    PolicyDocument,
    build_transport,
    load_config,
    run_policy,
)
from gdpr_gap.embeddings import load_legal_bert

config = load_config()
text = Path("data/corpus/markdown/745.md").read_text(encoding="utf-8")
client = LLMClient(config, build_transport(config))
encoder = load_legal_bert(config.embedding, confirm=True)
embeddings = EmbeddingClient(config.embedding, encoder)

whole = run_policy(text, config, policy_id="745", client=client)
signal = run_policy(
    text,
    config,
    "legal_bert_only",
    policy_id="745",
    embeddings=embeddings,
)
print(whole.method, whole.review.score.score)
print(signal.method, signal.retrieval.signal)
```

`run_batch` scores several policies with one method. Policy ids in one batch must be unique. The snippets below keep using `config`, `text`, `client`, `embeddings`, and `PolicyDocument` from this example.

### Calibration and the hybrid method

`configs/config.yaml` leaves `thresholds.retrieval_similarity` empty. Hybrid will not run until a calibration file supplies that threshold. The fit uses a dev split of labelled similarities. The saved file records `label_source` as `model-produced`. No calibration file is shipped with the repository.

```python
from gdpr_gap import apply_calibration, load_calibration

calibrated = apply_calibration(config, load_calibration("configs/calibration"))
hybrid = run_policy(
    text,
    calibrated,
    "hybrid",
    policy_id="745",
    client=client,
    embeddings=embeddings,
)
print(hybrid.hybrid.threshold, hybrid.hybrid.retrieval_misses)
```

Hybrid embeds the policy, keeps passages at or above the threshold, and sends only those passages to the whole-policy prompt. A criterion with no passage at the threshold is No, the model is not called, and the skip is logged as `retrieval_miss`. `calibrate_retrieval_threshold` and `save_calibration` in `gdpr_gap.calibration` write `configs/calibration/vN.json`. An unchanged body keeps its version. A changed body writes the next file.

## Phase 7 — Runs, manifests, and repeat agreement

`record_run` calls the same orchestrator and writes `data/runs/{run-id}.json`. The file holds the run id, seed, method, model id, embedding revision, prompt text, library versions, Python version, and each policy's text, hash, runtime, verdicts, evidence, feedback, and token counts. Cost is null when no token price is configured. The API key is omitted. `data/` is gitignored.

```python
from gdpr_gap import load_run, record_run, reproduce, run_inputs

manifest = record_run(
    [PolicyDocument(policy_id="745", text=text)],
    config,
    client=client,
)
loaded = load_run(f"data/runs/{manifest.run_id}.json")
restored = reproduce(loaded)
again = run_inputs(loaded)
print(loaded.settings.method, restored[0].review.score.score, again[0].policy_id)
```

`record_repeats` scores each policy 2 to 20 times and writes `data/runs/repeats/{repeat-id}.json`. Agreement is pairwise: for each policy and criterion, every pair of repeats counts once, and those pairs are pooled. Whole-policy and hybrid agreement uses the verdict. Legal-BERT agreement uses the top passage and its similarity. `agreement_report` prints the overall figure and one line per criterion.

```python
from gdpr_gap import agreement_report, record_repeats

report = record_repeats(
    [PolicyDocument(policy_id="745", text=text)],
    config,
    repeats=2,
    client=client,
)
print(agreement_report(report))
```

Leave the result cache unset when the repeats should call the model again. Passing a cache returns the stored completion for the same prompt and model.

## Layout

- `src/gdpr_gap/` — installable package
- `app/` — Dash application
- `configs/` — criterion and run configuration
- `prompts/` — versioned prompt files
- `data/` — local cache, run manifests, and repeat reports; not committed
- `experiments/` — notebooks and one-off runs
- `tests/` — automated tests
- `docs/` — project documentation
