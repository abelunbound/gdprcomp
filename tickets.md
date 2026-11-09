# Tickets

Implementation order: Phases 1–5, then T36, Phases 6–7, T37 and T40, Phase 8, T38 and T39, then Phase 10.

## Phase 0 — Scaffold

- [x] **T01. Repo and package skeleton**

  Create repo layout: `src/gdpr_gap/` (package), `app/` (Dash), `experiments/`, `data/`, `configs/`, `prompts/`, `tests/`, `docs/`. Add `pyproject.toml`, lint/format config, `README.md` stub, `.gitignore`, `.env.example`.

  **Done when:** `pip install -e .` works and a trivial module imports.

- [x] **T02. CI pipeline**

  Add CI to run lint, type-check, and tests on push.

  **Done when:** CI runs green on the skeleton and fails on a deliberately broken test.

- [x] **T03. Config loading and validation**

  Central config loader (paths, model names, thresholds, weights, seeds) with schema validation and environment-variable overrides for secrets.

  **Done when:** invalid config raises a clear error; missing API key fails fast with a helpful message.

- [x] **T04. Logging and secrets hygiene**

  Structured logging with secret redaction; policy text logging off by default behind a research flag.

  **Done when:** redaction test passes and no secrets appear in logs.

## Phase 1 — Domain logic (pure, no models)

- [x] **T05. 22-criterion config**

  Single source of truth (`configs/criteria.yaml`): criterion ID, name, legal basis, condition sentence, weight, display order.

  **Done when:** all 22 load, validate, and a summary table renders from config alone.

- [x] **T06. Scoring module**

  Yes = 1, Somewhat = 0.5, No = 0; score = 100 × weighted sum ÷ sum of weights; bands (≥80, 50–79, <50); 220-point conversion; reject zero/negative total weight.

  **Done when:** unit tests cover all-Yes, all-No, mixed weights, invalid weights, band boundaries, 220-point conversion.

- [x] **T07. Gap label helper**

  Canonical `is_gap(verdict)` = verdict is not Yes; used everywhere metrics and examples are computed.

  **Done when:** one helper, used by metrics and UI, with tests.

## Phase 2 — Data acquisition and sampling

- [x] **T08. Princeton-Leuven corpus loader**

  Load policy records with snapshot date, URL, text, domain; validate required fields; cache locally.

  **Done when:** loader produces a typed dataframe/records object and fails clearly on missing corpus.

- [x] **T09. Gold-set sampler (~20 policies)**

  10 pre-2018 / 10 post-2018, preferably same websites at both dates; buffer around 25 May 2018; fixed seed; logged inclusion/exclusion reasons.

  **Done when:** rerunning with the same seed yields identical policy IDs and a written sampling manifest.

- [x] **T10. Descriptive 40+40 corpus sampler**

  Separate sampler for 40 pre / 40 post for descriptive analysis only; output manifest states “not validation unless labelled”.

  **Done when:** sampler is distinct from gold-set sampler and cannot be mistaken for validation input.

- [x] **T11. Text extraction and cleaning**

  Whole-page text from URL or pasted text; whitespace normalization; section boundaries preserved; extraction failures logged.

  **Done when:** tests cover HTML, pasted text, empty page, and very long policy.

- [x] **T12. Segmentation (comparators only)**

  ~120-word overlapping passages with configured overlap; never used to gate the whole-policy LLM.

  **Done when:** coverage test proves no tail text is dropped; docs state segmentation is comparator-only.

## Phase 3 — Model layer

- [x] **T13. LLM provider abstraction + structured output schema**

  Provider-agnostic client: model config, retries, timeout, token/cost logging, strict schema `{criterion_id, verdict, justification, evidence_quotes[]}`.

  **Done when:** malformed responses fail safely and are logged; one mocked and one real-provider path tested.

- [x] **T14. Embedding client (Legal-BERT, mean pooling)**

  Embedding interface with mean pooling and batch support; model name/version captured.

  **Done when:** embeddings are deterministic for fixed input/model and model version is logged.

- [ ] **T15. Cache layer**

  Cache LLM and embedding results keyed by policy hash + criterion + prompt version + model.

  **Done when:** cache hits/misses are logged and never keyed on mutable config.

- [ ] **T16. Prompt registry with versioning**

  Prompts stored as versioned files; prompt ID + version recorded in every result.

  **Done when:** changing a prompt creates a new version and old runs remain reproducible.

## Phase 4 — Main method: whole-policy LLM

- [ ] **T17. Whole-policy judge**

  Send full policy against each criterion; return verdict + justification + evidence quotes; no retrieval gating.

  **Done when:** any policy (including >2,500 words) yields 22 verdicts; no criterion is marked No for retrieval reasons.

- [ ] **T18. Guard against silent truncation**

  If a policy exceeds a configured model limit, raise or explicitly chunk with a logged warning — never silently drop text.

  **Done when:** test proves over-limit policies trigger the explicit path, not silent truncation.

## Phase 5 — Split feedback

- [ ] **T19. Separate feedback call**

  Second LLM call produces suggested fixes only for Somewhat/No criteria; never alters verdicts.

  **Done when:** disabling feedback leaves verdicts and scores byte-identical.

- [ ] **T20. Citation-support check**

  Feedback must cite quotes that actually exist in the policy; flag unsupported citations.

  **Done when:** tests cover valid and fabricated citations.

- [ ] **T21. Feedback-quality export**

  Export feedback with fields for rater correctness/actionability and citation-support status; no F1 for feedback.

  **Done when:** feedback export is separate from classification results.

## Phase 6 — Comparators

- [ ] **T22. Legal-BERT retrieval baseline**

  Cosine similarity between passage embeddings and criterion condition sentences; ranked passages; documented as topical relevance, not compliance.

  **Done when:** baseline outputs similarity scores and ranked passages with model version recorded.

- [ ] **T23. Threshold calibration on dev split**

  Replace hardcoded thresholds with values calibrated on expert-labelled dev policies; persist and version calibration config.

  **Done when:** calibration file is loaded at runtime and recorded in the manifest.

- [ ] **T24. Hybrid comparator**

  Retrieval top passages + LLM judge; no relevant passage → No without LLM call; log retrieval misses separately.

  **Done when:** hybrid runs on the same labelled policies as the main method and retrieval misses are measurable.

- [ ] **T25. Legal-BERT-only comparator**

  Similarity-only baseline with a distinct output schema labelled “baseline/retrieval signal”.

  **Done when:** its outputs cannot be confused with LLM verdicts in data or UI.

- [ ] **T26. Alternative-embedding hook (optional)**

  Config-swappable embedding model behind the comparator interface.

  **Done when:** one alternative model runs without code changes.

## Phase 7 — Orchestration and persistence

- [ ] **T27. Unified pipeline orchestrator**

  One entry point for app and batch with method selector: `whole_policy_llm` (default), `hybrid`, `legal_bert_only`.

  **Done when:** app and batch call the same orchestrator; method recorded in every result.

- [ ] **T28. Run manifest and result persistence**

  Persist per-policy results, per-criterion verdicts, evidence, feedback, tokens, cost, runtime, model versions, prompt versions, library versions, seed, corpus snapshot, run ID.

  **Done when:** a run can be fully reproduced from its manifest alone.

- [ ] **T29. Repeat-run consistency**

  N repeats per policy/method; compute agreement across repeats.

  **Done when:** repeat agreement is stored and reportable per criterion and overall.

## Phase 8 — Gold set and validation

- [ ] **T30. Annotation schema and tooling**

  Blind, double-rated, adjudicated labelling: policy ID, criterion ID, annotator ID, verdict, notes, timestamp, adjudicated verdict; import/export.

  **Done when:** two annotators can label independently without seeing each other; adjudication produces final labels.

- [ ] **T31. Manual-review time tracking**

  Record labelling time to derive the manual-review baseline.

  **Done when:** time entries roll up into a reported baseline used in “time saved”.

- [ ] **T32. Classification metrics**

  Per criterion and overall: gap precision/recall/F1, 3-class macro-F1, weighted Cohen’s kappa, confusion matrix.

  **Done when:** metrics reject inputs where predictions and labels come from the same system.

- [ ] **T33. Bootstrap confidence intervals**

  Resample policies (not judgements) for CIs with ~20 policies.

  **Done when:** CIs are reported and seed behaviour is tested.

- [ ] **T34. Method comparison report**

  Compare whole-policy LLM (main) vs hybrid vs Legal-BERT only (vs alternative embeddings) on the same labelled policies: accuracy, runtime, cost, repeat consistency.

  **Done when:** report names the winner per metric and does not assume hybrid is best.

- [ ] **T35. Error examples report**

  True/false positives and negatives per method with policy ID, criterion, gold label, prediction, supporting text.

  **Done when:** examples are exportable and linked from validation output.

## Phase 9 — Dash app

- [ ] **T36. Single-policy view**

  Default to whole-policy LLM; show score/100, 220-point score, band, per-criterion verdict, evidence, and separate feedback for Somewhat/No.

  **Done when:** UI separates “relevant text found” from “condition met” and shows the not-legal-advice notice.

- [ ] **T37. Corpus/results view**

  Per-policy scores, pre/post-2018 comparison, jurisdiction splits for the descriptive corpus; visibly labelled non-validated.

  **Done when:** gold-set validation and descriptive corpus are visually and structurally distinct.

- [ ] **T38. Validation view**

  Metrics, inter-rater agreement, confusion matrices, bootstrap CIs, method comparison — read only from persisted outputs.

  **Done when:** view cannot compute metrics on the fly from unvalidated data.

- [ ] **T39. Feedback-quality view**

  Rater correctness/actionability and citation-support status, kept separate from classification metrics.

  **Done when:** no feedback metric appears in F1/accuracy displays.

- [ ] **T40. Method selector and docs view**

  Research method selector plus docs rendering criteria table, scoring formula, prompts, run settings, limitations.

  **Done when:** docs render from config/prompt files, not duplicated hardcoded text.

## Phase 10 — Hardening and delivery

- [ ] **T41. Integration and smoke tests**

  End-to-end fixture run: sample → extract → judge (mocked LLM) → feedback → score → metrics → manifest.

  **Done when:** CI runs the full pipeline on a fixture and asserts expected scores.

- [ ] **T42. Reproducibility check**

  Re-run a fixture from its manifest with fixed seed and mocked/cached LLM; verdicts and scores must match.

  **Done when:** reproducibility test passes and is documented.

- [ ] **T43. Docs and final report**

  README + project doc: main method = whole-policy LLM; hybrid and Legal-BERT only as comparators; ~20-policy gold set; split classification/feedback; metric definitions; limitations; “LLM output is not ground truth”.

  **Done when:** docs match implemented behaviour and include the ground-truth and not-legal-advice statements.
