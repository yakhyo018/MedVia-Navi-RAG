# Status

_Last updated: 2026-11-07 — R1–R5 built and measured, nothing committed yet._

## Done

- **R1 Skeleton** — FastAPI app factory, `Settings` (`NAVI_`, `SecretStr`, 32+ char key), `X-API-Key`
  (constant-time), `/health/live` + `/health/ready`, Qdrant store with a dimension check,
  Dockerfile (non-root), Compose, GitHub Actions (ruff + pytest, no torch).
- **R2 Ingest** — `POST/GET/DELETE /v1/documents`, chunking by Markdown headings with the heading
  path, `multilingual-e5-small`, replace = delete + write, content hash with `CHUNKER_VERSION`.
  Corpus: 55 documents from MedVia's public GraphQL + 7 hand-written guides → 100 chunks.
- **R3 Query** — `POST /v1/query`: threshold 0.80 → refusal without the LLM, chat template,
  `[n]` citations checked, refusal in EN/RU/UZ/KK, one generation at a time.
- **R4 Dataset + LoRA** — 348 train / 87 test (EN, RU, UZ), split by document; LoRA r=16
  all-linear on Qwen2.5-0.5B-Instruct, 1 epoch, batch 1 × 8 on MPS. Adapter: `models/navi-lora`.
- **R5 Eval** — `reports/EVAL.md` (v2) and `reports/v1/` (before the dataset fix).

Checked: pytest 36 passed, ruff clean; the API run end to end with e5 + Qwen + LoRA on embedded
Qdrant (answer with `[1]`, refusal without the LLM, 401 without the key).
Not checked: Docker Compose (Docker Desktop was off); CI on GitHub (no remote yet).

## Open

- Uzbek refusal is weak (1 of 3 test questions) — the train set has few Uzbek negatives.
- Free-form questions are weaker than the template-heavy test set suggests.
- Seed contradiction: Mintleaf's description lists 3 interpreter languages, its field only UZ.
  The dataset skips Mintleaf's interpreter question; the fix belongs in MedVia's seed.
- The threshold cannot separate answerable from unanswerable (score overlap 0.77–0.92).

## Next ideas (not planned)

A reranker (cross-encoder) before the threshold; more Uzbek and free-form training questions;
a 1.5B model on a Colab T4; MedVia's batch calling `POST /v1/documents` nightly.
