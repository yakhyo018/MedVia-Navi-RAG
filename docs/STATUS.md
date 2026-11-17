# Status

_Last updated: 2026-11-17 — deployed to MedVia's production server; production `askNavi` answers._

## Production (2026-11-17, `feature/deploy` → main)

- Runs beside MedVia on its server (`/home/MedVia-project/medvia-navi`, `docker-compose.prod.yml`):
  `medvia-navi` (Qwen2.5-0.5B + LoRA, bfloat16 on the CPU, 3 GB cap) and `medvia-navi-qdrant`
  (v1.20, internal network). No published port; MedVia reaches it at `http://medvia-navi:8100`.
- Server: AMD EPYC 9354P, 2 vCPU, 7.7 GB. Measured: float32 10.9 s / 2.8 GB → bfloat16
  6.6 s (5.6 s warm) / 1.7 GB, same answer; refusals below the threshold ~0 s (no LLM).
- Index: 7 guides (`scripts/ingest_guides.py`, through the API) + 55 MedVia documents from
  `npm run navi:sync` in medvia-batch (nightly 03:30 after that).
- Git on the server: read-only deploy key `~/.ssh/medvia_navi`, SSH host `github-medvia-navi`.
  The LoRA adapter was copied by hand to `models/navi-lora` (sha256 checked).
- Checked from the MedVia API container: 401 without the key, an English question answered with
  its cited guide, port 8100 closed from outside. Free-form RU / UZ questions score below 0.80
  and are refused (the known weakness below), English ones are answered.
- Updates: `./deploy.sh` on the server (manual; CI's forced command knows only `api` / `web`).

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
