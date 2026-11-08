# MedVia Navi

RAG (Retrieval-Augmented Generation) service for MedVia: answers patients' questions from
MedVia's own content (health tips, FAQ, treatments, clinics, guides), cites sources, refuses
when the content does not hold the answer. Python 3.11, FastAPI, Qdrant, a small Qwen + LoRA.
Portfolio project, companion of `~/Desktop/MedVia` (NestJS). Started 2026-11-06.

## Read first

- `docs/STATUS.md` — at the start of every session: where it stands, the next step.
- `LEARN.md` — why each mechanism exists, trade-offs, interview Q&A. Update after every milestone.

## Origin

Modelled on `~/Desktop/Jaydari_RAG` (read-only reference), with its review findings fixed:
no server-path ingest, chat template instead of raw "Question:/Answer:", documents managed by id
(re-ingest removes old chunks), threshold instead of a keyword heuristic, tests, an evaluation.
Never copy `/ingest-path` or `_is_general_question`.

## Rules

- `src/navi/{api,core,services}`; routes thin, logic in services. App factory `create_app()`,
  nothing connects at import time.
- Settings only through `navi.core.config.Settings` (prefix `NAVI_`); secrets are `SecretStr`.
- Every route except `/health/*` needs `X-API-Key` (`require_api_key`). Navi has no users.
- Only `VectorStore` talks to Qdrant.
- Content is MedVia's synthetic demo data — never real patient data, never a patient's question
  stored in Qdrant.
- Not a medical device: never diagnose, never name a disease for a patient's symptoms.
- Model weights, Qdrant data and `.env` never go into git.

## Workflow

Same as MedVia: plan before editing, wait for a yes on anything non-trivial; explain as a code
review (problem → alternatives → code → interview questions); expand every abbreviation;
answer in Uzbek, code and comments in English. No package without saying which problem it solves.

## Validation

```bash
conda activate navi
pytest
ruff check . && ruff format --check .
```

## Git

`main` ← `develop` ← `feature/<milestone>`. Conventional commits. Stage files by name, read
`git diff --cached`, scan for secrets before every commit:
`git diff --cached | grep -nE "NAVI_API_KEY=[0-9a-f]{20,}|[0-9a-f]{64}|hf_[A-Za-z0-9]{20,}|sk-(ant|or)-[A-Za-z0-9_-]{10,}"`
— `[0-9a-f]{64}` catches the key pasted into `naviKey` of the Postman environment (git-native
Postman writes it into `postman/postman/environments/*.yaml`, which is committed).
