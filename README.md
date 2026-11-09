# MedVia Navi

A RAG (Retrieval-Augmented Generation) service for [MedVia](https://github.com/yakhyo018/MedVia):
foreign patients ask about treatments, visits and aftercare in Seoul; Navi answers **only from
MedVia's own content**, cites its sources `[1] [2]`, and says "I don't know" when the content
does not hold the answer. A small Qwen model is fine-tuned with LoRA (Low-Rank Adaptation) for
exactly that behaviour, and the base and fine-tuned models are compared on a held-out set.

Not a medical device: Navi never diagnoses, and it only ever reads MedVia's synthetic demo content.

| Milestone | | Status |
| --- | --- | --- |
| R1 | Skeleton — FastAPI, settings, API key, health probes, Qdrant, Docker, CI | ✅ |
| R2 | Ingest — 62 documents → 100 chunks, heading-aware chunking, multilingual-e5, re-ingest | ✅ |
| R3 | Query — retrieval threshold, chat template, `[n]` citations, refusal in 4 languages | ✅ |
| R4 | Dataset (348 train / 87 test) + LoRA on Qwen2.5-0.5B-Instruct, Apple M2 8 GB | ✅ |
| R5 | Evaluation — retrieval metrics, threshold grid, base vs LoRA | ✅ |

## Results (87 held-out questions — EN 37, RU 30, UZ 20)

| Metric | Qwen2.5-0.5B-Instruct | + Navi LoRA |
| --- | ---: | ---: |
| Answer accuracy (key fact) | 76.8 % | 76.8 % |
| Grounded citation (cites the source that holds the fact) | 8.7 % | **84.1 %** |
| Answer in the question's language | 59.4 % | **98.6 %** |
| Refusal accuracy (unanswerable questions) | 22.2 % | **83.3 %** |
| False refusal (the source was there) | 1.5 % | 9.0 % |
| Mean latency (M2, MPS) | 597 ms | 532 ms |

Retrieval (multilingual-e5-small, top 4): Hit@1 95.7 %, Hit@4 100 %, MRR@10 0.978.
Full report: [`reports/EVAL.md`](reports/EVAL.md); the first run, before the dataset fix, in
[`reports/v1/`](reports/v1/EVAL.md).

**What the numbers do not say.** The test set is small (18 unanswerable questions — one
question is 5.6 points) and two thirds of it are template questions shaped like the training
data. Free-form questions through the API are visibly weaker: a wrong fact from the right
source, Uzbek answers that copy a template phrase. LoRA taught the *format* (cite, language,
refuse); a 0.5B model is still a 0.5B model. See [`docs/REPORT.uz.md`](docs/REPORT.uz.md).

## Architecture

```
MedVia GraphQL ──export_medvia.py──▶ data/corpus ──POST /v1/documents──▶ chunker ─▶ e5 ─▶ Qdrant
                                                                                        │
question ──POST /v1/query──▶ e5 (query:) ─▶ Qdrant top 4 ─▶ score ≥ 0.80? ──no──▶ refusal (no LLM)
                                                                  │yes
                                                      chat template + sources ─▶ Qwen + LoRA
                                                                  │
                                              answer + [n] checked against the sources
```

## Deploy (production, next to MedVia)

Runs on MedVia's server beside MedVia's own containers — `docker-compose.prod.yml`:

```
medvia-api / medvia-batch ──medvia_monorepo-network──▶ medvia-navi:8100 ──navi-internal──▶ medvia-navi-qdrant
```

- **No published port.** Only MedVia's API (`askNavi`) and batch (`NAVI_SYNC`) reach Navi, at
  `http://medvia-navi:8100`; Qdrant is on an `internal` network only Navi joins.
- **Image:** `python:3.11-slim` + CPU-only torch + the `[serve]` extra (no training tools).
  e5 and Qwen are downloaded on the first start into the `navi-hf-cache` volume.
- **Model:** Qwen2.5-0.5B-Instruct + the LoRA adapter, on the CPU. The adapter is not in git:
  it is copied once to `models/navi-lora` on the server and mounted read-only.
- **Limits:** 4 GB memory, half the CPU weight of MedVia's containers, `OMP_NUM_THREADS=2`.

First time (on the server, in `/home/MedVia-project`):

```bash
git clone git@github-medvia-navi:yakhyo018/MedVia-Navi-RAG.git medvia-navi
printf 'NAVI_API_KEY=%s\n' "$(openssl rand -hex 32)" > medvia-navi/.env && chmod 600 medvia-navi/.env
# from the laptop:  scp -r models/navi-lora root@<server>:/home/MedVia-project/medvia-navi/models/
cd medvia-navi && ./deploy.sh
docker exec medvia-navi python scripts/ingest_guides.py   # the 7 guides
docker exec medvia-batch npm run navi:sync                # MedVia's content, from production
```

Then MedVia's `.env` gets `NAVI_URL=http://medvia-navi:8100` and the same `NAVI_API_KEY`, and
the API and batch are recreated. Updates: `./deploy.sh` (pull main, rebuild, recreate).
`data/corpus/medvia.jsonl` is a development export and is never ingested in production.

## Run

```bash
conda create -n navi python=3.11 && conda activate navi
pip install -e '.[dev,ml]'
cp .env.example .env            # set NAVI_API_KEY (openssl rand -hex 32)
docker compose up -d qdrant
navi-api                        # http://127.0.0.1:8100/docs
```

Offline pipeline (embedded Qdrant in `qdrant_local/` — one process at a time):

```bash
python scripts/export_medvia.py      # MedVia's public GraphQL → data/corpus/medvia.jsonl
python scripts/ingest_corpus.py      # 62 docs → 100 chunks
python scripts/build_dataset.py      # data/dataset/train.jsonl, test.jsonl
caffeinate -i python scripts/train_lora.py   # ~10 min of compute on an M2 → models/navi-lora
caffeinate -i python scripts/evaluate.py     # reports/EVAL.md
```

```bash
pytest && ruff check . && ruff format --check .
```
