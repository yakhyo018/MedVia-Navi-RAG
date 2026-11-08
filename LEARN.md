# LEARN — MedVia Navi

## R1 — Skeleton (2026-11-06)

**Problem.** A service other services call, which must refuse to start when misconfigured and
tell an orchestrator honestly whether it can take traffic.

- **App factory** (`create_app(settings, store)`) instead of a module-level `app`: tests build
  isolated apps with an in-memory Qdrant; importing the module connects to nothing.
- **Settings at boot** (pydantic-settings): a short `NAVI_API_KEY` or an unknown environment
  stops the process before the first request. `SecretStr` keeps the key out of logs.
- **API key, constant time**: `hmac.compare_digest`, because `==` returns at the first different
  byte and the response time leaks how much of the key was right (timing attack).
- **Liveness vs readiness**: live = the process runs (no dependency checks, or the orchestrator
  restarts a healthy API whenever Qdrant blinks); ready = Qdrant answers (503 → the load
  balancer sends no traffic).
- **Collection dimension check**: a 768-dim collection with a 384-dim model would fail every
  search at runtime; it fails at boot instead.
- **`[ml]` extra**: torch is ~2 GB; CI and R1 install without it.

**Interview questions.**
1. Why does `/health/live` not check Qdrant?
2. Why `hmac.compare_digest` and not `==`?
3. What does the app factory buy in tests?
4. Why bind Qdrant to 127.0.0.1 in Compose?
5. Why are `/docs` and `/openapi.json` off in production?

## R2 — Ingest (2026-11-06)

**Problem.** Turn MedVia's content into searchable pieces, keep the index equal to the source
when content changes, and never read anything MedVia did not send.

- **Documents in the body, by id** (`POST /v1/documents`): Jaydari's `/ingest-path` read any
  server folder. `doc_id` has a strict pattern (`^[a-z0-9][a-z0-9_-]*$`).
- **Replace = delete + write**: a document that shrank from 5 chunks to 3 would otherwise keep
  2 stale chunks answering forever. Point ids are `uuid5(doc_id:chunk_index)` — stable.
- **Content hash with the chunker version**: unchanged documents are skipped (MedVia may resend
  the catalogue nightly); a new chunking rule changes every hash, so everything is re-chunked.
  Found while building: without the version, the heading fix left all 62 docs "unchanged".
- **Chunking by Markdown headings**, size only as a fallback (900 chars, 120 overlap), and every
  chunk carries its heading path ("Seoul Glow Dermatology › Opening hours").
- **multilingual-e5-small** (384 dim, MIT): EN-only MiniLM would miss RU / UZ questions. E5 needs
  the `query:` / `passage:` prefixes — the most common E5 mistake is leaving them out.
- **Export from MedVia's public GraphQL** (`scripts/export_medvia.py`), not its database: a
  service never reads another service's tables. Junk test posts (`dfdsfs`) are filtered.

## R3 — Query (2026-11-06)

- **Chat template, not a raw string**: `tokenizer.apply_chat_template` renders the turns the way
  the model was trained. Jaydari's `Question: … Answer:` made the model continue the pattern.
- **Retrieval decides first**: nothing above `min_score` → the refusal without calling the
  model. Jaydari decided by keywords in the question.
- **Refusal in the question's language**, picked by script (Cyrillic, Kazakh letters, Uzbek words).
- **Citations are checked**: numbers outside the sources are reported as `invalid_citations`.
- **One generation at a time** (`asyncio.Semaphore(1)` + `run_in_threadpool`): the event loop
  stays free for `/health`, the GPU is not shared by two generations.
- **Base model, before any training**: right facts, but no `[n]` in half the answers, Russian
  questions answered in English, the question echoed back. That is what R4 trains away.

## R4 — Dataset + LoRA (2026-11-06)

- **LoRA teaches behaviour, retrieval brings facts.** The target: cite the supporting source,
  answer in the question's language, refuse when the sources do not hold the answer.
- **Train like you serve**: every prompt is built by the real retriever + `build_messages`.
- **Three kinds of negatives**: hand-written off-topic questions, "a treatment at a clinic that
  does not offer it" (retrieval brings the same treatment elsewhere, with a tempting price),
  and hard negatives — an answerable question with every supporting chunk removed.
- **Answer key vs source key**: "closed" is in the source, "не работает" in the Russian answer.
  One key for both broke the builder; two keys made it honest.
- **Split by document** for template questions: the test asks about clinics and treatments the
  model never saw in training.
- **Prompt–completion format** (TRL): the loss is on the answer only. `all-linear` LoRA, r=16,
  ~8.8M trainable parameters (1.75 %), float32 on MPS (fp16 has no loss scaling there).
- **Version drift**: transformers 5 removed `warmup_ratio` (a float `warmup_steps` is a ratio now)
  and `apply_chat_template(tokenize=True)` returns a BatchEncoding, not a list.
- **8 GB is a budget, not a detail.** The first run (batch 2, eval batch 8) died after 8 hours
  with an MPS out-of-memory. The logits are the big tensor: 650 tokens × 151,936 vocabulary ×
  4 bytes ≈ 0.4 GB per example, doubled in backward. Batch 1 × accumulation 8 keeps the same
  effective batch; `per_device_eval_batch_size=1`; the adapter is saved *before* `evaluate()`.
- **A sleeping Mac pauses the GPU.** Steps jumped from 20 s to 15 min while `perf_counter`
  (which does not count sleep) reported 10 minutes of compute. `caffeinate -i` fixed it.
- **One epoch was enough**: the train loss was flat (~0.3) from half an epoch on.

## R5 — Evaluation (2026-11-07)

**Problem.** "The fine-tuned model is better" means nothing without numbers, a held-out set,
and the same retrieval for both models.

- **Separate retrieval from generation.** Hit@1 / Hit@4 / MRR@10 (Mean Reciprocal Rank) say
  whether the right chunk reached the model; answer accuracy *when retrieval found the source*
  says whether the model used it. A wrong answer is then either a search bug or a model bug.
- **Grounded citation, not "any citation"**: `[1]` counts only if source 1 holds the fact.
- **The threshold grid** shows that a score alone cannot refuse: answerable top-1 scores go
  down to 0.77, unanswerable ones up to 0.92. The LLM has to refuse; the threshold only removes
  the obvious off-topic questions for free.
- **Found by measuring, fixed in the data (v1 → v2).** v1 LoRA refused 10.4 % of answerable
  questions, almost all of them price questions. Cause: every "not offered" negative was a
  price question, so 38 of 110 price questions in train ended in a refusal — the model learnt
  the question's form, not the sources. v2 asks half of them about duration: accuracy back to
  the base model's 76.8 %, false refusals 9.0 %.
- **A bug in the eval itself**: the retrieval report used `rag.retrieve()`, which drops scores
  below `min_score` — the grid below 0.80 was empty. The report now uses an unfiltered service.
- **Honest limits**: 18 unanswerable test questions (one question = 5.6 points); template-shaped
  test questions overstate free-form quality, which the end-to-end API run showed.

**Interview questions.**
1. Why measure retrieval and generation separately?
2. Why can't a similarity threshold alone decide "I don't know"?
3. The LoRA refused more price questions than the base model. How did you find the cause?
4. Why is "any citation" a weaker metric than "grounded citation"?
5. What would you change first with a week and a GPU — model size, data, or a reranker?
