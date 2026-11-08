# MedVia Navi — loyiha hisoboti (R1–R5)

_2026-11-07. Repo: `~/Desktop/medvia-navi`. Hali commit qilinmagan._

## 1. Bu nima va nima uchun

**MedVia Navi** — MedVia uchun alohida Python xizmati. U **RAG** (Retrieval-Augmented
Generation, ya'ni "qidirib topilgan matn asosida javob yaratish") texnikasi bilan ishlaydi.
Chet ellik bemor (RU/UZ/KK/EN) muolaja, narx, bron qilish yoki tashrif kuni haqida savol beradi.
Navi uchta qoidaga amal qiladi:

1. **Faqat MedVia kontentidan javob beradi.** Klinika, muolaja, FAQ, maslahat, e'lon va
   qo'llanmalardan foydalanadi. Internetdan yoki model "xotirasi"dan javob bermaydi.
2. **Manba ko'rsatadi:** `[1]` yoki `[2]`. Server raqamni tekshiradi: manbalar ichida yo'q
   raqam `invalid_citations` ro'yxatiga tushadi.
3. **Bilmasa, "bilmayman" deydi**, savol qaysi tilda bo'lsa, o'sha tilda.

Bu tibbiy qurilma emas, tashxis qo'ymaydi. Kontent faqat MedVia'ning sintetik demo
ma'lumotlari. Bemorning savoli Qdrant'ga yozilmaydi.

Loyiha `Jaydari_RAG` ni ko'rib chiqib, uning kamchiliklarini tuzatgan holda qurildi (3-bo'lim).

## 2. Arxitektura

```
MedVia GraphQL ──export_medvia.py──▶ data/corpus ──POST /v1/documents──▶ chunker ─▶ e5 ─▶ Qdrant
                                                                                        │
savol ──POST /v1/query──▶ e5 ("query:") ─▶ Qdrant top 4 ─▶ ball ≥ 0.80? ──yo'q──▶ rad etish (LLMsiz)
                                                                  │ha
                                                  chat template + manbalar ─▶ Qwen + LoRA
                                                                  │
                                            javob + [n] manbalarga qarshi tekshirilgan
```

| Qism | Tanlov | Nima uchun |
| --- | --- | --- |
| API | FastAPI (app factory) | Testlar o'z ilovasini `:memory:` Qdrant bilan quradi |
| Vektor baza | Qdrant | Server, diskda (`path:`) yoki xotirada: bitta kod, uch rejim |
| Embedding (matnni vektorga aylantirish) | `intfloat/multilingual-e5-small`, 384 o'lcham | RU/UZ savollar; inglizcha MiniLM ularni topa olmaydi |
| LLM (Large Language Model, katta til modeli) | `Qwen/Qwen2.5-0.5B-Instruct` | 8 GB Apple M2'da o'qitiladi va ishlaydi |
| Fine-tune | LoRA (Low-Rank Adaptation), PEFT + TRL | 1.75% parametr o'qitiladi, adapter 35 MB |

MedVia bilan chegara: Navi MedVia bazasiga **ulanmaydi**, faqat ochiq GraphQL API'ni o'qiydi.
Bir xizmat boshqasining jadvallarini o'qisa, sxemasi o'zgarganda ikkalasi ham buziladi.

## 3. Jaydari'dan nimani tuzatdik

| Jaydari | Navi | Nima uchun |
| --- | --- | --- |
| Har 1,200 belgida kesish | Markdown sarlavhalari bo'yicha kesish, katta bo'lsa 900/120 | Narx o'z muolajasidan ajralib qolmaydi |
| Bo'lak sarlavhasiz | Har bo'lakda sarlavha yo'li ("Seoul Glow › Opening hours") | Savol so'zlari ko'pincha sarlavhada turadi |
| `/ingest-path` server papkasini o'qiydi | Hujjat body'da, qat'iy `doc_id` | Path traversal (fayl yo'lini aylanib o'tish hujumi) xavfi yo'q |
| Qayta yuklashda eski bo'laklar qoladi | Almashtirish = o'chirish + yozish, `uuid5` id | Kichraygan hujjat eski bo'laklarni qoldirmaydi |
| "Bilmayman" kalit so'zlar ro'yxati bilan | O'xshashlik chegarasi + LoRA'ning rad etishi | Savoldagi so'z emas, indeksdagi kontent qaror qiladi |
| `Question: … Answer:` xom satr | `apply_chat_template` | Model o'qitilgan formatda gapiradi |
| LoRA faqat `q_proj`, `v_proj` | `all-linear` | Format/xulq MLP (Multi-Layer Perceptron) qatlamlarida ham joylashgan |
| O'lchov yo'q | R5: alohida test to'plami, base va LoRA bir xil retrieval bilan | "Yaxshiroq" degan da'vo raqam bilan isbotlanadi |

## 4. Bosqichlar

### R1 — Skelet
- **`Settings`** (pydantic-settings, `NAVI_` prefiksi). 32 belgidan qisqa kalit bo'lsa, ilova
  ishga tushmaydi. `SecretStr` kalitni log'larga chiqarmaydi.
- **`X-API-Key`** `hmac.compare_digest` bilan solishtiriladi. Oddiy `==` birinchi farq qilgan
  baytda to'xtaydi. Javob vaqti kalitning qanchasi to'g'ri ekanini oshkor qiladi (timing
  attack, vaqt hujumi).
- **`/health/live`** faqat jarayon tirikligini bildiradi. **`/health/ready`** Qdrant javob
  berishini tekshiradi; javob bo'lmasa 503 qaytaradi. Liveness Qdrant'ni tekshirsa, orkestrator
  Qdrant bir lahza uzilganda sog'lom API'ni qayta ishga tushirib yuboradi.
- Dockerfile non-root (root bo'lmagan) foydalanuvchi bilan ishlaydi. Compose Qdrant'ni faqat
  127.0.0.1 da ochadi. CI (Continuous Integration) torch'siz ishlaydi.

### R2 — Ingest (yuklash)
- `scripts/export_medvia.py` MedVia GraphQL'dan 55 ta hujjat oladi: 8 klinika, 32 muolaja,
  5 maslahat, 6 FAQ, 4 e'lon. Bularga 7 ta qo'lda yozilgan qo'llanma qo'shiladi. Jami
  62 hujjat, 100 bo'lak (chunk).
- `document_hash` = SHA-256 (CHUNKER_VERSION, sarlavha, tur, til, matn). O'zgarmagan hujjat
  qayta embedding qilinmaydi. Chunker o'zgarsa, hamma hash o'zgaradi va hamma narsa qayta
  bo'linadi. Bu xato qurish jarayonida topildi.
- E5 modeli `query:` / `passage:` prefikslarini talab qiladi. E5 bilan ishlashdagi eng ko'p
  uchraydigan xato shu prefikslarni unutish.

### R3 — Query (so'rov)
- Hech bir manba 0.80 dan oshmasa, rad etish **LLM chaqirilmasdan** qaytadi. Bu arzonroq, va
  so'ralmagan model javob to'qiy olmaydi.
- Rad etish matni savol tilida bo'ladi. Til yozuvga qarab aniqlanadi: kirill, qozoqcha harflar,
  o'zbekcha so'zlar.
- `asyncio.Semaphore(1)` va `run_in_threadpool` ishlatiladi: bir vaqtda bitta generatsiya
  ketadi, event loop esa `/health` uchun bo'sh qoladi.

### R4 — Dataset va LoRA
- **Dataset:** train 348 ta, test 87 ta.
  - Shablon savollar: narx, davomiylik, manzil, telefon, yakshanba, tarjimon.
  - Qo'lda yozilgan savollar: javobli 51 ta, javobsiz 40 ta.
  - "Bu klinika bu muolajani qilmaydi" savollari.
  - **Hard negative** (qiyin salbiy misol): javobli savol, lekin javob turgan bo'laklar
    olib tashlangan; to'g'ri javob rad etish.
- **Train like you serve** (xizmatdagidek o'qit): har bir prompt haqiqiy retriever va
  `build_messages` orqali quriladi.
- **Hujjat bo'yicha ajratish:** test savollaridagi klinika va muolajalarni model o'qitishda
  ko'rmagan.
- **Ikki kalit:** `key` javobda bo'lishi shart, `source_key` manbada bo'lishi shart. Masalan,
  ruscha javobda "не работает", manbada esa "closed" turadi.
- **LoRA sozlamalari:** r=16, alpha 32, `all-linear`, 8.8M parametr (1.75%), float32,
  1 epoch, batch 1 × gradient accumulation 8. Natija: train loss 0.41, eval loss 0.30.
- **8 GB'da o'qitish (muhim tajriba):**
  - Birinchi urinish batch 2 va eval batch 8 bilan 8 soat yurdi, keyin MPS OOM (Out Of
    Memory, xotira tugashi) bilan qulab tushdi. Logit tensori bitta misol uchun
    650 × 151,936 × 4 bayt ≈ 0.4 GB, backward'da ikki baravar.
  - Tuzatish: batch 1, eval batch 1. Adapter `evaluate()` dan **oldin** saqlanadi.
  - Mac uyquga ketganda GPU ham to'xtadi. `caffeinate -i` (Mac'ni uyg'oq ushlab turuvchi
    buyruq) yechim bo'ldi.

### R5 — Baholash
`scripts/evaluate.py` uchta narsani o'lchaydi: retrieval, chegara jadvali (threshold grid) va
base va LoRA generatsiyasi. Ikkala model bir xil manbalarni oladi.

**v1 → v2: o'lchab topilgan va datada tuzatilgan xato.**
- v1 LoRA javobli savollarning 10.4% ini rad etdi, deyarli hammasi narx savoli edi.
- Sabab: "taklif qilinmaydi" turidagi salbiy misollarning hammasi narx savoli bo'lgan.
  Train'dagi 110 ta narx savolidan 38 tasi rad etish bilan tugagan. Model manbani emas,
  savol shaklini o'rganib olgan.
- v2'da bu savollarning yarmi davomiylik haqida bo'ldi. Javob aniqligi 73.9 dan 76.8 ga
  qaytdi, ortiqcha rad etish 9.0% ga tushdi.

Baholash skriptining o'zida ham xato topildi: retrieval hisoboti 0.80 dan past ballarni
filtrlab tashlagan edi. Endi filtrsiz hisoblanadi.

## 5. Natijalar (v2, 87 test savol: EN 37, RU 30, UZ 20)

| Metrika | Qwen2.5-0.5B | + Navi LoRA |
| --- | ---: | ---: |
| Javob aniqligi (asosiy fakt) | 76.8 | 76.8 |
| To'g'ri manbaga havola (grounded citation) | 8.7 | **84.1** |
| Savol tilida javob | 59.4 | **98.6** |
| Javobsiz savolda rad etish | 22.2 | **83.3** |
| Ortiqcha rad etish (manba bor edi) | 1.5 | 9.0 |
| O'rtacha kechikish, ms (M2) | 597 | 532 |

| Til | Aniqlik base → LoRA | Manbaga havola | Rad etish |
| --- | --- | --- | --- |
| EN | 92.9 → 85.7 | 17.9 → 89.3 | 22.2 → 100 |
| RU | 75.0 → 70.8 | 4.2 → 70.8 | 33.3 → 83.3 |
| UZ | 52.9 → 70.6 | 0.0 → 94.1 | 0.0 → 33.3 |

**Retrieval** (topish): Hit@1 95.7%, Hit@4 100%, MRR@10 (Mean Reciprocal Rank, birinchi
to'g'ri natija o'rnining teskari qiymati, o'rtacha) 0.978.

**Chegara jadvali** "faqat ball bilan rad etib bo'lmaydi" degan xulosani ko'rsatadi:
- javobli savollarning top-1 bali 0.77 gacha tushadi;
- javobsizlarniki 0.92 gacha chiqadi.

0.80 chegarasi javobli savollarning 97% ini o'tkazadi, javobsizlarning esa faqat 11% ini
to'xtatadi. Qolganini LLM rad etishi kerak, LoRA aynan shuning uchun kerak.

**Xulosa:** LoRA **xulqni** o'rgatdi: manba ko'rsatish, savol tilida javob berish va rad etish.
Faktlar retrieval'dan keladi. Aniqlik o'zgarmadi, chunki bilim o'qitilmagan, faqat format
o'rgatilgan.

## 6. Halol cheklovlar

1. **Test to'plami kichik.** 18 ta javobsiz savol bor, ya'ni bitta savol 5.6 punkt.
   v1 va v2 o'rtasidagi bir necha punktlik farq shovqin bo'lishi mumkin.
2. **Test shablonga o'xshaydi.** API orqali berilgan erkin savollar zaifroq chiqdi:
   - "slot qancha vaqt ushlanadi" savoliga to'g'ri hujjatdan noto'g'ri fakt qaytdi
     ("3 soat", aslida 10 daqiqa);
   - o'zbekcha erkin savolga shablon iborasi qaytdi ("narxi boshlanadi");
   - "Pico Laser"da to'qima javob bo'ldi.
3. **O'zbekcha rad etish zaif** (3 tadan 1 tasi): train'da o'zbekcha salbiy misollar kam.
4. **Seed ziddiyati:** Mintleaf tavsifida 3 ta tarjimon tili yozilgan, maydonda esa faqat UZ.
   Bu savol datasetdan chiqarib tashlangan; tuzatish MedVia seed'iga tegishli.
5. **Docker Compose va GitHub CI** sinab ko'rilmagan: Docker Desktop o'chiq, remote yo'q.

## 7. Keyingi qadamlar (rejada yo'q, g'oya)

- Chegaradan oldin reranker (cross-encoder, savol va bo'lakni birga baholovchi model) qo'yish.
- Ko'proq o'zbekcha va erkin shakldagi train savollari, ayniqsa salbiy misollar.
- Colab T4'da 1.5B model.
- MedVia batch'i har kecha `POST /v1/documents` ni chaqirishi; hash o'zgarmaganlarni o'tkazib
  yuboradi.

## 8. Code review: asosiy fayllar

| Fayl | Nima qiladi | Review'da nimaga e'tibor berish kerak |
| --- | --- | --- |
| `src/navi/core/config.py` | `Settings` | Validator ishga tushishda to'xtatadi; `lru_cache` testlarda `_env_file=None` bilan chetlab o'tiladi |
| `src/navi/core/security.py` | API kalit | `auto_error=False`: 401 bilan birga `WWW-Authenticate` ham qaytaradi |
| `src/navi/services/chunker.py` | Bo'lish | Sarlavha yo'li; H1 sarlavha takrorlanmasligi uchun `startswith` |
| `src/navi/services/ingest.py` | Yuklash | Hash'ga `CHUNKER_VERSION` kiradi; embedding'ga sarlavha ham qo'shiladi |
| `src/navi/services/vector_store.py` | Qdrant | `uuid5` id, almashtirish = o'chirish + yozish, o'lcham tekshiruvi |
| `src/navi/services/rag.py` | Javob | Avval chegara, keyin LLM; havola tekshiruvi |
| `src/navi/services/generator.py` | Qwen | GPU'da fp16, CPU'da fp32; adapter merge; greedy decoding (har qadamda eng ehtimolli token) |
| `scripts/build_dataset.py` | Dataset | Hujjat bo'yicha ajratish, hard negative, `key` va `source_key` |
| `scripts/train_lora.py` | O'qitish | Batch 1 × 8, eval batch 1, avval saqlash |
| `scripts/evaluate.py` | O'lchov | Filtrsiz retrieval, grounded citation, tillar kesimida |

Testlar: pytest 36/36, ruff toza. Testlar `HashEmbedder` va `FakeGenerator` bilan ishlaydi,
torch kerak emas.

## 9. Intervyu savollari

1. Nima uchun retrieval va generatsiya alohida o'lchanadi?
2. Nima uchun faqat o'xshashlik chegarasi "bilmayman" deyishga yetmaydi? (Javob: ballar
   ustma-ust tushadi, 0.77–0.92.)
3. LoRA narx savollarini ko'proq rad etdi. Sababini qanday topdingiz va qanday tuzatdingiz?
4. "Har qanday havola" va "to'g'ri manbaga havola" metrikalarining farqi nimada?
5. LoRA nima uchun aniqlikni oshirmadi, lekin formatni o'zgartirdi?
6. 8 GB xotirada o'qitishda eng katta tensor qaysi va uni qanday kamaytirdingiz?
7. Nima uchun `/health/live` Qdrant'ni tekshirmaydi?
8. Content hash'ga nima uchun chunker versiyasi kiradi?
9. Bir hafta va GPU bo'lsa, birinchi nima qilardingiz: kattaroq model, ko'proq data yoki reranker?
