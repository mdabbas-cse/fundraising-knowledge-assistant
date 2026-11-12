# Learning Guide: Building a RAG Baseline

This guide explains every concept and module in the project, step by step, in simple
English. Difficult AI ideas also have a **Bangla explanation (বাংলা ব্যাখ্যা)**. Each
module section ends with interview questions and short model answers.

If you know Laravel, these comparisons help:

| This project | Laravel equivalent |
|--------------|--------------------|
| `app/config.py` + `.env` | `config/*.php` + `.env` |
| `@dataclass(frozen=True) Document` | A DTO / value object |
| `knowledge_base.load_documents()` | A repository method |
| `AnswerService` | A service class |
| `typing.Protocol` (`LLMProvider`) | A PHP `interface` |
| Passing `llm_provider` into the constructor | Service container dependency injection |
| pytest fixtures | PHPUnit `setUp()` and model factories |

---

## Part 1: Core concepts

### 1.1 The complete RAG workflow

```text
INDEXING (once)                          QUERY (every question)
---------------                          ----------------------
documents (JSON)                         question
   |  load + validate                       |  validate + clean
   v                                        v
clean text                               question vector (same TF-IDF space)
   |  split into chunks                     |
   v                                        |  cosine similarity with every chunk
chunks (+ id, title)                        v
   |  TF-IDF fit                         scores -> best chunk per document
   v                                        |  threshold, top_k
chunk vectors  --------------------------->  v
                                         retrieved documents
                                            |  build grounded prompt
                                            v
                                         prompt + sources --(Phase 2)--> LLM --> answer
```

> **বাংলা ব্যাখ্যা — RAG কী?**
> RAG (Retrieval-Augmented Generation) মানে হলো: প্রশ্নের উত্তর দেওয়ার আগে প্রথমে
> নিজেদের ডকুমেন্ট থেকে প্রাসঙ্গিক অংশ খুঁজে বের করা (retrieval), তারপর সেই অংশগুলো
> LLM-কে দিয়ে বলা — "শুধু এই তথ্য ব্যবহার করে উত্তর দাও" (generation)। এটা অনেকটা
> ওপেন-বুক পরীক্ষার মতো: ছাত্র মুখস্থ থেকে না লিখে বই খুলে সঠিক পৃষ্ঠা দেখে উত্তর লেখে।

### 1.2 Retrieval vs generation

| | Retrieval | Generation |
|-|-----------|------------|
| Question it answers | *Which passages are relevant?* | *What is the answer, in words?* |
| Output | Ranked passages + scores | Natural-language text |
| Tool here | TF-IDF + cosine similarity | LLM (Phase 2) |
| Deterministic? | Yes, same input gives same output | Usually not |
| Can it invent facts? | No, it only returns existing text | Yes, this is hallucination |

> **বাংলা ব্যাখ্যা — Retrieval বনাম Generation**
> Retrieval হলো খোঁজা — লাইব্রেরিয়ান যেমন সঠিক বইটা খুঁজে দেন। Generation হলো লেখা —
> সেই বই পড়ে কেউ আপনার প্রশ্নের উত্তর নিজের ভাষায় লিখে দেয়। Phase 1-এ আমরা শুধু
> খুঁজছি এবং প্রম্পট তৈরি করছি; লেখার কাজ (LLM) Phase 2-এ হবে।

### 1.3 Why RAG reduces hallucination but cannot guarantee correctness

RAG helps because the model gets the right facts in the prompt, is told to use only
those facts, and must cite sources a human can check.

It cannot guarantee correctness because:

1. **Retrieval can fail.** The right chunk may not be found (synonyms), or a wrong one
   may be found (shared words). Example in this project: "football team performance"
   retrieves Campaign Performance.
2. **The source can be wrong or outdated.** RAG faithfully repeats a bad document.
3. **The model can still misread or ignore the context**, or mix in its own memory.
4. **Missing context.** If only part of the answer is retrieved, the model may "fill
   the gap".

Mitigations used here: a similarity threshold, no LLM call without context, an exact
refusal sentence, required citations, and a check for citations outside the context.

> **বাংলা ব্যাখ্যা — Hallucination**
> Hallucination মানে LLM আত্মবিশ্বাসের সাথে এমন কিছু বলে যা সত্য নয় বা কোনো উৎসে নেই।
> RAG এটা কমায়, কারণ মডেলকে নির্দিষ্ট তথ্য দেওয়া হয় এবং উৎস (document ID) উল্লেখ করতে
> বলা হয়। কিন্তু পুরোপুরি বন্ধ করতে পারে না: ভুল ডকুমেন্ট খুঁজে আসতে পারে, ডকুমেন্ট
> নিজেই পুরনো বা ভুল হতে পারে, আর মডেল context ভুল বুঝতে পারে। তাই উৎস দেখানো এবং
> মানুষের যাচাই (human review) জরুরি।

### 1.4 How a document becomes a vector

Take this chunk: *"Donor Retention. To improve donor retention, thank every donor..."*

1. **Clean**: normalize unicode, collapse whitespace (`clean_text`).
2. **Add title**: `"Donor Retention. " + chunk text` (contextual header).
3. **Lowercase**: `"donor retention. to improve donor retention, thank every donor..."`
4. **Tokenize**: split into words of 2+ letters: `donor, retention, to, improve, ...`
5. **Remove stop words**: drop `to`, `every`, `how`, `can`, `we`, `the`...
6. **Make n-grams**: single words and pairs: `donor`, `retention`, `donor retention`,
   `improve`, `improve donor`, ...
7. **Count (TF)** and apply `1 + log(count)`.
8. **Multiply by IDF** (rare terms count more).
9. **L2-normalize** so the vector has length 1.

The result is one row of a matrix with one column per vocabulary term (1,227 terms in
this project). Almost all values are 0, so it is stored as a **sparse** matrix.

> **বাংলা ব্যাখ্যা — Vector কী?**
> কম্পিউটার শব্দ বোঝে না, সংখ্যা বোঝে। তাই প্রতিটি chunk-কে একটি লম্বা সংখ্যার তালিকা
> (vector) বানানো হয় — vocabulary-র প্রতিটি শব্দের জন্য একটি ঘর। যে শব্দ chunk-এ নেই
> তার ঘরে 0। আমাদের vocabulary-তে ১,২২৭টি term, তাই প্রতিটি chunk ১,২২৭ সংখ্যার vector,
> যার বেশিরভাগ ঘরই 0 — একে বলে sparse vector।

### 1.5 TF-IDF in one minute

- **TF** = how often a term appears in this chunk.
- **IDF** = how rare the term is across all chunks: `ln((1 + n) / (1 + df)) + 1`.
- A term that is frequent here **and** rare elsewhere gets the highest weight.

Worked example (real numbers, 3 chunks): `donors` appears in 2 chunks, so
`idf = ln(4/3) + 1 = 1.288`; `retention` appears in 1, so `idf = ln(4/2) + 1 = 1.693`.
After normalization, in chunk "Thank donors to improve retention", `donors = 0.402`
while `improve`, `retention` and `thank` each get `0.529`.

> **বাংলা ব্যাখ্যা — TF-IDF**
> TF-IDF প্রতিটি শব্দকে একটি ওজন (weight) দেয়। TF: শব্দটি এই chunk-এ কতবার এসেছে।
> IDF: শব্দটি সব chunk-এর মধ্যে কত বিরল। যে শব্দ সব জায়গায় আছে (যেমন fundraising
> ডকুমেন্টে "donor"), তার ওজন কম; যে শব্দ অল্প জায়গায় আছে (যেমন "segmentation"), তার
> ওজন বেশি — কারণ সেই শব্দটাই বিষয়টাকে আলাদা করে চিনিয়ে দেয়।

### 1.6 Cosine similarity in one minute

Cosine similarity compares the **direction** of two vectors:
`cos = (a . b) / (|a| |b|)`. TF-IDF vectors already have length 1, so it is simply
"multiply matching terms, then add".

Example: question vector `improve = 0.707, retention = 0.707`; chunk A has
`improve = 0.529, retention = 0.529`. Score = `0.707 x 0.529 + 0.707 x 0.529 = 0.748`.
A chunk with no shared terms scores `0`.

> **বাংলা ব্যাখ্যা — Cosine similarity**
> Cosine similarity দুটি vector-এর মধ্যে কোণ (angle) মাপে। দুটো তীর যদি একই দিকে
> নির্দেশ করে, মান 1; একদম আলাদা দিকে (কোনো শব্দ মিল নেই) হলে 0। তীরের দৈর্ঘ্য
> গুরুত্বপূর্ণ নয়, দিকটাই আসল — তাই লম্বা আর ছোট লেখাও তুলনা করা যায়।

### 1.7 Lexical vs semantic retrieval

| | Lexical (TF-IDF, BM25) | Semantic (embeddings) |
|-|------------------------|-----------------------|
| Matches | Exact words | Meaning |
| "supporter details" vs "personal information" | No match | Match |
| Product codes, IDs, rare names | Very good | Sometimes weak |
| Explainable? | Yes, shows matched terms | Hard to explain |
| Cost | Tiny, CPU only | Model inference, vector storage |

Production systems often use **hybrid search**: both, combined.

> **বাংলা ব্যাখ্যা — Lexical বনাম Semantic**
> Lexical retrieval শব্দ মেলায়, অর্থ বোঝে না। "supporter details" আর "personal
> information" একই অর্থ হলেও শব্দ আলাদা বলে TF-IDF মিল খুঁজে পায় না। Semantic
> embedding একটি মডেল দিয়ে লেখার অর্থকে vector-এ রূপান্তর করে, তাই একই অর্থের ভিন্ন
> শব্দও কাছাকাছি আসে। এটা Phase 2-তে যোগ করা হবে।

### 1.8 This baseline vs production RAG

| Area | Baseline (Phase 1) | Production |
|------|-------------------|------------|
| Retrieval | TF-IDF | Embeddings + BM25 hybrid, reranker |
| Storage | In memory | Vector DB with metadata filters and access control |
| Documents | 6 JSON files | Thousands of PDFs/DOCX, incremental ingestion |
| Generation | None (prompt only) | LLM with streaming, retries, cost limits |
| Quality checks | Unit tests | Evaluation datasets, monitoring, user feedback |
| Security | Local only | Auth, roles, audit logs, PII redaction, data agreements |

---

## Part 2: Module walkthrough

### 2.1 Knowledge base (`app/knowledge_base.py`, `data/documents/`)

**Problem it solves.** The system needs a reliable, validated source of documents.
Bad input (missing title, duplicate ID) must fail early with a clear message instead
of producing confusing search results.

**1. What was implemented?**
Six synthetic JSON documents; a `Document` dataclass; `load_documents(dir)` with
validation of required fields, empty values, invalid JSON and duplicate IDs; a
`LOADERS` registry mapping file extensions to loader functions.

**2. How does it work?**
`load_documents` loops over the folder, picks a loader by extension (`.json` now),
skips other files such as `README.md`, validates each record in
`document_from_dict`, checks IDs are unique, and returns documents sorted by ID.

**3. Why is it needed?**
Retrieval quality starts with data quality. Stable IDs are what the LLM will cite, so
they must be unique and preserved. The loader registry means PDF/DOCX support is one
new function plus `LOADERS[".pdf"] = load_pdf_file`, with no other changes.

**4. How can I test it?**
```bash
python -m pytest tests/test_knowledge_base.py -v
python -c "from app.config import load_settings; from app.knowledge_base import load_documents; print([d.id for d in load_documents(load_settings().documents_dir)])"
```

**5. Interview questions**
- *Why validate documents at load time?* Fail fast. A broken document should stop
  startup with a clear error, not silently degrade answers.
- *Why are document IDs important in RAG?* They are the citation keys. Users and
  automated checks use them to verify answers.
- *How would you add PDF support?* Write `load_pdf_file(path) -> list[Document]`
  (for example with `pypdf`), register it in `LOADERS`. Watch out for scanned PDFs
  (need OCR), headers/footers, and tables.
- *Why keep the disclaimer out of `content`?* Everything in `content` is indexed.
  The same sentence in every document would add noise to every search.

### 2.2 Document processing (`app/document_processor.py`)

**Problem it solves.** Raw documents are too long and messy for precise retrieval.

**1. What was implemented?**
`clean_text` (unicode normalization, typographic quotes/dashes, whitespace);
`split_into_sentences`; `chunk_text` (sentence-aware, ~80 words, 1 sentence overlap,
tiny final chunk merged); `process_documents` producing `Chunk` objects with
`chunk_id`, `document_id`, `title`, `category`, `source_type`, `chunk_index`.

**2. How does it work?**
Sentences are packed into a chunk until adding the next one would pass `max_words`.
The next chunk starts with the last sentence of the previous one (overlap). If the
final chunk has fewer than half of `max_words` of new text, it is merged into the
previous chunk.

**3. Why is it needed?**
- One vector for a whole document mixes topics, so specific questions match weakly.
- LLM prompts have limited space; we send focused passages, not whole documents.
- Overlap protects ideas that cross a chunk boundary.
- Merging tiny chunks fixed a real bug: a 28-word chunk outranked the definition
  chunk for "What is donor segmentation?" (0.2693 vs 0.2645), because short chunks
  get inflated cosine scores.

> **বাংলা ব্যাখ্যা — Chunking ও Overlap**
> Chunking মানে বড় ডকুমেন্টকে ছোট ছোট টুকরো করা। কারণ একটা বড় ডকুমেন্টে অনেক বিষয়
> মিশে থাকে, ফলে নির্দিষ্ট প্রশ্নের সাথে মিল দুর্বল হয়; আর LLM-এর প্রম্পটে জায়গাও
> সীমিত। Overlap মানে আগের টুকরোর শেষ বাক্যটা পরের টুকরোর শুরুতে আবার রাখা, যাতে
> দুই টুকরোর সীমানায় কোনো ধারণা হারিয়ে না যায়।

**4. How can I test it?**
```bash
python -m pytest tests/test_document_processor.py -v
python -c "
from app.config import load_settings
from app.knowledge_base import load_documents
from app.document_processor import process_documents
for c in process_documents(load_documents(load_settings().documents_dir), 80, 1):
    print(c.chunk_id, len(c.text.split()), c.text[:60])"
```
Try `FKA_CHUNK_MAX_WORDS=40 python main.py` and compare scores.

**5. Interview questions**
- *How do you choose chunk size?* Trade-off: small chunks are precise but lose
  context; large chunks keep context but dilute the match and use prompt space.
  Measure retrieval quality on an evaluation set for several sizes.
- *Why overlap?* So a fact split across two chunks is still fully present in one.
- *Why not lowercase in `clean_text`?* The text is shown to users and the LLM;
  lowercasing for matching is done inside the vectorizer.
- *Other chunking strategies?* Fixed-size tokens, recursive splitting by headings and
  paragraphs, semantic chunking (split where meaning changes), parent-child chunks
  (search small chunks, send the larger parent to the LLM).

### 2.3 Retriever (`app/retriever.py`)

**Problem it solves.** Find the chunks that best match a question, fast and
explainably, without any external service.

**1. What was implemented?**
`TfidfRetriever` (fit on chunks, `vectorize_question`, `score_chunks`, `retrieve`,
`matched_terms`), `normalize_question` (empty / too long / wrong type),
`build_retriever(settings)`, and the convenience function
`retrieve_documents(question, top_k=3)`.

**2. How does it work?**
At startup: `TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
sublinear_tf=True).fit_transform(chunks)`. Per question: `transform([question])`,
`cosine_similarity(question_vector, chunk_matrix)`, keep the best chunk per document,
drop scores `<= 0` or below the threshold, sort by score (ties by ID), take `top_k`.
`matched_terms` multiplies the two vectors element-wise; the biggest products are the
terms that produced the score.

**3. Why is it needed?**
Retrieval decides what the LLM sees. If retrieval is wrong, the answer will be wrong
no matter how good the LLM is ("garbage in, garbage out").

> **বাংলা ব্যাখ্যা — Threshold ও top_k**
> top_k হলো সর্বোচ্চ কয়টি ডকুমেন্ট ফেরত দেবে। Threshold হলো ন্যূনতম স্কোর — এর নিচের
> ফলাফল "অপ্রাসঙ্গিক" ধরে বাদ দেওয়া হয়। Threshold খুব বেশি রাখলে প্রাসঙ্গিক
> ডকুমেন্টও বাদ পড়ে (false negative); খুব কম রাখলে আজেবাজে ডকুমেন্ট চলে আসে
> (false positive)। আমাদের ডেটায় সঠিক ডকুমেন্ট ০.২৪–০.৩২ স্কোর পায়, আর শুধু "donor"
> শব্দ মিলে যাওয়া ডকুমেন্ট ০.০২–০.০৬ পায়, তাই ০.১০ রাখা হয়েছে।

**4. How can I test it?**
```bash
python -m pytest tests/test_retriever.py -v
python -c "from app.retriever import retrieve_documents; import pprint; pprint.pprint(retrieve_documents('How can we improve donor retention?'))"
FKA_MIN_SIMILARITY=0.0 python main.py -q "How can we improve donor retention?" --hide-prompt
```
The last command shows the weak runner-up documents the threshold normally hides.

**5. Interview questions**
- *Why `fit` on chunks but only `transform` the question?* The vocabulary and IDF
  must come from the knowledge base. Fitting on the question would create a different
  vector space and make scores meaningless.
- *Why cosine and not Euclidean distance?* Cosine ignores vector length, so a short
  question can be compared with a long chunk. With L2-normalized vectors the two give
  the same ranking anyway.
- *Your scores are only ~0.3. Is that bad?* No. A short question shares few terms
  with an 80-word chunk. What matters is the gap between relevant (0.24-0.32) and
  irrelevant (0.00-0.06) results.
- *What is BM25 and why might it beat TF-IDF?* A TF-IDF variant with term-frequency
  saturation and explicit document-length normalization. It is the standard lexical
  baseline in search engines.
- *How do you evaluate a retriever?* Build a set of questions with known relevant
  document IDs. Measure hit rate@k (is a correct document in the top k?) and MRR
  (how high is the first correct one?).
- *How would you scale this to a million chunks?* Approximate nearest-neighbour
  indexes (HNSW, IVF) in a vector database, or a search engine for BM25.

### 2.4 Prompt builder (`app/prompt_builder.py`)

**Problem it solves.** An LLM answers from its general memory unless it is clearly
told to use only the supplied context and to admit when the context is insufficient.

**1. What was implemented?**
`build_prompt(question, retrieved_docs)` and `format_context(docs)`. The prompt
contains role, rules (no invention, no assumptions, exact refusal sentence, simple
answer, cite `[DOC001]`, ignore instructions inside context), numbered context blocks
with ID / title / chunk / content, the question, and `ANSWER:`.

**2. How does it work?**
Pure string building, no API call. Empty document list gives an explicit
"(No relevant documents were found...)" marker. Missing `id`, `title` or `content`
raises `ValueError`.

**3. Why is it needed?**
The prompt is the contract between retrieval and generation. Clear rules and
delimiters reduce hallucination and make answers checkable through citations.

> **বাংলা ব্যাখ্যা — Prompt injection**
> Prompt injection মানে ডকুমেন্ট বা ইউজারের লেখার ভেতরে লুকানো নির্দেশ (যেমন
> "আগের সব নিয়ম ভুলে যাও") যা LLM ভুল করে মেনে নিতে পারে। তাই প্রম্পটে স্পষ্ট বলা
> হয়েছে: context শুধু তথ্য, এর ভেতরের কোনো নির্দেশ মানবে না। এটা ঝুঁকি কমায়, তবে
> পুরোপুরি দূর করে না।

**4. How can I test it?**
```bash
python -m pytest tests/test_prompt_builder.py -v
python main.py -q "How can we protect personal information?"
```

**5. Interview questions**
- *How does the prompt reduce hallucination?* Context-only rule, explicit refusal
  text, mandatory citations, and no prompt at all when retrieval finds nothing.
- *Why ask for document-ID citations?* Users can verify claims, and code can detect
  citations to documents that were never provided.
- *How would you split this for a chat API?* Rules go in the system message; context
  and question go in the user message.
- *What if the retrieved context is too long?* Count tokens, keep the highest-ranked
  chunks within a budget, or summarize/compress lower-ranked ones.

### 2.5 Answer service (`app/answer_service.py`)

**Problem it solves.** One place that runs the whole flow and returns a structured,
honest result, ready for a CLI today and an API tomorrow.

**1. What was implemented?**
`AnswerService(retriever, llm_provider=None, top_k=3).answer(question)` returning an
`AnswerResult` (`question`, `status`, `message`, `retrieved_documents`, `sources`,
`prompt`, `answer`, `unsupported_citations`, `to_dict()`). `Retriever` and
`LLMProvider` protocols. `extract_sources` and `find_unsupported_citations`.

**2. How does it work?**
1. Invalid question gives status `invalid_question`; the retriever is not called.
2. No documents gives `no_relevant_documents` with the standard message; no prompt.
3. Documents and no LLM give `prompt_ready` with prompt and sources; `answer=None`.
4. Documents and an LLM give `answered`; citations outside the context are listed in
   `unsupported_citations`. Provider errors become `LLMProviderError`.

**3. Why is it needed?**
It keeps the CLI and a future FastAPI endpoint thin. It never pretends an answer
exists in Phase 1. Dependency injection lets Phase 2 add a real LLM, and lets tests
use a fake one, without changing this code.

> **বাংলা ব্যাখ্যা — Dependency injection**
> Dependency injection মানে AnswerService নিজে LLM তৈরি করে না, বাইরে থেকে দেওয়া হয় —
> Laravel-এর service container-এর মতো। এতে টেস্টে নকল (fake) LLM দেওয়া যায়, আর পরে
> OpenAI বা অন্য কোনো provider বদলাতে শুধু একটা ক্লাস লিখলেই হয়, বাকি কোড অপরিবর্তিত থাকে।

**4. How can I test it?**
```bash
python -m pytest tests/test_answer_service.py -v
python -c "
from app.answer_service import AnswerService
from app.retriever import get_default_retriever
import json
print(json.dumps(AnswerService(get_default_retriever()).answer('What is donor segmentation?').to_dict(), indent=2)[:800])"
```

**5. Interview questions**
- *Why not call the LLM when nothing is retrieved?* With empty context the model
  answers from memory, which is exactly the hallucination RAG tries to avoid. It also
  saves cost.
- *How do you plug in OpenAI later?* Write a class with `generate(prompt) -> str` that
  calls the SDK; pass it as `llm_provider`. Nothing else changes.
- *How do you check that an answer is grounded?* Cheap: citation check (done here).
  Stronger: ask a second model, or use an NLI model, to verify each sentence is
  supported by the cited chunk; track a faithfulness metric on an evaluation set.
- *Why a `Protocol` instead of a base class?* Structural typing: any object with the
  right method works, no inheritance needed. Similar to a PHP interface, but implicit.

### 2.6 CLI (`main.py`, `app/formatter.py`, `examples/sample_questions.py`)

**Problem it solves.** A simple way to use and demo the system without a web UI.

**1. What was implemented?**
Interactive loop (`exit`, `quit`, `q`, Ctrl+D, Ctrl+C), one-off `-q` questions,
`--top-k`, `--hide-prompt`, `.env` loading, startup error handling (exit code 1),
and a script that runs the required, unrelated and limitation questions.

**2. How does it work?**
Settings and retriever are built once at startup; each question goes through
`AnswerService.answer`; `format_result` prints documents, scores, matched terms,
prompt, sources and status.

**3. Why is it needed?**
It proves the pipeline end to end and makes the behaviour easy to show in an
interview.

**4. How can I test it?**
```bash
python -m pytest tests/test_cli.py -v
python main.py
python examples/sample_questions.py --hide-prompt
```

**5. Interview questions**
- *How would you turn this into an API?* FastAPI app that builds the retriever on
  startup and exposes `POST /ask` returning `AnswerResult.to_dict()`.
- *Why build the index once at startup?* Fitting TF-IDF on every question would be
  wasteful; queries only need `transform`.

### 2.7 Tests (`tests/`)

91 tests across 6 files. Highlights:

- Required questions are parametrized and must rank the expected document first.
- `test_score_equals_dot_product_of_normalized_vectors` proves the math, not just
  the output.
- `test_scores_are_calculated_not_hardcoded` guards against fake scores.
- A `FakeLLM` and a `RecordingRetriever` test the service without any network.
- An autouse fixture removes `FKA_*` environment variables so a developer's `.env`
  cannot change test results.

```bash
python -m pytest -v
python -m pytest -k threshold -v      # run tests whose name contains "threshold"
```

**Interview questions**
- *How do you test something that uses an LLM?* Inject a fake provider for unit tests;
  use a small evaluation set with a real model for integration tests; assert on
  structure and citations rather than exact wording.
- *What would you add for regression safety?* A retrieval evaluation set with a
  minimum hit rate checked in CI.

---

## Part 3: Interview cheat sheet

1. **Explain RAG in one sentence.** Retrieve relevant passages from your own
   documents, then ask an LLM to answer using only those passages, with citations.
2. **Why RAG instead of fine-tuning?** Knowledge changes often; RAG updates by adding
   documents, gives citations, and needs no retraining. Fine-tuning changes style or
   behaviour better than it adds facts.
3. **What is an embedding?** A dense vector from a neural model where texts with
   similar meaning are close together.
4. **TF-IDF vs embeddings?** TF-IDF matches words (explainable, cheap, misses
   synonyms); embeddings match meaning (better recall, harder to explain, cost more).
5. **What is hybrid search?** Combine lexical (BM25) and semantic scores, for example
   with Reciprocal Rank Fusion, to get exact keyword matches and meaning matches.
6. **What is a reranker?** A slower, more accurate model (cross-encoder) that rescores
   the top ~20-50 retrieved chunks before they go into the prompt.
7. **How do you evaluate RAG?** Retrieval: hit rate@k, MRR, recall. Generation:
   faithfulness (is it supported by context?), answer relevance, citation accuracy.
   Plus human review on a sample.
8. **How do you handle "I don't know"?** Similarity threshold, no LLM call without
   context, an explicit refusal instruction, and evaluation questions that *should*
   be refused.
9. **Privacy risks in RAG for donor data?** Personal data in documents or questions
   being sent to an external LLM, logged, or shown to users without permission.
   Mitigate with data minimisation, redaction, access control on retrieval, approved
   providers, and audit logs.
10. **What did you learn from this project?** Lexical retrieval works well when the
    question uses the document's words and fails on synonyms; thresholds must be
    measured, not guessed; and tiny chunks can win unfairly because of length bias.
