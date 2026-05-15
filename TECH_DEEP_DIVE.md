# Technical Deep Dive — RAG Pipeline Tech Stack

> **Query under analysis:**
> ```
> python3 benchmark.py --queries "How does rate limiting work?" --top-k 5
> ```

---

## Table of Contents

1. [Benchmark Output for This Query](#1-benchmark-output-for-this-query)
2. [First Principles: How the Pipeline Works End-to-End](#2-first-principles-how-the-pipeline-works-end-to-end)
3. [Vector Database: FAISS](#3-vector-database-faiss)
4. [Embedding Model: sentence-transformers (all-MiniLM-L6-v2)](#4-embedding-model-sentence-transformers-all-minilm-l6-v2)
5. [Similarity Algorithm: Cosine via IndexFlatIP](#5-similarity-algorithm-cosine-via-indexflatip)
6. [Strategy A — Raw Vector Search](#6-strategy-a--raw-vector-search)
7. [Strategy B — AI-Enhanced Retrieval (Query Expansion)](#7-strategy-b--ai-enhanced-retrieval-query-expansion)
8. [Re-ranking: What Happened and Why](#8-re-ranking-what-happened-and-why)
9. [GCP Mock Layer (vertexai SDK)](#9-gcp-mock-layer-vertexai-sdk)
10. [Alternative Tech Choices](#10-alternative-tech-choices)
11. [Summary Decision Table](#11-summary-decision-table)

---

## 1. Benchmark Output for This Query

```
Query        : How does rate limiting work?
Expanded Q   : How does rate limiting work? system design architecture
               performance scalability reliability distributed cloud infrastructure
Re-ranking   : YES
```

| Rank | Strategy A — Raw Vector Search | Score A  | Strategy B — AI-Enhanced | Score B  |
|------|-------------------------------|----------|--------------------------|----------|
| 1    | `doc_005` Rate limiting protects APIs… | 0.637664 | `doc_005` Rate limiting protects APIs… | 0.601815 |
| 2    | `doc_001` Horizontal scaling in distributed… | 0.322017 | `doc_001` Horizontal scaling in distributed… | 0.454555 |
| 3    | `doc_003` A message queue such as Kafka… | 0.290755 | `doc_002` Kubernetes manages containerized… | 0.400312 |
| 4    | `doc_002` Kubernetes manages containerized… | 0.214991 | `doc_003` A message queue such as Kafka… | 0.368045 |
| 5    | `doc_008` Circuit breakers prevent cascading… | 0.200485 | `doc_010` Observability in production systems… | 0.329884 |

**Observation:** The top result is identical in both strategies (rate limiting doc is clearly the most relevant), but Strategy B reorders ranks 3–5. `doc_002` (Kubernetes) and `doc_010` (Observability) get promoted because the expanded query adds architectural/scalability terms that pull their embeddings closer to the query vector.

---

## 2. First Principles: How the Pipeline Works End-to-End

```
INGEST PHASE
────────────
 Raw text (10 docs)
      │
      ▼
 EmbeddingModel.get_embeddings(texts)
      │   sentence-transformers encodes text → 384-dim float32 vector
      │   L2-normalisation applied (unit sphere)
      ▼
 FaissVectorStore.add_batch(docs)
      │   FAISS IndexFlatIP stores raw float32 matrix
      ▼
 Index ready in memory

QUERY PHASE – STRATEGY A
─────────────────────────
 User query string
      │
      ▼
 EmbeddingModel.embed_single(query)
      │   Same encoder, same 384-dim space
      ▼
 FaissVectorStore.search(query_vec, top_k=5)
      │   FAISS computes Inner Product (≡ cosine on unit vectors)
      │   Returns indices + scores sorted descending
      ▼
 Top-5 RetrievalResult list

QUERY PHASE – STRATEGY B
─────────────────────────
 User query string
      │
      ▼
 QueryExpander.expand(query)         ← (mocked) GenerativeModel call
      │   Appends synonym clusters / architectural terms
      ▼
 Expanded query string
      │
      ▼
 EmbeddingModel.embed_single(expanded_query)
      │   Richer token surface → different 384-dim vector
      ▼
 FaissVectorStore.search(expanded_vec, top_k=5)
      ▼
 Top-5 RetrievalResult list  (different ranking from A)
```

The key insight: **the embedding space is fixed after training**. The only lever available at query time is *what text you feed the encoder*. Strategy B moves the query vector to a different position in that fixed space by making the input text richer.

---

## 3. Vector Database: FAISS

### What is it?

**FAISS** (Facebook AI Similarity Search) is a C++ library (with Python bindings) developed by Meta Research for efficient similarity search over dense floating-point vectors. It is one of the most widely used ANN (Approximate Nearest Neighbour) libraries in production ML systems.

### Which index is used here?

```python
# src/storage.py  line 66
self._index = faiss.IndexFlatIP(embedding_dim)
```

`IndexFlatIP` = **Flat Inner Product** index.

- **Flat** means it stores every vector as-is with no compression.
- **IP** (Inner Product) is the distance metric.
- On L2-normalised vectors: `IP(a, b) = a · b = cos(θ)` — so it computes exact cosine similarity.

### Why FAISS?

| Criterion | Reasoning |
|-----------|-----------|
| Speed | Vectorised BLAS operations; easily 100–1000× faster than pure Python loops over the same data |
| Correctness | `IndexFlatIP` is **exact** (no approximation) — correct for a 10-doc corpus where precision matters more than latency |
| Ecosystem | Battle-tested in production at Meta, Airbnb, Spotify, etc. |
| Local operation | No network calls, no server, no Docker — just a pip package |
| Production path | FAISS indexes can be sharded and served via Triton or wrapped in Vertex AI |
| GPU support | Drop-in `faiss-gpu` upgrade when needed |

### FAISS Pros

- Exact search (`IndexFlatIP`) — guaranteed to return the true top-k
- Supports many index types: `IVF`, `HNSW`, `PQ` for billion-scale ANN
- Batch operations are highly optimised (SIMD/AVX2 instructions)
- Persistence: `write_index` / `read_index` for zero-cost serialisation
- In-process: no network latency added

### FAISS Cons

- **In-memory only** — entire index must fit in RAM
- No built-in metadata filtering (you manage doc IDs externally)
- No native real-time update support — rebuilding an `IndexFlat` is O(N)
- `IndexFlat` search is O(N·d) — becomes slow for N > ~10M documents
- No distributed sharding out of the box

### NumPy Fallback

When FAISS is not installed (CI, minimal Docker image), the codebase falls back to:
```python
# src/storage.py  lines 137-142
matrix = np.vstack(self._embeddings)   # (N, dim)
sims = (matrix @ q.T).flatten()        # dot product on unit vecs = cosine
top_indices = np.argsort(sims)[::-1][:k]
```
This is mathematically identical but ~10–100× slower for large N because it is not SIMD-accelerated.

---

## 4. Embedding Model: sentence-transformers (all-MiniLM-L6-v2)

### What is it?

`all-MiniLM-L6-v2` is a **distilled** transformer model from the `sentence-transformers` library. It is a 6-layer BERT-based model fine-tuned via **knowledge distillation** from a larger teacher model on 1-billion sentence-pair training examples.

- **Output:** 384-dimensional dense float32 vector per input text
- **Max tokens:** 256 word-pieces (truncates longer inputs)
- **Training objective:** Contrastive learning — semantically similar texts are pulled close together in the 384-dim space; dissimilar texts are pushed apart
- **Normalisation:** `normalize_embeddings=True` applied in `get_embeddings()` — all vectors have unit L2-norm

### Why this model?

| Criterion | Reasoning |
|-----------|-----------|
| Simulates gecko | Assessment requires simulating `textembedding-gecko`. MiniLM is the same family (transformer → dense pooled vector). The pipeline interface is identical; swapping is a one-line env var change |
| Speed | ~14k sentences/second on CPU — fast enough for a 10-doc corpus to be near-instant |
| Quality | MTEB leaderboard score of ~56 — excellent for its size |
| Zero external calls | Fully local, no API key required |
| Small footprint | ~80 MB model weights, lazy-loaded on first call |

### How it produces a vector

```
Input: "How does rate limiting work?"
       │
       ▼
Tokenizer  →  ["how", "does", "rate", "limiting", "work", "?"]  →  token IDs
       │
       ▼
6-layer Transformer  →  contextualised token embeddings  (384-dim each)
       │
       ▼
Mean Pooling  →  single 384-dim sentence vector
       │
       ▼
L2 Normalisation  →  unit vector on the 384-dim hypersphere
```

---

## 5. Similarity Algorithm: Cosine via IndexFlatIP

### The Math

For two unit vectors **a** and **b**:

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
                        = a · b          (since |a| = |b| = 1)
                        = Inner Product(a, b)
```

This is why `IndexFlatIP` on L2-normalised vectors **exactly computes cosine similarity** — no division needed.

### Cosine vs Euclidean — First Principles

The fundamental question is: *what property of two vectors encodes semantic relatedness?*

```
Cosine focuses on DIRECTION (angle between vectors)
Euclidean focuses on DISTANCE (magnitude of the difference vector)

For text embeddings:
  - Two documents about "rate limiting" will point in similar DIRECTIONS
    regardless of their length or how many times they repeat keywords.
  - Euclidean distance is sensitive to vector magnitude. A longer document
    that mentions "rate limiting" 10 times will have a larger-norm embedding
    than a short one — making it artificially "far" from the query even
    though its meaning is identical.
```

Proof that they agree on unit-norm vectors:

```
|a - b|² = |a|² - 2(a·b) + |b|²
         = 1 - 2·cos(θ) + 1
         = 2 - 2·cos(θ)

So: Euclidean_distance² = 2 - 2·cosine_similarity

They are monotonically equivalent on unit sphere — ranking is identical.
```

**Conclusion:** For normalised embeddings, both metrics produce the same ranking. The choice of cosine is idiomatic in NLP and avoids any magnitude-bias when embeddings are not normalised.

### Why Not Dot Product Without Normalisation?

Raw dot product without normalisation biases toward **high-magnitude vectors** (e.g., long documents). By normalising to unit norm before indexing, every document gets equal weight in the similarity computation — only its directional meaning counts.

---

## 6. Strategy A — Raw Vector Search

### Algorithm

```
Step 1: query_text → embed_single(query_text) → q ∈ R^384
Step 2: FAISS.search(q, top_k=5)
         for each doc_i with vector d_i:
             score_i = q · d_i  (= cosine similarity)
         return top-5 by score descending
```

**Time complexity:** O(N × d) per query, where N = corpus size, d = 384

### What it retrieved for "How does rate limiting work?"

| Rank | Doc | Score | Why |
|------|-----|-------|-----|
| 1 | `doc_005` Rate limiting | 0.638 | Exact semantic match — "rate limiting" appears in both |
| 2 | `doc_001` Horizontal scaling | 0.322 | Both discuss traffic and peak load — contextually adjacent |
| 3 | `doc_003` Kafka queues | 0.291 | Both relate to buffering/overflow under traffic |
| 4 | `doc_002` Kubernetes HPA | 0.215 | Both involve request-rate management |
| 5 | `doc_008` Circuit breakers | 0.200 | Both are API protection / resilience patterns |

### Pros

- Simple and deterministic — same query always returns same result
- No additional latency from LLM call
- Scores are interpretable (cosine similarity has a bounded range of [-1, 1])
- Works well when the user query is already precise and technical

### Cons

- **Vocabulary mismatch problem:** If the user asks *"throttling"* instead of *"rate limiting"*, the query vector shifts and might miss `doc_005` entirely — even though they mean the same thing
- The query is treated as a single flat text — no semantic decomposition
- Short queries embed into a noisy area of the vector space (few tokens → less informative mean-pooled vector)

---

## 7. Strategy B — AI-Enhanced Retrieval (Query Expansion)

### Algorithm

```
Step 1: query_text → QueryExpander.expand(query_text) → expanded_text
         (mocked GenerativeModel or keyword heuristic dictionary)

Step 2: expanded_text → embed_single(expanded_text) → q' ∈ R^384
         q' ≠ q because the input text is richer (more tokens → different mean pool)

Step 3: FAISS.search(q', top_k=5)
         Same index, different query vector → potentially different ranking
```

### Query Expansion in Detail

**Original:** `"How does rate limiting work?"`

**Expanded:** `"How does rate limiting work? system design architecture performance scalability reliability distributed cloud infrastructure"`

The query *did not match* any keyword in the local expansion dictionary (the dictionary keys are `"peak load"`, `"cache"`, `"failure"`, `"embedding"`, `"kubernetes"`, `"database"`). So the **generic fallback** fired:

```python
# src/retrieval.py  line 126-129
return (
    f"{query} system design architecture performance scalability "
    "reliability distributed cloud infrastructure"
)
```

This is the "generic expansion" — it broadens the query into a general distributed-systems vocabulary, which is why ranks 3–5 shifted toward architecture-related documents (`doc_002` Kubernetes, `doc_010` Observability).

### Why Query Expansion Works — Intuition

```
Without expansion:
  query vector  ≈  centroid of ["how", "does", "rate", "limiting", "work"]
  → lands in a specific spot in the 384-dim space

With expansion:
  query vector  ≈  centroid of ["how", "does", "rate", "limiting", "work",
                                 "system", "design", "architecture", ...]
  → shifts toward a broader cloud-infrastructure cluster
  → retrieves documents whose embeddings overlap with any of those concepts
```

This is effective because:
1. The embedding is the **mean** of all token representations
2. Adding more relevant tokens pulls the mean vector toward a richer semantic neighbourhood
3. Documents that would have been at rank 6 or 7 now appear in the top 5

### What Changed for This Query

| Rank | Strategy A | Strategy B | Changed? |
|------|-----------|------------|----------|
| 1 | `doc_005` Rate limiting (0.638) | `doc_005` Rate limiting (0.602) | Same doc, lower score (diluted by extra tokens) |
| 2 | `doc_001` Horizontal scaling (0.322) | `doc_001` Horizontal scaling (0.455) | Same doc, **higher score** (expansion added scaling terms) |
| 3 | `doc_003` Kafka (0.291) | **`doc_002` Kubernetes (0.400)** | Promoted |
| 4 | `doc_002` Kubernetes (0.215) | `doc_003` Kafka (0.368) | Demoted |
| 5 | `doc_008` Circuit breakers (0.200) | **`doc_010` Observability (0.330)** | Replaced |

Notable: the top-1 score **dropped** from 0.638 → 0.602. This is expected — adding general terms dilutes the specific "rate limiting" signal. The trade-off is intentional: you get broader coverage at the cost of top-1 precision.

### Pros

- Overcomes vocabulary mismatch / query under-specification
- Brings in synonyms and related concepts the user didn't type
- Can surface documents the user didn't know to ask for
- With a real LLM (Gemini/GPT-4), expansion can be deeply context-aware

### Cons

- **Topic drift:** adding broad terms can introduce noise — for this query, `doc_010` (Observability) appearing at rank 5 is arguably off-topic
- Extra latency from the LLM call (in production with real Vertex AI)
- Non-deterministic with a real LLM — same query can return different expansions
- Hurts precision for already-precise queries (top-1 score dilution seen above)

---

## 8. Re-ranking: What Happened and Why

### Is there a dedicated re-ranking step?

**No.** The current pipeline does not have a separate re-ranking model. The ordering in both strategies comes purely from cosine similarity scores produced by FAISS. The "re-ranking" observed is simply that a different query vector (after expansion) produces different cosine scores.

```
Strategy A ranking ← cosine(q_original, doc_i)   for all i
Strategy B ranking ← cosine(q_expanded, doc_i)   for all i
```

They share the same FAISS index — only the query vector changes.

### What a Real Re-ranker Would Look Like

A production RAG pipeline typically adds a **cross-encoder re-ranker** as a third stage:

```
Stage 1 (Retrieval)  →  top-50 candidates via fast ANN (FAISS / Vertex)
Stage 2 (Re-ranking) →  cross-encoder scores each (query, doc) pair
                         and re-orders the top-50 down to top-5
Stage 3 (Generation) →  LLM reads top-5 + query → final answer
```

A cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) jointly encodes the query and document together:

```
Input:  [CLS] query [SEP] document [SEP]
Output: scalar relevance score  (0.0 – 1.0)
```

This is more accurate than cosine similarity because it sees both texts simultaneously and can model their **interaction** — not just their independent directions in the vector space.

### Re-ranking Pros / Cons

| | Pros | Cons |
|-|------|------|
| Cross-encoder re-ranker | Significantly higher precision; captures query-document interaction | Slow — O(k) LLM forward passes; impractical for k > 100 |
| BM25 + cosine fusion | Combines keyword and semantic signals | Requires tuning the fusion weight α |
| Cohere Rerank API | Production-ready, fast, high quality | External API call; cost per query |
| ColBERT late interaction | Fine-grained token-level matching | Larger index size; complex deployment |

---

## 9. GCP Mock Layer (vertexai SDK)

### Why a Mock?

The assessment requires the code to reference `vertexai.language_models.TextEmbeddingModel` and `vertexai.generative_models.GenerativeModel` so it is reviewable as a GCP-native solution. Since the real SDK requires billing-enabled GCP credentials, we mock both interfaces.

### How the Mock Works

**Embedding Model Mock (`src/embedding.py`):**

```python
try:
    from vertexai.language_models import TextEmbeddingModel as _VertexTextEmbeddingModel
    _VERTEX_AVAILABLE = True
except ImportError:
    _VERTEX_AVAILABLE = False
    _VertexTextEmbeddingModel = MagicMock(name="TextEmbeddingModel")
```

When `USE_VERTEX_AI=0` (default), `EmbeddingModel` ignores `_VertexTextEmbeddingModel` entirely and delegates to sentence-transformers. When `USE_VERTEX_AI=1` and the SDK is installed, the same `EmbeddingModel` class calls `_VertexTextEmbeddingModel.from_pretrained("textembedding-gecko@003")` — no code changes needed.

**Generative Model Mock (`src/retrieval.py`):**

```python
try:
    from vertexai.generative_models import GenerativeModel as _VertexGenerativeModel
    _VERTEX_GEN_AVAILABLE = True
except ImportError:
    _VERTEX_GEN_AVAILABLE = False
    _VertexGenerativeModel = MagicMock(name="GenerativeModel")
```

The mock mirrors the real SDK call: `model.generate_content(prompt)` → `.text`. In tests, `pytest` patches this mock to return controlled strings, enabling deterministic test execution.

---

## 10. Alternative Tech Choices

### Alternative Vector Databases

| Library | Type | Best For | Why Not Used Here |
|---------|------|----------|--------------------|
| **ChromaDB** | Embedded / server | Dev prototyping, metadata filtering | More dependencies; FAISS is leaner for this scale |
| **Weaviate** | Full server | Production multi-modal search | Requires a running Docker container |
| **Pinecone** | Managed cloud | Production at scale | External API; costs money; not local |
| **Qdrant** | Embedded / server | Production with filtering | Excellent choice but heavier than FAISS |
| **pgvector** (Postgres) | SQL extension | When you already use Postgres | SQL overhead; not specialised for pure ANN |
| **Milvus** | Distributed server | Billion-scale production | Heavy ops burden |
| **NumPy (fallback)** | In-process array | CI / tests / tiny corpora | No SIMD; scales poorly past ~100K docs |

**For this assessment:** FAISS `IndexFlatIP` is the right tool — exact, fast, zero infrastructure, and it's the underlying engine Vertex AI Vector Search (Matching Engine) exposes in its managed form.

### Alternative Similarity Metrics

| Metric | Formula | When to Use | When to Avoid |
|--------|---------|-------------|---------------|
| **Cosine** (used) | `a·b / (|a||b|)` | Text embeddings, normalised vectors | When magnitude is meaningful |
| **Euclidean (L2)** | `√Σ(aᵢ-bᵢ)²` | Image/audio embeddings, spatial data | Text with variable-length docs (magnitude skew) |
| **Dot Product** | `a·b` | When you want magnitude to influence score (e.g., importance-weighted) | High-norm outliers dominate |
| **Manhattan (L1)** | `Σ|aᵢ-bᵢ|` | Sparse vectors, robust to outliers | Dense float vectors (cosine beats it) |
| **Hamming** | bit XOR count | Binary / hashed embeddings | Continuous float vectors |

### Alternative Embedding Models

| Model | Dim | Quality (MTEB) | Speed | When to Use |
|-------|-----|---------------|-------|-------------|
| **all-MiniLM-L6-v2** (used) | 384 | ~56 | ~14k/s CPU | Dev / assessment; fast & good |
| `all-mpnet-base-v2` | 768 | ~57 | ~2k/s CPU | Better quality, acceptable speed |
| `BAAI/bge-large-en-v1.5` | 1024 | ~64 | ~500/s CPU | Production high-quality retrieval |
| `text-embedding-3-large` (OpenAI) | 3072 | ~64 | API call | External API; best quality |
| `textembedding-gecko@003` (GCP) | 768 | ~62 | API call | GCP-native production |
| `E5-large-v2` | 1024 | ~63 | ~500/s CPU | Open source production |

### Alternative Query Expansion Strategies

| Strategy | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **Keyword dictionary** (used local fallback) | Hard-coded synonym clusters | Deterministic, fast, zero cost | Brittle; doesn't generalise |
| **LLM rewriting** (Strategy B production path) | Gemini/GPT rewrites the query | Deep semantic understanding | Latency + cost per query |
| **HyDE (Hypothetical Document Embeddings)** | LLM generates a fake answer document, embed it instead of the query | Query vector aligns with answer space | LLM call required; hallucination risk |
| **BM25 + Dense fusion (Hybrid)** | Run BM25 (keyword) and cosine (dense) in parallel, merge scores | Captures both exact keyword and semantic matches | α-weight tuning required |
| **Pseudo-Relevance Feedback (PRF)** | Fetch top-3 results, extract keywords, add to query, re-search | No LLM needed; automatic | Two search round-trips; top-3 may be noisy |
| **Synonyms via WordNet** | Look up synonyms in lexical database | No LLM; deterministic | Doesn't understand domain context |

---

## 11. Summary Decision Table

| Component | Chosen | Key Reason | Best Production Alternative |
|-----------|--------|------------|----------------------------|
| Vector DB | FAISS `IndexFlatIP` | Exact, fast, zero infra | Vertex AI Vector Search (Matching Engine) |
| Embedding model | `all-MiniLM-L6-v2` | Fast, local, mocks gecko API | `textembedding-gecko@003` via Vertex AI |
| Similarity metric | Cosine (via Inner Product on unit vecs) | Direction-only; length-invariant; NLP standard | Same — cosine is correct for text |
| Strategy A algo | Exact ANN via FAISS | Small corpus; correctness over speed | HNSW / IVF-PQ for > 1M docs |
| Strategy B expansion | Keyword heuristic dict + generic fallback | Deterministic, no LLM call | `GenerativeModel("gemini-pro")` via Vertex AI |
| Re-ranking | None (score-based only) | Sufficient at 10-doc scale | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| GCP SDK | Mocked via `unittest.mock` | No credentials in assessment | Real `google-cloud-aiplatform` SDK |

---

*Generated from:* `src/embedding.py`, `src/storage.py`, `src/retrieval.py`, `src/pipeline.py`
