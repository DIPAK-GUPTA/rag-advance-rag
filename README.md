# Context-Aware Retrieval Engine

Senior Gen AI Assessment – Semantic RAG & Vector Search

---

## Overview

A local **Retrieval-Augmented Generation (RAG) pipeline** that ingests raw technical documents, generates dense vector embeddings, and benchmarks two retrieval strategies:

| Strategy | Description |
|----------|-------------|
| **A – Raw Vector Search** | The user query is embedded directly and matched against the indexed corpus via cosine similarity. |
| **B – AI-Enhanced Retrieval** | A (mocked) `GenerativeModel` rewrites/expands the query into a richer, keyword-dense form before embedding and retrieval. |

---

## Project Structure

```
.
├── data/
│   └── corpus.py              # 10 technical paragraphs (knowledge base)
├── src/
│   ├── __init__.py
│   ├── embedding.py           # EmbeddingModel – local ST + Vertex AI mock
│   ├── storage.py             # FaissVectorStore (FAISS / NumPy backend)
│   ├── retrieval.py           # QueryExpander, Retriever (Strategy A & B)
│   └── pipeline.py            # RAGPipeline orchestration class
├── tests/
│   ├── conftest.py            # Shared fixtures (mock embedder, expander)
│   ├── test_embedding.py      # Embedding + Vertex AI SDK mock tests
│   ├── test_storage.py        # FaissVectorStore tests
│   ├── test_retrieval.py      # QueryExpander + Retriever tests
│   ├── test_pipeline.py       # RAGPipeline integration tests
│   └── test_benchmark.py      # Benchmark runner & output tests
├── benchmark.py               # CLI benchmarking script
├── retrieval_benchmark.md     # Dev evidence – Strategy A vs B output
├── benchmark_results.json     # Full structured JSON benchmark output
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the benchmark

```bash
python3 benchmark.py
```

Output:
- A Rich colour table in the terminal comparing both strategies.
- `retrieval_benchmark.md` – Markdown report (dev evidence).
- `benchmark_results.json` – Full structured JSON report.

#### Custom queries / options

```bash
python3 benchmark.py --queries "How does rate limiting work?" "What is a circuit breaker?" --top-k 5
python3 benchmark.py --no-json      # skip JSON output
python3 benchmark.py --no-markdown  # skip Markdown output
```

### 3. Run the test suite

```bash
python3 -m pytest tests/ -v
```

All **80 tests** should pass with no external network calls required.

---

## Technical Design

### Embedding Model (`src/embedding.py`)

`EmbeddingModel` provides the **same public interface** as
`vertexai.language_models.TextEmbeddingModel`:

```python
model.get_embeddings(["text 1", "text 2"])  # → List[TextEmbeddingValue]
model.embed_single("text")                   # → np.ndarray
```

Locally it delegates to `sentence-transformers` (`all-MiniLM-L6-v2`), which
simulates `textembedding-gecko` behaviour.  Setting `USE_VERTEX_AI=1` in the
environment transparently switches to the real GCP endpoint with **zero code
changes**.

### Vector Store (`src/storage.py`)

`FaissVectorStore` wraps a **FAISS `IndexFlatIP`** index (Inner Product on
L2-normalised vectors = cosine similarity).  A pure-NumPy fallback is available
for CI environments without a native FAISS build.

Supports `add`, `add_batch`, `search`, `save`, and `load` operations.

### Retrieval (`src/retrieval.py`)

**`QueryExpander`** wraps `vertexai.generative_models.GenerativeModel`.
Locally it uses a deterministic keyword-expansion dictionary keyed on terms such
as `"peak load"`, `"cache"`, `"failure"`, `"kubernetes"`, etc.  Setting
`USE_VERTEX_AI=1` routes to the live Gemini endpoint.

**`Retriever`** exposes:
- `retrieve_strategy_a(query, top_k)` – raw vector search.
- `retrieve_strategy_b(query, top_k)` – expand → embed → search.

### Pipeline (`src/pipeline.py`)

`RAGPipeline` is the main orchestration class:

```python
pipeline = RAGPipeline()
pipeline.ingest(CORPUS)                          # embed + index all documents
results_a = pipeline.query_strategy_a("...")     # Strategy A
results_b, expanded = pipeline.query_strategy_b("...")  # Strategy B
pipeline.save("./index")                         # persist to disk
pipeline.load("./index")                         # restore from disk
```

---

## Similarity Metric: Cosine vs Euclidean

### Why Cosine Similarity?

| Criterion | Cosine | Euclidean |
|-----------|--------|-----------|
| Sensitive to vector magnitude? | No (direction only) | Yes |
| Works well with text embeddings? | **Yes** | Partially |
| Affected by document length? | No | Yes (longer → larger norm → biased) |
| Standard for semantic search? | **Yes** | Not standard |

Text embedding models (including sentence-transformers and gecko) produce vectors
where **direction encodes meaning** and magnitude is largely an artefact of the
model's normalisation.  Cosine similarity correctly ignores magnitude and
measures pure semantic alignment.

Euclidean distance can still be useful when the embedding distribution is
controlled (e.g., unit-norm spherical embeddings – at which point
`|a - b|² = 2 - 2·cos(a,b)`, making them equivalent).  FAISS's `IndexFlatIP`
on L2-normalised vectors provides exact cosine similarity in O(1) additional
computation.

---

## Migrating to Vertex AI Vector Search (Matching Engine) in Production

### Architecture Overview

```
                  ┌─────────────────────────────────┐
  Ingest          │  Cloud Storage bucket            │
  ─────────────▶  │  gs://my-bucket/embeddings/      │
                  │  (JSON / Avro embedding files)   │
                  └────────────┬────────────────────┘
                               │ create/update Index
                               ▼
                  ┌─────────────────────────────────┐
  Deploy          │  Vertex AI Vector Search Index   │
  ─────────────▶  │  (Approximate Nearest Neighbor)  │
                  └────────────┬────────────────────┘
                               │ deploy
                               ▼
                  ┌─────────────────────────────────┐
  Query           │  IndexEndpoint                   │
  ─────────────▶  │  match(queries, num_neighbors=k) │
                  └─────────────────────────────────┘
```

### Step-by-step Migration

1. **Replace `EmbeddingModel`**  
   Set `USE_VERTEX_AI=1` and ensure `google-cloud-aiplatform` is installed.
   The existing `EmbeddingModel` class already delegates to
   `vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`.

2. **Upload embeddings to Cloud Storage**  
   After ingestion, serialise embedding vectors + metadata to JSONL:
   ```json
   {"id": "doc_001", "embedding": [0.12, -0.34, ...]}
   ```
   Upload with `gsutil cp embeddings.jsonl gs://my-bucket/embeddings/`.

3. **Create the Matching Engine Index**  
   ```python
   from google.cloud import aiplatform
   aiplatform.init(project="my-project", location="us-central1")
   index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
       display_name="rag-index",
       contents_delta_uri="gs://my-bucket/embeddings/",
       dimensions=768,          # gecko output dim
       approximate_neighbors_count=150,
       distance_measure_type="COSINE_DISTANCE",
   )
   ```

4. **Deploy to an IndexEndpoint**  
   ```python
   endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
       display_name="rag-endpoint", public_endpoint_enabled=True
   )
   endpoint.deploy_index(index=index, deployed_index_id="rag_v1")
   ```

5. **Replace `FaissVectorStore.search`**  
   ```python
   response = endpoint.match(
       deployed_index_id="rag_v1",
       queries=[query_embedding.tolist()],
       num_neighbors=top_k,
   )
   ```

6. **Real-time updates**  
   Use `index.upsert_datapoints()` for streaming updates or schedule batch
   re-indexing via Cloud Scheduler + Cloud Run.

7. **Replace `QueryExpander`**  
   Set `USE_VERTEX_AI=1`; the existing class already delegates to
   `vertexai.generative_models.GenerativeModel("gemini-pro")`.

The local `RAGPipeline` interface (`ingest`, `query_strategy_a`,
`query_strategy_b`) remains **unchanged** – only the injected backends swap.

---

## Running Tests with Coverage

```bash
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_VERTEX_AI` | `0` | Set to `1` to use real GCP Vertex AI endpoints |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Local ST model name |
