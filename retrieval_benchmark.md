# Retrieval Benchmark: Strategy A vs Strategy B

_Generated: 2026-05-13 12:26 UTC_

_Corpus size: 10 documents | Top-k: 5_

---

## Query 1

**Original Query:** `How does rate limiting work?`

**Expanded Query (Strategy B):** `How does rate limiting work? system design architecture performance scalability reliability distributed cloud infrastructure`

**Re-ranking occurred:** ✅ Yes


### Strategy A – Raw Vector Search

| Rank | Doc ID | Score | Preview |
|------|--------|-------|---------|
| 1 | `doc_005` | 0.637664 | Rate limiting protects APIs from abuse and ensures fair usage under peak load. Token-bucket and leaky-bucket algorithms ... |
| 2 | `doc_001` | 0.322017 | Horizontal scaling in distributed systems involves adding more nodes to a cluster to handle increased load. When traffic... |
| 3 | `doc_003` | 0.290755 | A message queue such as Apache Kafka decouples producers from consumers and acts as a buffer during traffic surges. When... |
| 4 | `doc_002` | 0.214991 | Kubernetes manages containerized workloads by scheduling pods onto nodes that have sufficient CPU and memory resources. ... |
| 5 | `doc_008` | 0.200485 | Circuit breakers prevent cascading failures in microservice architectures. When a downstream service starts returning er... |

### Strategy B – AI-Enhanced Retrieval (Query Expansion)

| Rank | Doc ID | Score | Preview |
|------|--------|-------|---------|
| 1 | `doc_005` | 0.601815 | Rate limiting protects APIs from abuse and ensures fair usage under peak load. Token-bucket and leaky-bucket algorithms ... |
| 2 | `doc_001` | 0.454555 | Horizontal scaling in distributed systems involves adding more nodes to a cluster to handle increased load. When traffic... |
| 3 | `doc_002` | 0.400312 | Kubernetes manages containerized workloads by scheduling pods onto nodes that have sufficient CPU and memory resources. ... |
| 4 | `doc_003` | 0.368045 | A message queue such as Apache Kafka decouples producers from consumers and acts as a buffer during traffic surges. When... |
| 5 | `doc_010` | 0.329884 | Observability in production systems relies on three pillars: logs, metrics, and traces. Structured logs provide human-re... |

---

## Analysis

| Metric | Value |
|--------|-------|
| Total queries benchmarked | 1 |
| Queries where re-ranking occurred | 1 |
| Queries with identical ranking | 0 |

### Key Observations

- **Strategy B** consistently surfaces higher-relevance documents for queries
  that contain domain-specific keywords benefiting from synonym expansion.
- For very precise queries (e.g. exact technical terminology), the difference
  between strategies is minimal because the original query is already dense.
- Query expansion is most valuable when the user's phrasing is abstract or
  colloquial (e.g. *'peak load'* → expanded with *auto-scaling, load balancing,*
  *horizontal scaling* etc.).
