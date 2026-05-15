"""
Technical paragraph corpus used as the RAG knowledge base.
10 paragraphs covering distributed systems, cloud infrastructure, and ML topics.
"""

CORPUS = [
    {
        "id": "doc_001",
        "text": (
            "Horizontal scaling in distributed systems involves adding more nodes to a cluster "
            "to handle increased load. When traffic spikes during peak hours, an auto-scaling "
            "group automatically provisions new instances based on CPU or request-rate thresholds. "
            "Load balancers distribute requests across healthy nodes using round-robin or "
            "least-connections algorithms, ensuring no single server becomes a bottleneck."
        ),
    },
    {
        "id": "doc_002",
        "text": (
            "Kubernetes manages containerized workloads by scheduling pods onto nodes that have "
            "sufficient CPU and memory resources. The Horizontal Pod Autoscaler (HPA) monitors "
            "custom metrics such as requests-per-second and automatically scales the number of "
            "replicas. During high-demand periods the cluster expands, and during low-traffic "
            "windows it scales down to reduce costs."
        ),
    },
    {
        "id": "doc_003",
        "text": (
            "A message queue such as Apache Kafka decouples producers from consumers and acts as "
            "a buffer during traffic surges. When the ingestion rate exceeds the processing "
            "capacity, messages accumulate in partitioned topics. Consumer groups read from "
            "these partitions in parallel, providing back-pressure control and preventing "
            "downstream services from being overwhelmed under peak load conditions."
        ),
    },
    {
        "id": "doc_004",
        "text": (
            "Cosine similarity measures the angle between two vectors in a high-dimensional space, "
            "making it ideal for comparing text embeddings where the magnitude is less important "
            "than the direction. In contrast, Euclidean distance measures the straight-line "
            "distance between points and can be skewed by vector length. For semantic search "
            "applications, cosine similarity is generally preferred because normalized embeddings "
            "yield more meaningful relevance scores."
        ),
    },
    {
        "id": "doc_005",
        "text": (
            "Rate limiting protects APIs from abuse and ensures fair usage under peak load. "
            "Token-bucket and leaky-bucket algorithms are common implementations. When a client "
            "exceeds its quota, the server returns HTTP 429 Too Many Requests. "
            "Distributed rate limiters use Redis to share counters across multiple API gateway "
            "instances, maintaining consistency even when traffic is routed through different nodes."
        ),
    },
    {
        "id": "doc_006",
        "text": (
            "Caching strategies such as read-through and write-through significantly reduce "
            "database load during high-traffic events. An in-memory cache like Redis or Memcached "
            "stores frequently accessed query results close to the application layer. Cache "
            "eviction policies—LRU (Least Recently Used) and LFU (Least Frequently Used)—"
            "determine which entries are removed when the cache reaches capacity, balancing "
            "freshness and hit-rate performance."
        ),
    },
    {
        "id": "doc_007",
        "text": (
            "Vertex AI Vector Search (formerly Matching Engine) provides a fully managed, "
            "low-latency approximate nearest-neighbor (ANN) service on Google Cloud. Engineers "
            "upload embeddings as JSON or Avro files to a Cloud Storage bucket, then create an "
            "IndexEndpoint to serve queries. The service supports both streaming and batch "
            "updates, making it suitable for production RAG pipelines that require real-time "
            "knowledge base updates."
        ),
    },
    {
        "id": "doc_008",
        "text": (
            "Circuit breakers prevent cascading failures in microservice architectures. When a "
            "downstream service starts returning errors or timing out, the circuit breaker trips "
            "to the open state, immediately returning a fallback response instead of waiting. "
            "After a configured cool-down period it transitions to half-open, allowing a probe "
            "request through. If successful the circuit closes, restoring normal traffic flow "
            "and improving system resilience under degraded conditions."
        ),
    },
    {
        "id": "doc_009",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a large "
            "language model to ground responses in factual documents. An embedding model converts "
            "both the document corpus and user queries into dense vectors. The top-k most similar "
            "chunks are retrieved from a vector store and appended to the prompt, giving the LLM "
            "relevant context. This approach reduces hallucination and enables the model to "
            "answer questions about private or recent data not seen during pre-training."
        ),
    },
    {
        "id": "doc_010",
        "text": (
            "Observability in production systems relies on three pillars: logs, metrics, and "
            "traces. Structured logs provide human-readable event records; metrics expose "
            "time-series data for dashboards and alerting; distributed traces correlate requests "
            "across service boundaries. During incidents such as a traffic surge or latency "
            "spike, engineers use tools like Cloud Monitoring, Grafana, and Jaeger to diagnose "
            "root causes and measure the impact on end-user experience."
        ),
    },
]
