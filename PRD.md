# OmniRetrieve - Product Requirements Document

## Overview

OmniRetrieve is a unified retrieval library that abstracts the complexity of different RAG (Retrieval-Augmented Generation) strategies behind a consistent Go interface. It enables developers to build AI applications that leverage vector search, knowledge graphs, or hybrid approaches without being locked into specific backends.

## Problem Statement

Building RAG systems today requires:

1. **Choosing between paradigms** - Vector RAG and Graph RAG have different strengths, but most libraries force an either/or choice
2. **Backend lock-in** - Switching from pgvector to Pinecone, or Neo4j to Neptune, requires significant code changes
3. **Missing observability** - Debugging retrieval quality is difficult without proper tracing
4. **Inconsistent interfaces** - Each vector/graph database has its own API patterns

## Goals

### Primary Goals

1. **Unified Interface** - Single `Retriever` interface works with any backend
2. **Strategy Flexibility** - Support vector, graph, and hybrid retrieval
3. **Backend Agnostic** - Swap pgvector for Pinecone without changing application code
4. **Observable** - Built-in tracing for debugging and optimization

### Non-Goals

1. **Embedding Generation** - OmniRetrieve accepts embeddings but doesn't generate them (use external providers)
2. **LLM Integration** - Focus is on retrieval, not generation
3. **Data Ingestion Pipelines** - Chunking, parsing are out of scope
4. **Infrastructure Deployment** - No Terraform/Pulumi/CDK included

## User Personas

### Application Developer

- Building RAG-powered applications
- Wants simple APIs that just work
- Needs to iterate quickly on retrieval strategies
- May not be a vector search expert

### ML/AI Engineer

- Optimizing retrieval quality
- Needs observability into what's being retrieved
- Wants to A/B test different strategies
- Cares about latency and throughput

### Platform Engineer

- Deploying and scaling RAG infrastructure
- Needs to swap backends based on requirements
- Wants consistent patterns across services
- Cares about operational complexity

## Requirements

### Functional Requirements

#### FR1: Vector Retrieval

- FR1.1: Support semantic similarity search using embeddings
- FR1.2: Support metadata filtering on queries
- FR1.3: Support configurable top-k results
- FR1.4: Support minimum score thresholds

#### FR2: Graph Retrieval

- FR2.1: Support entity-relationship traversal
- FR2.2: Support configurable traversal depth
- FR2.3: Support relationship type filtering
- FR2.4: Return paths with context

#### FR3: Hybrid Retrieval

- FR3.1: Support parallel execution of vector and graph
- FR3.2: Support sequential strategies (vector-then-graph, graph-then-vector)
- FR3.3: Support configurable weighting for result merging
- FR3.4: Support deduplication of merged results

#### FR4: Index Management

- FR4.1: Support CRUD operations for vectors/nodes
- FR4.2: Support batch operations for efficiency
- FR4.3: Support upsert semantics
- FR4.4: Support index creation and deletion

#### FR5: Observability

- FR5.1: Generate spans for all retrieval operations
- FR5.2: Include query, results, and latency in spans
- FR5.3: Support pluggable exporters (Phoenix, Opik, Langfuse)
- FR5.4: Propagate trace context through operations

#### FR6: Reranking

- FR6.1: Support cross-encoder reranking
- FR6.2: Support heuristic reranking (RRF, min-score)
- FR6.3: Support chained rerankers

### Non-Functional Requirements

#### NFR1: Performance

- NFR1.1: Add < 5ms overhead to backend query latency
- NFR1.2: Support concurrent operations
- NFR1.3: Minimize memory allocations in hot paths

#### NFR2: Reliability

- NFR2.1: Graceful degradation on backend failures
- NFR2.2: Context cancellation support
- NFR2.3: Timeout handling

#### NFR3: Extensibility

- NFR3.1: Easy to add new vector backends
- NFR3.2: Easy to add new graph backends
- NFR3.3: Custom reranking strategies

#### NFR4: Developer Experience

- NFR4.1: Intuitive API design
- NFR4.2: Comprehensive documentation
- NFR4.3: In-memory implementations for testing

## Success Metrics

| Metric | Target |
|--------|--------|
| API Surface Complexity | < 10 public types per package |
| Test Coverage | > 80% |
| Documentation Coverage | 100% of public APIs |
| Backend Integration Time | < 1 day for new provider |

## Milestones

### v0.1.0 - Foundation

- Core interfaces (Retriever, Query, Result)
- Vector retrieval implementation
- Graph retrieval implementation
- Hybrid retrieval with policies
- In-memory backends for testing
- pgvector provider
- Basic observability

### v0.2.0 - Production Ready

- Additional vector providers (Pinecone, Weaviate)
- Graph provider (Neo4j)
- Enhanced observability exporters
- Performance optimizations
- Comprehensive benchmarks

### v0.3.0 - Advanced Features

- Streaming results
- Caching layer
- Query planning optimization
- Additional graph providers (Neptune)

## Open Questions

1. Should we support async/streaming retrieval in v0.1.0?
2. What's the right abstraction for graph schema differences?
3. How do we handle provider-specific features (e.g., pgvector's HNSW vs IVFFlat)?

## Appendix

### Competitive Analysis

| Library | Language | Vector | Graph | Hybrid |
|---------|----------|--------|-------|--------|
| LlamaIndex | Python | ✅ | ✅ | ✅ |
| LangChain | Python | ✅ | ✅ | Partial |
| Haystack | Python | ✅ | ❌ | ❌ |
| **OmniRetrieve** | **Go** | ✅ | ✅ | ✅ |

### References

- [Vector RAG vs Graph RAG](https://medium.com/@nebulagraph/graph-rag-the-new-llm-stack-e9db8d61f02b)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [OpenTelemetry Tracing](https://opentelemetry.io/docs/concepts/signals/traces/)
