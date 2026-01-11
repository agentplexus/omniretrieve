# OmniRetrieve

[![CI](https://github.com/agentplexus/omniretrieve/actions/workflows/ci.yaml/badge.svg)](https://github.com/agentplexus/omniretrieve/actions/workflows/ci.yaml)
[![Go Reference](https://pkg.go.dev/badge/github.com/agentplexus/omniretrieve.svg)](https://pkg.go.dev/github.com/agentplexus/omniretrieve)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OmniRetrieve is a unified retrieval library for Go that supports Vector RAG, Graph RAG, and Hybrid retrieval strategies. It provides a consistent interface for building retrieval-augmented generation (RAG) systems with pluggable backends.

## Features

- **Vector Retrieval** - Semantic similarity search using embeddings
- **Graph Retrieval** - Relationship-aware traversal for structured knowledge
- **Hybrid Retrieval** - Combine vector and graph strategies with configurable policies
- **Observability** - Built-in tracing compatible with Phoenix, Opik, and Langfuse
- **Reranking** - Cross-encoder and heuristic reranking support
- **Pluggable Backends** - Use pgvector, Pinecone, Neo4j, or implement your own

## Installation

```bash
go get github.com/agentplexus/omniretrieve
```

For pgvector support:

```bash
go get github.com/agentplexus/omniretrieve/providers/pgvector
```

## Quick Start

### Vector Retrieval

```go
package main

import (
    "context"
    "log"

    "github.com/agentplexus/omniretrieve/memory"
    "github.com/agentplexus/omniretrieve/vector"
)

func main() {
    ctx := context.Background()

    // Create in-memory index and embedder for testing
    index := memory.NewVectorIndex("documents", 384)
    embedder := memory.NewHashEmbedder(384)

    // Create retriever
    retriever := vector.NewRetriever(vector.RetrieverConfig{
        Index:    index,
        Embedder: embedder,
        TopK:     10,
    })

    // Insert documents
    docs := []vector.Node{
        {ID: "1", Content: "Go is a statically typed language"},
        {ID: "2", Content: "Python is dynamically typed"},
        {ID: "3", Content: "Rust has strong memory safety"},
    }

    for _, doc := range docs {
        emb, _ := embedder.Embed(ctx, doc.Content)
        doc.Embedding = emb
        index.Insert(ctx, doc)
    }

    // Query
    results, err := retriever.Retrieve(ctx, retrieve.Query{
        Text: "programming languages with type systems",
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, item := range results.Items {
        log.Printf("Score: %.3f, Content: %s", item.Score, item.Content)
    }
}
```

### Hybrid Retrieval

```go
package main

import (
    "context"

    "github.com/agentplexus/omniretrieve/hybrid"
    "github.com/agentplexus/omniretrieve/memory"
    "github.com/agentplexus/omniretrieve/vector"
    "github.com/agentplexus/omniretrieve/graph"
)

func main() {
    ctx := context.Background()

    // Create vector retriever
    vectorIndex := memory.NewVectorIndex("docs", 384)
    embedder := memory.NewHashEmbedder(384)
    vectorRetriever := vector.NewRetriever(vector.RetrieverConfig{
        Index:    vectorIndex,
        Embedder: embedder,
    })

    // Create graph retriever
    kg := memory.NewKnowledgeGraph()
    graphRetriever := graph.NewRetriever(graph.RetrieverConfig{
        Graph: kg,
    })

    // Create hybrid retriever
    hybridRetriever := hybrid.NewRetriever(hybrid.Config{
        VectorRetriever: vectorRetriever,
        GraphRetriever:  graphRetriever,
        Policy:          hybrid.PolicyParallel,
        VectorWeight:    0.7,
        GraphWeight:     0.3,
    })

    // Query both systems
    results, _ := hybridRetriever.Retrieve(ctx, retrieve.Query{
        Text: "What technologies does Company X use?",
    })
}
```

### Using pgvector

```go
package main

import (
    "database/sql"
    "log"

    _ "github.com/lib/pq"
    "github.com/agentplexus/omniretrieve/providers/pgvector"
    "github.com/agentplexus/omniretrieve/vector"
)

func main() {
    // Connect to PostgreSQL
    db, err := sql.Open("postgres", "postgres://user:pass@localhost/mydb?sslmode=disable")
    if err != nil {
        log.Fatal(err)
    }

    // Create pgvector index
    index, err := pgvector.New(db, pgvector.DefaultConfig("embeddings", 1536))
    if err != nil {
        log.Fatal(err)
    }

    // Use with vector retriever
    retriever := vector.NewRetriever(vector.RetrieverConfig{
        Index:    index,
        Embedder: myEmbedder, // Your embedding provider
        TopK:     10,
    })
}
```

## Architecture

```
omniretrieve/
├── retrieve/      # Core interfaces (Retriever, Query, Result)
├── vector/        # Vector retrieval implementation
├── graph/         # Graph retrieval implementation
├── hybrid/        # Hybrid retrieval with policies
├── observe/       # Observability and tracing
├── rerank/        # Reranking implementations
├── memory/        # In-memory implementations for testing
└── providers/
    └── pgvector/  # PostgreSQL pgvector provider
```

## Retrieval Strategies

| Strategy | Best For | Trade-offs |
|----------|----------|------------|
| **Vector** | Semantic similarity, fuzzy matching | May miss explicit relationships |
| **Graph** | Structured knowledge, relationships | Requires schema, less flexible |
| **Hybrid** | Complex queries needing both | Higher latency, more complexity |

### Hybrid Policies

- `PolicyParallel` - Run vector and graph in parallel, merge results
- `PolicyVectorThenGraph` - Vector first, enhance with graph context
- `PolicyGraphThenVector` - Graph first, expand with vector similarity

## Observability

OmniRetrieve includes built-in observability support compatible with:

- [Phoenix](https://phoenix.arize.com/) - LLM observability
- [Opik](https://www.comet.com/site/products/opik/) - Experiment tracking
- [Langfuse](https://langfuse.com/) - LLM analytics

```go
import "github.com/agentplexus/omniretrieve/observe"

// Create observer with exporters
obs := observe.NewObserver(
    observe.WithExporter(myPhoenixExporter),
)

// Create traced context
ctx := observe.NewContext(context.Background())

// Retrieval operations are automatically traced
results, _ := retriever.Retrieve(ctx, query)
```

## Providers

| Provider | Type | Status |
|----------|------|--------|
| pgvector | Vector | ✅ Available |
| Pinecone | Vector | Planned |
| Weaviate | Vector | Planned |
| Neo4j | Graph | Planned |
| Neptune | Graph | Planned |

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.
