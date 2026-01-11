// Package retrieve defines the core interfaces and types for OmniRetrieve.
// OmniRetrieve is a unified retrieval layer supporting VectorRAG, GraphRAG,
// and Hybrid approaches with built-in observability.
package retrieve

import (
	"context"
)

// Mode represents the retrieval strategy to use.
type Mode string

const (
	// ModeVector uses vector similarity search.
	ModeVector Mode = "vector"
	// ModeGraph uses knowledge graph traversal.
	ModeGraph Mode = "graph"
	// ModeHybrid combines vector and graph retrieval.
	ModeHybrid Mode = "hybrid"
)

// EntityHint provides hints for entity-based retrieval in graph traversal.
type EntityHint struct {
	// ID is the unique identifier for the entity.
	ID string
	// Type is the entity type (e.g., "person", "concept", "document").
	Type string
	// Name is the human-readable name of the entity.
	Name string
	// Confidence is the confidence score for this hint (0.0-1.0).
	Confidence float64
}

// Query represents a retrieval request with intent, not implementation details.
type Query struct {
	// Text is the raw query text.
	Text string
	// Embedding is an optional pre-computed embedding vector.
	// If nil, the retriever should compute it.
	Embedding []float32
	// Entities are optional entity hints for graph-based retrieval.
	Entities []EntityHint
	// Filters are key-value filters to apply to results.
	Filters map[string]string
	// MaxDepth is the maximum traversal depth for graph retrieval.
	MaxDepth int
	// TopK is the maximum number of results to return.
	TopK int
	// Modes specifies which retrieval strategies to use.
	// If empty, the retriever chooses the default.
	Modes []Mode
	// MinScore is the minimum relevance score threshold (0.0-1.0).
	MinScore float64
	// Metadata contains additional query metadata.
	Metadata map[string]any
}

// ContextItem represents a single piece of retrieved context.
type ContextItem struct {
	// ID is the unique identifier for this item.
	ID string
	// Content is the text content of this item.
	Content string
	// Source identifies where this item came from.
	Source string
	// Score is the relevance score (0.0-1.0).
	Score float64
	// Metadata contains additional item metadata.
	Metadata map[string]string
	// Provenance tracks how this item was retrieved.
	Provenance Provenance
}

// Provenance tracks the retrieval path for a context item.
type Provenance struct {
	// Mode indicates which retrieval strategy found this item.
	Mode Mode
	// Backend identifies the specific backend used.
	Backend string
	// GraphPath contains the traversal path for graph-retrieved items.
	GraphPath []string
	// SimilarityScore is the raw vector similarity score.
	SimilarityScore float64
	// RerankerScore is the score after reranking (if applied).
	RerankerScore float64
}

// Result contains the complete retrieval response.
type Result struct {
	// Items are the retrieved context items, ordered by relevance.
	Items []ContextItem
	// Query is the original query.
	Query Query
	// Metadata contains response metadata.
	Metadata ResultMetadata
}

// ResultMetadata contains metadata about the retrieval operation.
type ResultMetadata struct {
	// TotalCandidates is the number of candidates before filtering/reranking.
	TotalCandidates int
	// LatencyMS is the total retrieval latency in milliseconds.
	LatencyMS int64
	// ModesUsed lists which retrieval modes were actually executed.
	ModesUsed []Mode
	// CacheHit indicates if results came from cache.
	CacheHit bool
}

// Retriever is the core interface for all retrieval operations.
// Implementations may use vector search, graph traversal, or hybrid approaches.
type Retriever interface {
	// Retrieve executes a retrieval query and returns matching context items.
	Retrieve(ctx context.Context, q Query) (*Result, error)
}

// RetrieverFunc is a function adapter for Retriever.
type RetrieverFunc func(ctx context.Context, q Query) (*Result, error)

// Retrieve implements Retriever for RetrieverFunc.
func (f RetrieverFunc) Retrieve(ctx context.Context, q Query) (*Result, error) {
	return f(ctx, q)
}

// Option configures a retrieval operation.
type Option func(*Options)

// Options holds retrieval configuration.
type Options struct {
	// Reranker to apply to results.
	Reranker Reranker
	// Cache to use for results.
	Cache Cache
	// Observer for tracing and metrics.
	Observer Observer
}

// Reranker reorders and rescores retrieval results.
type Reranker interface {
	// Rerank reorders the given items based on relevance to the query.
	Rerank(ctx context.Context, q Query, items []ContextItem) ([]ContextItem, error)
}

// Cache provides caching for retrieval results.
type Cache interface {
	// Get retrieves a cached result for the given query.
	Get(ctx context.Context, q Query) (*Result, bool)
	// Set stores a result in the cache.
	Set(ctx context.Context, q Query, r *Result) error
}

// Observer receives retrieval events for observability.
type Observer interface {
	// OnRetrieveStart is called when a retrieval operation begins.
	OnRetrieveStart(ctx context.Context, q Query) context.Context
	// OnRetrieveEnd is called when a retrieval operation completes.
	OnRetrieveEnd(ctx context.Context, r *Result, err error)
	// OnVectorSearch is called during vector search.
	OnVectorSearch(ctx context.Context, backend string, topK int, resultCount int, latencyMS int64)
	// OnGraphTraverse is called during graph traversal.
	OnGraphTraverse(ctx context.Context, backend string, depth int, nodeCount int, latencyMS int64)
	// OnRerank is called during reranking.
	OnRerank(ctx context.Context, model string, inputCount int, outputCount int, latencyMS int64)
}
