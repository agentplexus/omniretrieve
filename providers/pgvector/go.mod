module github.com/agentplexus/omniretrieve/providers/pgvector

go 1.25.3

require (
	github.com/agentplexus/omniretrieve v0.1.0
	github.com/lib/pq v1.10.9
)

// For local development within the monorepo.
// This directive is ignored when the module is used as a dependency.
replace github.com/agentplexus/omniretrieve => ../..
