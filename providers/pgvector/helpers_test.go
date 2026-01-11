package pgvector

import (
	"testing"
)

func TestVectorToString(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected string
	}{
		{
			name:     "empty vector",
			input:    []float32{},
			expected: "[]",
		},
		{
			name:     "single element",
			input:    []float32{1.5},
			expected: "[1.500000]",
		},
		{
			name:     "multiple elements",
			input:    []float32{1.0, 2.5, 3.14159},
			expected: "[1.000000,2.500000,3.141590]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := vectorToString(tt.input)
			if result != tt.expected {
				t.Errorf("vectorToString(%v) = %s, want %s", tt.input, result, tt.expected)
			}
		})
	}
}

func TestParseVector(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []float32
	}{
		{
			name:     "empty vector",
			input:    "[]",
			expected: nil,
		},
		{
			name:     "single element",
			input:    "[1.5]",
			expected: []float32{1.5},
		},
		{
			name:     "multiple elements",
			input:    "[1.0, 2.5, 3.14]",
			expected: []float32{1.0, 2.5, 3.14},
		},
		{
			name:     "no brackets",
			input:    "1.0,2.0,3.0",
			expected: []float32{1.0, 2.0, 3.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseVector(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("parseVector(%s) length = %d, want %d", tt.input, len(result), len(tt.expected))
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("parseVector(%s)[%d] = %f, want %f", tt.input, i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestDistanceOpClass(t *testing.T) {
	tests := []struct {
		metric   DistanceMetric
		expected string
	}{
		{DistanceCosine, "vector_cosine_ops"},
		{DistanceEuclidean, "vector_l2_ops"},
		{DistanceInnerProduct, "vector_ip_ops"},
		{"unknown", "vector_cosine_ops"}, // Default
	}

	for _, tt := range tests {
		t.Run(string(tt.metric), func(t *testing.T) {
			idx := &Index{config: Config{DistanceMetric: tt.metric}}
			result := idx.distanceOpClass()
			if result != tt.expected {
				t.Errorf("distanceOpClass() = %s, want %s", result, tt.expected)
			}
		})
	}
}

func TestDistanceOperator(t *testing.T) {
	tests := []struct {
		metric   DistanceMetric
		expected string
	}{
		{DistanceCosine, "<=>"},
		{DistanceEuclidean, "<->"},
		{DistanceInnerProduct, "<#>"},
		{"unknown", "<=>"}, // Default
	}

	for _, tt := range tests {
		t.Run(string(tt.metric), func(t *testing.T) {
			idx := &Index{config: Config{DistanceMetric: tt.metric}}
			result := idx.distanceOperator()
			if result != tt.expected {
				t.Errorf("distanceOperator() = %s, want %s", result, tt.expected)
			}
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig("my_table", 1536)

	if cfg.TableName != "my_table" {
		t.Errorf("TableName = %s, want my_table", cfg.TableName)
	}
	if cfg.Dimensions != 1536 {
		t.Errorf("Dimensions = %d, want 1536", cfg.Dimensions)
	}
	if cfg.DistanceMetric != DistanceCosine {
		t.Errorf("DistanceMetric = %s, want cosine", cfg.DistanceMetric)
	}
	if cfg.IndexType != IndexTypeHNSW {
		t.Errorf("IndexType = %s, want hnsw", cfg.IndexType)
	}
	if !cfg.CreateTableIfNotExists {
		t.Error("CreateTableIfNotExists should be true")
	}
	if cfg.HNSWConfig == nil {
		t.Error("HNSWConfig should not be nil")
	} else {
		if cfg.HNSWConfig.M != 16 {
			t.Errorf("HNSWConfig.M = %d, want 16", cfg.HNSWConfig.M)
		}
		if cfg.HNSWConfig.EfConstruction != 64 {
			t.Errorf("HNSWConfig.EfConstruction = %d, want 64", cfg.HNSWConfig.EfConstruction)
		}
	}
}
