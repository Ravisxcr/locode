package rag

import (
	"context"
	"testing"
)

func TestTokenizeCode(t *testing.T) {
	code := "func GetUserProfileByID(userID string) (*UserProfile, error)"
	tokens := TokenizeCode(code)

	expectedTokens := []string{"func", "get", "user", "profile", "by", "id", "string", "error"}
	tokenMap := make(map[string]bool)
	for _, tok := range tokens {
		tokenMap[tok] = true
	}

	for _, exp := range expectedTokens {
		if !tokenMap[exp] {
			t.Errorf("expected token %q to be in tokenized output %v", exp, tokens)
		}
	}
}

func TestLocalBM25Embedder(t *testing.T) {
	embedder := NewLocalBM25Embedder(128)

	vec1, err := embedder.EmbedQuery(context.Background(), "function to handle user authentication and tokens")
	if err != nil {
		t.Fatalf("EmbedQuery failed: %v", err)
	}

	if len(vec1) != 128 {
		t.Fatalf("expected dim 128, got %d", len(vec1))
	}

	// Related query
	vec2, _ := embedder.EmbedQuery(context.Background(), "user authentication token handler")
	// Unrelated query
	vec3, _ := embedder.EmbedQuery(context.Background(), "kubernetes cluster deployment terraform")

	simRelated := CosineSimilarity(vec1, vec2)
	simUnrelated := CosineSimilarity(vec1, vec3)

	if simRelated <= simUnrelated {
		t.Errorf("expected related similarity (%f) > unrelated similarity (%f)", simRelated, simUnrelated)
	}
}
