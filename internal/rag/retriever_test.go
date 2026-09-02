package rag

import (
	"context"
	"testing"
)

func TestRetriever_HybridSearch(t *testing.T) {
	embedder := NewLocalBM25Embedder(128)
	store := NewVectorStore(embedder.ModelName(), embedder.Dimension())

	chunks := []CodeChunk{
		{
			ID:        "auth.go#L1-L10",
			FilePath:  "auth.go",
			Language:  "go",
			Content:   "func CheckUserToken(token string) bool {\n\t// validate JWT auth token\n\treturn isValidToken(token)\n}",
			StartLine: 1,
			EndLine:   10,
		},
		{
			ID:        "db.go#L1-L10",
			FilePath:  "db.go",
			Language:  "go",
			Content:   "func ConnectDatabase(dsn string) (*sql.DB, error) {\n\treturn sql.Open(\"postgres\", dsn)\n}",
			StartLine: 1,
			EndLine:   10,
		},
		{
			ID:        "router.go#L1-L10",
			FilePath:  "router.go",
			Language:  "go",
			Content:   "func RegisterRoutes(r *chi.Mux) {\n\tr.Get(\"/health\", HealthCheck)\n}",
			StartLine: 1,
			EndLine:   10,
		},
	}

	embs, _ := embedder.EmbedDocuments(context.Background(), []string{
		chunks[0].Content,
		chunks[1].Content,
		chunks[2].Content,
	})

	_ = store.AddDocuments(chunks, embs)

	retriever := NewRetriever(store, embedder)

	results, err := retriever.Retrieve(context.Background(), "validate jwt token", 2, "")
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatalf("expected results, got 0")
	}

	if results[0].Chunk.FilePath != "auth.go" {
		t.Errorf("expected top result to be auth.go, got %s", results[0].Chunk.FilePath)
	}

	formatted := FormatContext(results)
	if formatted == "" {
		t.Errorf("FormatContext returned empty string")
	}
}

