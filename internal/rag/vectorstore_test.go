package rag

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	v1 := []float32{1.0, 0.0, 0.0}
	v2 := []float32{1.0, 0.0, 0.0}
	v3 := []float32{0.0, 1.0, 0.0}

	simIdentical := CosineSimilarity(v1, v2)
	if simIdentical < 0.99 {
		t.Errorf("expected ~1.0 for identical vectors, got %f", simIdentical)
	}

	simOrthogonal := CosineSimilarity(v1, v3)
	if simOrthogonal != 0.0 {
		t.Errorf("expected 0.0 for orthogonal vectors, got %f", simOrthogonal)
	}
}

func TestVectorStore_AddSearchPersist(t *testing.T) {
	tmpDir := t.TempDir()
	indexPath := filepath.Join(tmpDir, "index.json")

	store := NewVectorStore("test-model", 3)

	chunks := []CodeChunk{
		{
			ID:        "auth.go#L1-L10",
			FilePath:  "auth.go",
			Language:  "go",
			Content:   "func AuthenticateUser(token string) bool { return true }",
			StartLine: 1,
			EndLine:   10,
		},
		{
			ID:        "db.go#L1-L10",
			FilePath:  "db.go",
			Language:  "go",
			Content:   "func ConnectDatabase(url string) (*DB, error) { return nil, nil }",
			StartLine: 1,
			EndLine:   10,
		},
	}

	embeddings := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
	}

	if err := store.AddDocuments(chunks, embeddings); err != nil {
		t.Fatalf("AddDocuments failed: %v", err)
	}

	// Search for auth-related vector
	queryVec := []float32{0.9, 0.1, 0.0}
	results := store.Search(queryVec, 2, 0.0)

	if len(results) != 2 {
		t.Fatalf("expected 2 search results, got %d", len(results))
	}
	if results[0].Chunk.FilePath != "auth.go" {
		t.Errorf("top match = %s, want auth.go", results[0].Chunk.FilePath)
	}

	// Persist and reload
	if err := store.Save(indexPath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	if _, err := os.Stat(indexPath); err != nil {
		t.Fatalf("index file was not created: %v", err)
	}

	newStore := NewVectorStore("test-model", 3)
	if err := newStore.Load(indexPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	stats := newStore.Stats()
	if stats.TotalChunks != 2 {
		t.Errorf("loaded chunks = %d, want 2", stats.TotalChunks)
	}
}

