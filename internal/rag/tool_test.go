package rag

import (
	"context"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/toolimpl"
)

func TestRagSearchTool(t *testing.T) {
	embedder := NewLocalBM25Embedder(64)
	store := NewVectorStore(embedder.ModelName(), 64)

	chunks := []CodeChunk{
		{
			ID:        "main.go#L1-L10",
			FilePath:  "main.go",
			StartLine: 1,
			EndLine:   10,
			Language:  "go",
			Content:   "package main\n\nfunc HandleAuthentication() bool { return true }",
		},
	}
	embs, _ := embedder.EmbedDocuments(context.Background(), []string{chunks[0].Content})
	_ = store.AddDocuments(chunks, embs)

	retriever := NewRetriever(store, embedder)
	tool := NewRagSearchTool(retriever)

	res := tool.Execute(map[string]interface{}{
		"query": "HandleAuthentication",
		"limit": 5,
	})

	if !res.Success {
		t.Fatalf("expected success, got error: %v", res.Error)
	}
	if res.Output == "" {
		t.Fatalf("expected non-empty output")
	}
}

func TestRagCodeContextTool(t *testing.T) {
	embedder := NewLocalBM25Embedder(64)
	store := NewVectorStore(embedder.ModelName(), 64)

	chunks := []CodeChunk{
		{
			ID:        "internal/auth/token.go#L1-L15",
			FilePath:  "internal/auth/token.go",
			StartLine: 1,
			EndLine:   15,
			Language:  "go",
			Content:   "type TokenValidator interface { ValidateToken(token string) bool }",
		},
		{
			ID:        "internal/auth/token_test.go#L1-L15",
			FilePath:  "internal/auth/token_test.go",
			StartLine: 1,
			EndLine:   15,
			Language:  "go",
			Content:   "func TestValidateToken(t *testing.T) { // test token validation }",
		},
	}
	embs, _ := embedder.EmbedDocuments(context.Background(), []string{chunks[0].Content, chunks[1].Content})
	_ = store.AddDocuments(chunks, embs)

	retriever := NewRetriever(store, embedder)
	contextTool := NewRagCodeContextTool(retriever)

	res := contextTool.Execute(map[string]interface{}{
		"symbol_or_task": "TokenValidator token validation",
		"target_file":    "internal/auth/custom.go",
		"include_tests":  true,
	})

	if !res.Success {
		t.Fatalf("expected success, got error: %v", res.Error)
	}
	if res.Output == "" {
		t.Fatalf("expected non-empty output")
	}

	reg := toolimpl.NewRegistry()
	RegisterRagTools(reg, retriever)

	if reg.Get("rag_search") == nil {
		t.Errorf("expected registry to have rag_search")
	}
	if reg.Get("rag_code_context") == nil {
		t.Errorf("expected registry to have rag_code_context")
	}
}
