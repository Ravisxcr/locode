package rag

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestIndexer_Workspace(t *testing.T) {
	tmpDir := t.TempDir()

	// Create dummy workspace files
	f1 := filepath.Join(tmpDir, "main.go")
	_ = os.WriteFile(f1, []byte("package main\n\nfunc main() {\n\tprintln(\"hello world\")\n}\n"), 0o644)

	f2 := filepath.Join(tmpDir, "helper.go")
	_ = os.WriteFile(f2, []byte("package main\n\nfunc Helper() string {\n\treturn \"help\"\n}\n"), 0o644)

	embedder := NewLocalBM25Embedder(64)
	chunker := NewCodeChunker(DefaultChunkOptions())
	store := NewVectorStore(embedder.ModelName(), embedder.Dimension())

	indexer := NewIndexer(chunker, embedder, store)

	opts := IndexOptions{
		RootPath:    tmpDir,
		IndexPath:   filepath.Join(tmpDir, ".gocode", "rag", "index.json"),
		ChunkOpts:   DefaultChunkOptions(),
		MaxFileSize: 512 * 1024,
	}

	report, err := indexer.IndexWorkspace(context.Background(), opts)
	if err != nil {
		t.Fatalf("IndexWorkspace failed: %v", err)
	}

	if report.FilesIndexed != 2 {
		t.Errorf("FilesIndexed = %d, want 2", report.FilesIndexed)
	}
	if report.TotalChunks != 2 {
		t.Errorf("TotalChunks = %d, want 2", report.TotalChunks)
	}

	// Verify incremental index skips unchanged files
	report2, err := indexer.IndexWorkspace(context.Background(), opts)
	if err != nil {
		t.Fatalf("Second IndexWorkspace failed: %v", err)
	}
	if report2.FilesIndexed != 0 {
		t.Errorf("expected 0 files indexed on second run, got %d", report2.FilesIndexed)
	}
	if report2.FilesSkipped != 2 {
		t.Errorf("expected 2 files skipped on second run, got %d", report2.FilesSkipped)
	}
}

