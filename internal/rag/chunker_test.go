package rag

import (
	"strings"
	"testing"
)

func TestDetectLanguage(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"main.go", "go"},
		{"internal/rag/chunker.go", "go"},
		{"app/page.tsx", "typescript"},
		{"scripts/test.py", "python"},
		{"src/lib.rs", "rust"},
		{"README.md", "markdown"},
		{"data/tools.json", "json"},
		{"unknown.xyz", "text"},
	}

	for _, tt := range tests {
		got := DetectLanguage(tt.path)
		if got != tt.expected {
			t.Errorf("DetectLanguage(%q) = %q, want %q", tt.path, got, tt.expected)
		}
	}
}

func TestCodeChunker_SmallFile(t *testing.T) {
	chunker := NewCodeChunker(DefaultChunkOptions())
	content := "package main\n\nfunc main() {\n\tprintln(\"hello\")\n}\n"
	chunks := chunker.ChunkFile("main.go", content)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk for small file, got %d", len(chunks))
	}

	c := chunks[0]
	if c.FilePath != "main.go" {
		t.Errorf("FilePath = %q, want main.go", c.FilePath)
	}
	if c.Language != "go" {
		t.Errorf("Language = %q, want go", c.Language)
	}
	if c.StartLine != 1 {
		t.Errorf("StartLine = %d, want 1", c.StartLine)
	}
	if c.EndLine != 5 {
		t.Errorf("EndLine = %d, want 5", c.EndLine)
	}
}

func TestCodeChunker_LargeFileSplitting(t *testing.T) {
	opts := ChunkOptions{
		MaxChunkTokens: 50,
		OverlapLines:   2,
		MinChunkTokens: 10,
	}
	chunker := NewCodeChunker(opts)

	// Build a larger file with multiple functions
	var sb strings.Builder
	sb.WriteString("package test\n\n")
	for i := 0; i < 20; i++ {
		sb.WriteString(strings.Repeat("func Function"+string(rune('A'+i))+"() {\n\t// comment line inside function\n\tval := 123 + 456\n\t_ = val\n}\n\n", 2))
	}

	content := sb.String()
	chunks := chunker.ChunkFile("test.go", content)

	if len(chunks) <= 1 {
		t.Fatalf("expected multiple chunks for large file, got %d", len(chunks))
	}

	// Verify line number continuity and chunk boundaries
	for i, c := range chunks {
		if c.StartLine <= 0 || c.EndLine < c.StartLine {
			t.Errorf("chunk #%d invalid line range: %d-%d", i, c.StartLine, c.EndLine)
		}
		if c.Content == "" {
			t.Errorf("chunk #%d has empty content", i)
		}
		if c.Checksum == "" {
			t.Errorf("chunk #%d has empty checksum", i)
		}
	}
}

