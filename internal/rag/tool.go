package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/Ravisxcr/gocode-rag/internal/toolimpl"
)

// RagSearchTool provides semantic codebase search via vector RAG for the agent.
type RagSearchTool struct {
	retriever *Retriever
}

// NewRagSearchTool creates a new RagSearchTool.
func NewRagSearchTool(retriever *Retriever) *RagSearchTool {
	return &RagSearchTool{retriever: retriever}
}

// Execute performs semantic codebase search given a query string.
func (t *RagSearchTool) Execute(params map[string]interface{}) toolimpl.ToolResult {
	query, _ := params["query"].(string)
	if query == "" {
		return toolimpl.ToolResult{
			Success: false,
			Error:   "missing required parameter: query",
		}
	}

	limit := 5
	if v, ok := params["limit"]; ok {
		switch lv := v.(type) {
		case float64:
			limit = int(lv)
		case json.Number:
			if n, err := lv.Int64(); err == nil {
				limit = int(n)
			}
		}
	}

	pathFilter, _ := params["path_filter"].(string)
	if pathFilter == "" {
		pathFilter, _ = params["path"].(string)
	}

	// Auto-initialize retriever if nil
	if t.retriever == nil {
		cwd, _ := os.Getwd()
		indexPath := filepath.Join(cwd, ".gocode", "rag", "index.json")
		embedder := ResolveEmbedder(EmbedderConfig{Provider: "auto"})
		store := NewVectorStore(embedder.ModelName(), embedder.Dimension())
		_ = store.Load(indexPath)
		t.retriever = NewRetriever(store, embedder)
	}

	results, err := t.retriever.Retrieve(context.Background(), query, limit, pathFilter)
	if err != nil {
		return toolimpl.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("rag search error: %v", err),
		}
	}

	output := FormatContext(results)
	return toolimpl.ToolResult{
		Success: true,
		Output:  output,
	}
}

// RagCodeContextTool provides specialized codebase context for writing and editing code.
type RagCodeContextTool struct {
	retriever *Retriever
}

// NewRagCodeContextTool creates a new RagCodeContextTool.
func NewRagCodeContextTool(retriever *Retriever) *RagCodeContextTool {
	return &RagCodeContextTool{retriever: retriever}
}

// Execute retrieves comprehensive context (types, implementations, and test patterns) for a coding task.
func (t *RagCodeContextTool) Execute(params map[string]interface{}) toolimpl.ToolResult {
	symbolOrTask, _ := params["symbol_or_task"].(string)
	if symbolOrTask == "" {
		symbolOrTask, _ = params["query"].(string)
	}
	if symbolOrTask == "" {
		return toolimpl.ToolResult{
			Success: false,
			Error:   "missing required parameter: symbol_or_task (or query)",
		}
	}

	targetFile, _ := params["target_file"].(string)
	includeTests := true
	if v, ok := params["include_tests"].(bool); ok {
		includeTests = v
	}

	// Auto-initialize retriever if nil
	if t.retriever == nil {
		cwd, _ := os.Getwd()
		indexPath := filepath.Join(cwd, ".gocode", "rag", "index.json")
		embedder := ResolveEmbedder(EmbedderConfig{Provider: "auto"})
		store := NewVectorStore(embedder.ModelName(), embedder.Dimension())
		_ = store.Load(indexPath)
		t.retriever = NewRetriever(store, embedder)
	}

	// 1. Retrieve main code patterns & type definitions
	implResults, err := t.retriever.Retrieve(context.Background(), symbolOrTask, 4, "")
	if err != nil {
		return toolimpl.ToolResult{
			Success: false,
			Error:   fmt.Sprintf("retrieving code context: %v", err),
		}
	}

	var testResults []SearchResult
	if includeTests {
		// 2. Retrieve related test cases
		tResults, err := t.retriever.Retrieve(context.Background(), symbolOrTask+" test", 3, "*test*")
		if err == nil {
			testResults = tResults
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# Code Context & Architectural Patterns for: %s\n\n", symbolOrTask))
	if targetFile != "" {
		sb.WriteString(fmt.Sprintf("**Target File**: `%s`\n\n", targetFile))
	}

	sb.WriteString("## Reference Implementations & Types:\n\n")
	if len(implResults) > 0 {
		sb.WriteString(FormatContext(implResults))
	} else {
		sb.WriteString("No exact implementation matches found.\n")
	}

	if len(testResults) > 0 {
		sb.WriteString("\n\n## Related Test Patterns & Edge Cases:\n\n")
		sb.WriteString(FormatContext(testResults))
	}

	sb.WriteString("\n\n> **Guideline for Code Generation**:\n")
	sb.WriteString("> 1. Reuse existing types, error handlers, and helper utilities shown above.\n")
	sb.WriteString("> 2. Follow the naming conventions, struct tags, and idioms established in these reference files.\n")
	sb.WriteString("> 3. Ensure your implementation satisfies any interface contracts and handles the edge cases tested above.\n")

	return toolimpl.ToolResult{
		Success: true,
		Output:  sb.String(),
	}
}

// RegisterRagTools registers RAG search and code context tools into the tool registry.
func RegisterRagTools(reg *toolimpl.Registry, retriever *Retriever) {
	searchTool := NewRagSearchTool(retriever)
	reg.Set("rag_search", searchTool)
	reg.Set("ragsearchtool", searchTool)
	reg.Set("semantic_search", searchTool)

	contextTool := NewRagCodeContextTool(retriever)
	reg.Set("rag_code_context", contextTool)
	reg.Set("ragcodecontexttool", contextTool)
}

