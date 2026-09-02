package apiclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

func TestCleanModelName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"ollama/llama3.3", "llama3.3:latest"},
		{"ollama:qwen2.5-coder", "qwen2.5-coder:latest"},
		{"ollama-llama", "llama3.3:latest"},
		{"llama3.2", "llama3.2:latest"},
		{"sonnet", "claude-sonnet-4-6"},
	}

	for _, tt := range tests {
		got := CleanModelName(tt.input)
		if got != tt.expected {
			t.Errorf("CleanModelName(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestDefaultOllamaHost(t *testing.T) {
	os.Unsetenv("OLLAMA_HOST")
	os.Unsetenv("OLLAMA_BASE_URL")

	if got := DefaultOllamaHost(); got != "http://localhost:11434" {
		t.Errorf("DefaultOllamaHost() = %q, want http://localhost:11434", got)
	}

	os.Setenv("OLLAMA_HOST", "192.168.1.100:11434")
	if got := DefaultOllamaHost(); got != "http://192.168.1.100:11434" {
		t.Errorf("DefaultOllamaHost() with OLLAMA_HOST = %q, want http://192.168.1.100:11434", got)
	}

	os.Unsetenv("OLLAMA_HOST")
	os.Setenv("OLLAMA_BASE_URL", "http://my-ollama:11434/v1")
	if got := DefaultOllamaHost(); got != "http://my-ollama:11434" {
		t.Errorf("DefaultOllamaHost() with OLLAMA_BASE_URL = %q, want http://my-ollama:11434", got)
	}
	os.Unsetenv("OLLAMA_BASE_URL")
}

func TestOllamaProvider_ListLocalModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := OllamaTagsResponse{
				Models: []OllamaModelInfo{
					{Name: "llama3.3:latest", Model: "llama3.3:latest", Size: 4300000000},
					{Name: "nomic-embed-text:latest", Model: "nomic-embed-text:latest", Size: 274000000},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	prov := NewOllamaProvider(server.URL, apitypes.AuthSource{})
	models, err := prov.ListLocalModels(context.Background())
	if err != nil {
		t.Fatalf("ListLocalModels failed: %v", err)
	}

	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].Name != "llama3.3:latest" {
		t.Errorf("model[0] name = %q, want llama3.3:latest", models[0].Name)
	}
}

func TestOllamaProvider_Embeddings(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/embeddings" {
			var req OllamaEmbeddingRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			resp := OllamaEmbeddingResponse{
				Embedding: []float32{0.1, 0.2, 0.3, 0.4},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	prov := NewOllamaProvider(server.URL, apitypes.AuthSource{})
	emb, err := prov.GenerateEmbedding(context.Background(), "nomic-embed-text", "hello world")
	if err != nil {
		t.Fatalf("GenerateEmbedding failed: %v", err)
	}
	if len(emb) != 4 || emb[0] != 0.1 {
		t.Errorf("unexpected embedding: %v", emb)
	}
}

func TestOllamaProvider_SanitizeResponseToolCalls(t *testing.T) {
	prov := NewOllamaProvider("http://localhost:11434", apitypes.AuthSource{})

	rawText := "I will run the command now.\n```json\n{\"name\": \"BashTool\", \"parameters\": {\"command\": \"ls -la\"}}\n```\nDone."
	resp := &apitypes.MessageResponse{
		Content: []apitypes.OutputContentBlock{
			{Kind: "text", Text: rawText},
		},
	}

	prov.sanitizeResponseToolCalls(resp)

	hasToolUse := false
	for _, b := range resp.Content {
		if b.Kind == "tool_use" {
			hasToolUse = true
			if b.Name != "BashTool" {
				t.Errorf("tool_use name = %q, want BashTool", b.Name)
			}
			var inputMap map[string]interface{}
			if err := json.Unmarshal(b.Input, &inputMap); err != nil {
				t.Fatalf("failed to unmarshal b.Input: %v", err)
			}
			if cmd, _ := inputMap["command"].(string); cmd != "ls -la" {
				t.Errorf("tool_use input['command'] = %q, want 'ls -la'", cmd)
			}
		}
	}
	if !hasToolUse {
		t.Errorf("expected tool_use block to be extracted, got: %+v", resp.Content)
	}
}
