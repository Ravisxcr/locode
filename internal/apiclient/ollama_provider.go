package apiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

// OllamaModelInfo represents a model returned by Ollama's /api/tags endpoint.
type OllamaModelInfo struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	ModifiedAt time.Time `json:"modified_at"`
}

// OllamaTagsResponse represents the response from GET /api/tags.
type OllamaTagsResponse struct {
	Models []OllamaModelInfo `json:"models"`
}

// OllamaEmbeddingRequest is the payload for POST /api/embeddings.
type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaEmbeddingResponse is the response from POST /api/embeddings.
type OllamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// OllamaEmbedRequest is the payload for POST /api/embed (newer Ollama versions).
type OllamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// OllamaEmbedResponse is the response from POST /api/embed.
type OllamaEmbedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float32 `json:"embeddings"`
}

// OllamaProvider communicates with local or remote Ollama servers.
type OllamaProvider struct {
	Host       string
	BaseURL    string // e.g. http://localhost:11434/v1
	NativeURL  string // e.g. http://localhost:11434
	Client     *http.Client
	compatProv *OpenAiCompatProvider
}

// DefaultOllamaHost returns the resolved Ollama host string.
func DefaultOllamaHost() string {
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
			return "http://" + host
		}
		return host
	}
	if baseURL := os.Getenv("OLLAMA_BASE_URL"); baseURL != "" {
		baseURL = strings.TrimSuffix(baseURL, "/v1")
		baseURL = strings.TrimSuffix(baseURL, "/")
		if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
			return "http://" + baseURL
		}
		return baseURL
	}
	return "http://localhost:11434"
}

// NewOllamaProvider creates a new Ollama provider.
func NewOllamaProvider(host string, auth apitypes.AuthSource) *OllamaProvider {
	if host == "" {
		host = DefaultOllamaHost()
	}
	host = strings.TrimRight(host, "/")
	if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
		host = "http://" + host
	}

	baseURL := host + "/v1"
	compat := NewOpenAiCompatProvider(OpenAiCompatConfig{
		ProviderName: "Ollama",
		BaseURLEnv:   "OLLAMA_BASE_URL",
		DefaultBase:  baseURL,
	}, auth)
	compat.BaseURL = baseURL

	return &OllamaProvider{
		Host:       host,
		BaseURL:    baseURL,
		NativeURL:  host,
		Client:     &http.Client{Timeout: 5 * time.Minute},
		compatProv: compat,
	}
}

func (p *OllamaProvider) Kind() ProviderKind {
	return ProviderOllama
}

// CleanModelName strips provider prefixes such as "ollama/" or "ollama:".
func CleanModelName(model string) string {
	m := strings.TrimSpace(model)
	if strings.HasPrefix(strings.ToLower(m), "ollama/") {
		m = m[7:]
	} else if strings.HasPrefix(strings.ToLower(m), "ollama:") {
		m = m[7:]
	}
	return ResolveModelAlias(m)
}

// SendMessage sends a request to Ollama and normalizes the response.
func (p *OllamaProvider) SendMessage(ctx context.Context, req apitypes.MessageRequest) (*apitypes.MessageResponse, error) {
	req.Model = CleanModelName(req.Model)
	resp, err := p.compatProv.SendMessage(ctx, req)
	if err != nil {
		return nil, err
	}

	// Post-process response to sanitize any raw tool calls output as markdown text by local models
	p.sanitizeResponseToolCalls(resp)
	return resp, nil
}

// StreamMessage streams a request from Ollama with unified stream events.
func (p *OllamaProvider) StreamMessage(ctx context.Context, req apitypes.MessageRequest) (<-chan apitypes.StreamEvent, error) {
	req.Model = CleanModelName(req.Model)
	return p.compatProv.StreamMessage(ctx, req)
}

// ListLocalModels retrieves the list of installed models from Ollama's GET /api/tags.
func (p *OllamaProvider) ListLocalModels(ctx context.Context) ([]OllamaModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.NativeURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("connecting to Ollama at %s: %w", p.NativeURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var tagsResp OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return nil, fmt.Errorf("decoding tags response: %w", err)
	}

	return tagsResp.Models, nil
}

// GenerateEmbedding generates a vector embedding for a single text using Ollama's /api/embeddings.
func (p *OllamaProvider) GenerateEmbedding(ctx context.Context, model string, prompt string) ([]float32, error) {
	if model == "" {
		model = "nomic-embed-text"
	}
	model = CleanModelName(model)

	// Try /api/embed first (newer Ollama API batch endpoint)
	embedReq := OllamaEmbedRequest{
		Model: model,
		Input: []string{prompt},
	}
	payload, err := json.Marshal(embedReq)
	if err == nil {
		req, rErr := http.NewRequestWithContext(ctx, http.MethodPost, p.NativeURL+"/api/embed", bytes.NewReader(payload))
		if rErr == nil {
			req.Header.Set("Content-Type", "application/json")
			if resp, dErr := p.Client.Do(req); dErr == nil {
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					var embedResp OllamaEmbedResponse
					if decErr := json.NewDecoder(resp.Body).Decode(&embedResp); decErr == nil && len(embedResp.Embeddings) > 0 {
						return embedResp.Embeddings[0], nil
					}
				}
			}
		}
	}

	// Fall back to classic /api/embeddings
	legacyReq := OllamaEmbeddingRequest{
		Model:  model,
		Prompt: prompt,
	}
	payload, err = json.Marshal(legacyReq)
	if err != nil {
		return nil, fmt.Errorf("marshaling embedding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.NativeURL+"/api/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("creating embedding request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling Ollama embedding API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embedding error (status %d): %s", resp.StatusCode, string(body))
	}

	var embResp OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("decoding embedding response: %w", err)
	}

	return embResp.Embedding, nil
}

// GenerateEmbeddingsBatch generates vector embeddings for a list of texts.
func (p *OllamaProvider) GenerateEmbeddingsBatch(ctx context.Context, model string, prompts []string) ([][]float32, error) {
	if len(prompts) == 0 {
		return nil, nil
	}
	if model == "" {
		model = "nomic-embed-text"
	}
	model = CleanModelName(model)

	// Try /api/embed batch call
	embedReq := OllamaEmbedRequest{
		Model: model,
		Input: prompts,
	}
	payload, err := json.Marshal(embedReq)
	if err == nil {
		req, rErr := http.NewRequestWithContext(ctx, http.MethodPost, p.NativeURL+"/api/embed", bytes.NewReader(payload))
		if rErr == nil {
			req.Header.Set("Content-Type", "application/json")
			if resp, dErr := p.Client.Do(req); dErr == nil {
				defer resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					var embedResp OllamaEmbedResponse
					if decErr := json.NewDecoder(resp.Body).Decode(&embedResp); decErr == nil && len(embedResp.Embeddings) == len(prompts) {
						return embedResp.Embeddings, nil
					}
				}
			}
		}
	}

	// Sequential fallback for servers without /api/embed
	result := make([][]float32, len(prompts))
	for i, prompt := range prompts {
		emb, err := p.GenerateEmbedding(ctx, model, prompt)
		if err != nil {
			return nil, fmt.Errorf("embedding prompt #%d: %w", i, err)
		}
		result[i] = emb
	}
	return result, nil
}

// sanitizeResponseToolCalls inspects text blocks for markdown tool call syntax
// (e.g. ```json { "name": "...", "parameters": {...} } ``` or <tool_call>)
// emitted by smaller local models that don't output OpenAI tool_calls objects natively.
var toolCallBlockRegex = regexp.MustCompile("(?s)```(?:json)?\\s*\\{\\s*\"(?:name|tool|tool_name)\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"(?:parameters|arguments|input)\"\\s*:\\s*(\\{.*?\\})\\s*\\}\\s*```")

func (p *OllamaProvider) sanitizeResponseToolCalls(resp *apitypes.MessageResponse) {
	if resp == nil || len(resp.Content) == 0 {
		return
	}

	// Check if tool_use is already present
	hasToolUse := false
	for _, block := range resp.Content {
		if block.Kind == "tool_use" {
			hasToolUse = true
			break
		}
	}
	if hasToolUse {
		return
	}

	// Attempt to extract structured tool calls from text
	var newBlocks []apitypes.OutputContentBlock
	for _, block := range resp.Content {
		if block.Kind != "text" {
			newBlocks = append(newBlocks, block)
			continue
		}

		matches := toolCallBlockRegex.FindAllStringSubmatchIndex(block.Text, -1)
		if len(matches) == 0 {
			newBlocks = append(newBlocks, block)
			continue
		}

		lastIdx := 0
		for _, m := range matches {
			// Preceding text
			if m[0] > lastIdx {
				textBefore := strings.TrimSpace(block.Text[lastIdx:m[0]])
				if textBefore != "" {
					newBlocks = append(newBlocks, apitypes.OutputContentBlock{Kind: "text", Text: textBefore})
				}
			}

			toolName := block.Text[m[2]:m[3]]
			argsRaw := block.Text[m[4]:m[5]]
			rawInput := json.RawMessage(argsRaw)
			var inputMap map[string]interface{}
			if err := json.Unmarshal([]byte(argsRaw), &inputMap); err != nil {
				rawInput = json.RawMessage("{}")
			}

			newBlocks = append(newBlocks, apitypes.OutputContentBlock{
				Kind:  "tool_use",
				ID:    fmt.Sprintf("ollama_call_%d", time.Now().UnixNano()),
				Name:  toolName,
				Input: rawInput,
			})
			lastIdx = m[1]
		}

		if lastIdx < len(block.Text) {
			tail := strings.TrimSpace(block.Text[lastIdx:])
			if tail != "" {
				newBlocks = append(newBlocks, apitypes.OutputContentBlock{Kind: "text", Text: tail})
			}
		}
	}

	resp.Content = newBlocks
}
