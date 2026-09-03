package apiclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
	"github.com/ollama/ollama/api"
)

// OllamaModelInfo represents a model returned by Ollama's tags/list endpoint.
type OllamaModelInfo struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	ModifiedAt time.Time `json:"modified_at"`
}

// OllamaProvider communicates with local or remote Ollama servers using the official Ollama Go SDK.
type OllamaProvider struct {
	Host       string
	Client     *api.Client
	HTTPClient *http.Client
}

// NormalizeOllamaHost normalizes a host string, ensuring a valid http:// or https:// scheme.
func NormalizeOllamaHost(host string) string {
	h := strings.TrimSpace(host)
	if h == "" {
		return "http://localhost:11434"
	}
	if h == "localhost" {
		return "http://localhost:11434"
	}
	if h == "127.0.0.1" {
		return "http://127.0.0.1:11434"
	}
	h = strings.TrimRight(h, "/")
	h = strings.TrimSuffix(h, "/v1")
	h = strings.TrimRight(h, "/")

	if !strings.HasPrefix(h, "http://") && !strings.HasPrefix(h, "https://") {
		h = "http://" + h
	}

	return h
}

// DefaultOllamaHost returns the resolved Ollama host string.
func DefaultOllamaHost() string {
	if host := os.Getenv("OLLAMA_HOST"); host != "" {
		return NormalizeOllamaHost(host)
	}
	if baseURL := os.Getenv("OLLAMA_BASE_URL"); baseURL != "" {
		return NormalizeOllamaHost(baseURL)
	}
	return "http://localhost:11434"
}

// NewOllamaProvider creates a new Ollama provider powered by the official Ollama SDK.
func NewOllamaProvider(host string, auth apitypes.AuthSource) *OllamaProvider {
	if host == "" {
		host = DefaultOllamaHost()
	} else {
		host = NormalizeOllamaHost(host)
	}

	u, err := url.Parse(host)
	httpClient := &http.Client{Timeout: 5 * time.Minute}

	var client *api.Client
	if err == nil {
		client = api.NewClient(u, httpClient)
	} else {
		client, _ = api.ClientFromEnvironment()
	}

	return &OllamaProvider{
		Host:       host,
		Client:     client,
		HTTPClient: httpClient,
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

// buildChatRequest converts an apitypes.MessageRequest into the official api.ChatRequest.
func (p *OllamaProvider) buildChatRequest(req apitypes.MessageRequest, stream bool) *api.ChatRequest {
	var msgs []api.Message

	if req.System != "" {
		msgs = append(msgs, api.Message{
			Role:    "system",
			Content: req.System,
		})
	}

	for _, m := range req.Messages {
		role := strings.ToLower(m.Role)
		if role == "assistant" {
			var textBuilder strings.Builder
			var toolCalls []api.ToolCall
			for _, b := range m.Content {
				switch b.Kind {
				case "text":
					textBuilder.WriteString(b.Text)
				case "tool_use":
					var args map[string]interface{}
					if len(b.Input) > 0 {
						_ = json.Unmarshal(b.Input, &args)
					}
					toolCalls = append(toolCalls, api.ToolCall{
						Function: api.ToolCallFunction{
							Name:      b.Name,
							Arguments: args,
						},
					})
				}
			}
			msgs = append(msgs, api.Message{
				Role:      "assistant",
				Content:   textBuilder.String(),
				ToolCalls: toolCalls,
			})
			continue
		}

		for _, b := range m.Content {
			switch b.Kind {
			case "text":
				msgs = append(msgs, api.Message{
					Role:    "user",
					Content: b.Text,
				})
			case "tool_result":
				msgs = append(msgs, api.Message{
					Role:    "tool",
					Content: b.Content,
				})
			}
		}
	}

	var tools api.Tools
	for _, t := range req.Tools {
		var schema map[string]interface{}
		if len(t.InputSchema) > 0 {
			_ = json.Unmarshal(t.InputSchema, &schema)
		}

		props := make(map[string]struct {
			Type        string   `json:"type"`
			Description string   `json:"description"`
			Enum        []string `json:"enum,omitempty"`
		})
		var required []string

		if schema != nil {
			if r, ok := schema["required"].([]interface{}); ok {
				for _, reqField := range r {
					if s, ok := reqField.(string); ok {
						required = append(required, s)
					}
				}
			}
			if pMap, ok := schema["properties"].(map[string]interface{}); ok {
				for k, v := range pMap {
					if vm, ok := v.(map[string]interface{}); ok {
						tp := struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						}{}
						if tType, ok := vm["type"].(string); ok {
							tp.Type = tType
						}
						if tDesc, ok := vm["description"].(string); ok {
							tp.Description = tDesc
						}
						if enumList, ok := vm["enum"].([]interface{}); ok {
							for _, e := range enumList {
								if es, ok := e.(string); ok {
									tp.Enum = append(tp.Enum, es)
								}
							}
						}
						props[k] = tp
					}
				}
			}
		}

		tools = append(tools, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters: struct {
					Type       string   `json:"type"`
					Required   []string `json:"required"`
					Properties map[string]struct {
						Type        string   `json:"type"`
						Description string   `json:"description"`
						Enum        []string `json:"enum,omitempty"`
					} `json:"properties"`
				}{
					Type:       "object",
					Required:   required,
					Properties: props,
				},
			},
		})
	}

	var chatTools api.Tools
	if len(tools) > 0 {
		chatTools = tools
	}

	options := make(map[string]interface{})
	if req.MaxTokens > 0 {
		options["num_predict"] = req.MaxTokens
	}

	return &api.ChatRequest{
		Model:    CleanModelName(req.Model),
		Messages: msgs,
		Stream:   &stream,
		Tools:    chatTools,
		Options:  options,
	}
}

// SendMessage sends a request to Ollama via the official Ollama SDK.
func (p *OllamaProvider) SendMessage(ctx context.Context, req apitypes.MessageRequest) (*apitypes.MessageResponse, error) {
	chatReq := p.buildChatRequest(req, false)

	var finalResp *api.ChatResponse
	err := p.Client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
		finalResp = &resp
		return nil
	})
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "not found") {
			return nil, fmt.Errorf("ollama sdk chat error (%s): %w\nHint: run 'ollama pull %s' on %s or check installed models with './bin/gocode ollama list --host %s'", p.Host, err, chatReq.Model, p.Host, p.Host)
		}
		return nil, fmt.Errorf("ollama sdk chat error (%s): %w", p.Host, err)
	}

	if finalResp == nil {
		return nil, fmt.Errorf("empty response from ollama at %s", p.Host)
	}

	var blocks []apitypes.OutputContentBlock
	if finalResp.Message.Content != "" {
		blocks = append(blocks, apitypes.OutputContentBlock{
			Kind: "text",
			Text: finalResp.Message.Content,
		})
	}

	for i, tc := range finalResp.Message.ToolCalls {
		rawInput, _ := json.Marshal(tc.Function.Arguments)
		blocks = append(blocks, apitypes.OutputContentBlock{
			Kind:  "tool_use",
			ID:    fmt.Sprintf("call_%d", i+1),
			Name:  tc.Function.Name,
			Input: rawInput,
		})
	}

	resp := &apitypes.MessageResponse{
		ID:         fmt.Sprintf("ollama-%d", time.Now().UnixNano()),
		Model:      finalResp.Model,
		Role:       "assistant",
		Content:    blocks,
		StopReason: "end_turn",
		Usage: apitypes.Usage{
			InputTokens:  finalResp.PromptEvalCount,
			OutputTokens: finalResp.EvalCount,
		},
	}

	p.sanitizeResponseToolCalls(resp)
	return resp, nil
}

// StreamMessage streams a request from Ollama via the official Ollama SDK.
func (p *OllamaProvider) StreamMessage(ctx context.Context, req apitypes.MessageRequest) (<-chan apitypes.StreamEvent, error) {
	chatReq := p.buildChatRequest(req, true)
	eventCh := make(chan apitypes.StreamEvent, 100)

	go func() {
		defer close(eventCh)
		eventCh <- apitypes.StreamEvent{Kind: "message_start"}

		toolCallIdx := 0
		hasText := false
		err := p.Client.Chat(ctx, chatReq, func(resp api.ChatResponse) error {
			if resp.Message.Content != "" {
				if !hasText {
					hasText = true
					eventCh <- apitypes.StreamEvent{
						Kind:  "content_block_start",
						Index: 0,
						ContentBlock: &apitypes.OutputContentBlock{
							Kind: "text",
						},
					}
				}
				eventCh <- apitypes.StreamEvent{
					Kind:  "content_block_delta",
					Index: 0,
					BlockDelta: &apitypes.ContentBlockDelta{
						Kind: "text_delta",
						Text: resp.Message.Content,
					},
				}
			}

			for _, tc := range resp.Message.ToolCalls {
				idx := toolCallIdx
				if hasText {
					idx = toolCallIdx + 1
				}
				rawInput, _ := json.Marshal(tc.Function.Arguments)
				eventCh <- apitypes.StreamEvent{
					Kind:  "content_block_start",
					Index: idx,
					ContentBlock: &apitypes.OutputContentBlock{
						Kind:  "tool_use",
						ID:    fmt.Sprintf("call_%d", toolCallIdx+1),
						Name:  tc.Function.Name,
						Input: rawInput,
					},
				}
				toolCallIdx++
			}

			if resp.Done {
				eventCh <- apitypes.StreamEvent{
					Kind: "message_delta",
					DeltaUsage: &apitypes.Usage{
						InputTokens:  resp.PromptEvalCount,
						OutputTokens: resp.EvalCount,
					},
				}
				eventCh <- apitypes.StreamEvent{Kind: "message_stop"}
			}
			return nil
		})

		if err != nil {
			msg := fmt.Sprintf("Error from Ollama (%s): %v", p.Host, err)
			if strings.Contains(strings.ToLower(err.Error()), "not found") {
				msg += fmt.Sprintf("\n\nHint: Model %q is not installed on %s.\nTo pull it, run on the Ollama host:\n  ollama pull %s\nOr check installed models with:\n  ./bin/gocode ollama list --host %s", chatReq.Model, p.Host, chatReq.Model, p.Host)
			}
			eventCh <- apitypes.StreamEvent{
				Kind: "error",
				BlockDelta: &apitypes.ContentBlockDelta{
					Kind: "text_delta",
					Text: msg,
				},
			}
			eventCh <- apitypes.StreamEvent{Kind: "message_stop"}
		}
	}()

	return eventCh, nil
}

// ListLocalModels retrieves the list of installed models using the official Ollama SDK.
func (p *OllamaProvider) ListLocalModels(ctx context.Context) ([]OllamaModelInfo, error) {
	listResp, err := p.Client.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("connecting to Ollama at %s: %w", p.Host, err)
	}

	var models []OllamaModelInfo
	for _, m := range listResp.Models {
		models = append(models, OllamaModelInfo{
			Name:       m.Name,
			Model:      m.Model,
			Size:       m.Size,
			Digest:     m.Digest,
			ModifiedAt: m.ModifiedAt,
		})
	}
	return models, nil
}

// GenerateEmbedding generates a vector embedding for a single text using the official Ollama SDK.
func (p *OllamaProvider) GenerateEmbedding(ctx context.Context, model string, prompt string) ([]float32, error) {
	if model == "" {
		model = "nomic-embed-text"
	}
	model = CleanModelName(model)

	// Try Embed first
	embedResp, err := p.Client.Embed(ctx, &api.EmbedRequest{
		Model: model,
		Input: []string{prompt},
	})
	if err == nil && len(embedResp.Embeddings) > 0 {
		return embedResp.Embeddings[0], nil
	}

	// Fallback to Embeddings
	resp, err := p.Client.Embeddings(ctx, &api.EmbeddingRequest{
		Model:  model,
		Prompt: prompt,
	})
	if err != nil {
		return nil, fmt.Errorf("calling Ollama embedding SDK: %w", err)
	}

	var emb []float32
	for _, v := range resp.Embedding {
		emb = append(emb, float32(v))
	}
	return emb, nil
}

// GenerateEmbeddingsBatch generates vector embeddings for a list of texts using the official Ollama SDK.
func (p *OllamaProvider) GenerateEmbeddingsBatch(ctx context.Context, model string, prompts []string) ([][]float32, error) {
	if len(prompts) == 0 {
		return nil, nil
	}
	if model == "" {
		model = "nomic-embed-text"
	}
	model = CleanModelName(model)

	embedResp, err := p.Client.Embed(ctx, &api.EmbedRequest{
		Model: model,
		Input: prompts,
	})
	if err == nil && len(embedResp.Embeddings) == len(prompts) {
		return embedResp.Embeddings, nil
	}

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
// emitted by smaller local models that don't output tool_calls objects natively.
var toolCallBlockRegex = regexp.MustCompile("(?s)```(?:json)?\\s*\\{\\s*\"(?:name|tool|tool_name)\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"(?:parameters|arguments|input)\"\\s*:\\s*(\\{.*?\\})\\s*\\}\\s*```")

func (p *OllamaProvider) sanitizeResponseToolCalls(resp *apitypes.MessageResponse) {
	if resp == nil || len(resp.Content) == 0 {
		return
	}

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
