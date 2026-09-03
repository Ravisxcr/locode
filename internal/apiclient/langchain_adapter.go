package apiclient

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
	"github.com/tmc/langchaingo/llms"
)

// LangChainModelAdapter adapts any langchaingo llms.Model into an apiclient.Provider.
type LangChainModelAdapter struct {
	model llms.Model
	kind  ProviderKind
}

// NewLangChainModelAdapter creates an adapter wrapping a langchaingo llms.Model.
func NewLangChainModelAdapter(model llms.Model, kind ProviderKind) *LangChainModelAdapter {
	return &LangChainModelAdapter{
		model: model,
		kind:  kind,
	}
}

func (a *LangChainModelAdapter) Kind() ProviderKind {
	return a.kind
}

// SendMessage sends a non-streaming message request to the underlying langchaingo Model.
func (a *LangChainModelAdapter) SendMessage(ctx context.Context, req apitypes.MessageRequest) (*apitypes.MessageResponse, error) {
	messages := convertRequestToLangChainMessages(req)
	options := buildLangChainOptions(req, nil)

	resp, err := a.model.GenerateContent(ctx, messages, options...)
	if err != nil {
		return nil, fmt.Errorf("langchain model error: %w", err)
	}

	return convertLangChainResponse(resp, req.Model), nil
}

// StreamMessage streams tokens and tool calls from the underlying langchaingo Model.
func (a *LangChainModelAdapter) StreamMessage(ctx context.Context, req apitypes.MessageRequest) (<-chan apitypes.StreamEvent, error) {
	eventCh := make(chan apitypes.StreamEvent, 100)

	go func() {
		defer close(eventCh)
		eventCh <- apitypes.StreamEvent{Kind: "message_start"}

		messages := convertRequestToLangChainMessages(req)

		hasEmittedStart := false
		options := buildLangChainOptions(req, func(ctx context.Context, chunk []byte) error {
			if len(chunk) > 0 {
				if !hasEmittedStart {
					hasEmittedStart = true
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
						Text: string(chunk),
					},
				}
			}
			return nil
		})

		resp, err := a.model.GenerateContent(ctx, messages, options...)
		if err != nil {
			eventCh <- apitypes.StreamEvent{
				Kind: "error",
				BlockDelta: &apitypes.ContentBlockDelta{
					Kind: "text_delta",
					Text: fmt.Sprintf("langchain stream error: %v", err),
				},
			}
			eventCh <- apitypes.StreamEvent{Kind: "message_stop"}
			return
		}

		if hasEmittedStart {
			eventCh <- apitypes.StreamEvent{
				Kind:  "content_block_stop",
				Index: 0,
			}
		}

		// Emit any tool calls from choices
		toolCallOffset := 0
		if hasEmittedStart {
			toolCallOffset = 1
		}

		if resp != nil {
			for _, choice := range resp.Choices {
				for j, tc := range choice.ToolCalls {
					callIdx := toolCallOffset + j
					var rawArgs json.RawMessage
					if tc.FunctionCall != nil {
						rawArgs = json.RawMessage(tc.FunctionCall.Arguments)
					} else {
						rawArgs = json.RawMessage("{}")
					}

					name := ""
					if tc.FunctionCall != nil {
						name = tc.FunctionCall.Name
					}

					eventCh <- apitypes.StreamEvent{
						Kind:  "content_block_start",
						Index: callIdx,
						ContentBlock: &apitypes.OutputContentBlock{
							Kind:  "tool_use",
							ID:    tc.ID,
							Name:  name,
							Input: rawArgs,
						},
					}
					eventCh <- apitypes.StreamEvent{
						Kind:  "content_block_stop",
						Index: callIdx,
					}
				}
			}
		}

		eventCh <- apitypes.StreamEvent{Kind: "message_stop"}
	}()

	return eventCh, nil
}

func convertRequestToLangChainMessages(req apitypes.MessageRequest) []llms.MessageContent {
	var messages []llms.MessageContent

	if req.System != "" {
		messages = append(messages, llms.TextParts(llms.ChatMessageTypeSystem, req.System))
	}

	for _, msg := range req.Messages {
		var role llms.ChatMessageType
		switch strings.ToLower(msg.Role) {
		case "assistant", "ai":
			role = llms.ChatMessageTypeAI
		case "system":
			role = llms.ChatMessageTypeSystem
		default:
			role = llms.ChatMessageTypeHuman
		}

		var parts []string
		for _, b := range msg.Content {
			if b.Kind == "text" && b.Text != "" {
				parts = append(parts, b.Text)
			} else if b.Kind == "tool_result" && b.Content != "" {
				parts = append(parts, fmt.Sprintf("[Tool Result %s]: %s", b.ToolUseID, b.Content))
			}
		}

		if len(parts) > 0 {
			messages = append(messages, llms.TextParts(role, parts...))
		}
	}

	return messages
}

func buildLangChainOptions(req apitypes.MessageRequest, streamFunc func(context.Context, []byte) error) []llms.CallOption {
	var opts []llms.CallOption

	if streamFunc != nil {
		opts = append(opts, llms.WithStreamingFunc(streamFunc))
	}
	if req.MaxTokens > 0 {
		opts = append(opts, llms.WithMaxTokens(req.MaxTokens))
	}

	// Convert tools
	if len(req.Tools) > 0 {
		var lcTools []llms.Tool
		for _, t := range req.Tools {
			var paramMap map[string]any
			if len(t.InputSchema) > 0 {
				_ = json.Unmarshal(t.InputSchema, &paramMap)
			}
			if paramMap == nil {
				paramMap = map[string]any{"type": "object"}
			}

			lcTools = append(lcTools, llms.Tool{
				Type: "function",
				Function: &llms.FunctionDefinition{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  paramMap,
				},
			})
		}
		opts = append(opts, llms.WithTools(lcTools))
	}

	return opts
}

func convertLangChainResponse(resp *llms.ContentResponse, model string) *apitypes.MessageResponse {
	if resp == nil || len(resp.Choices) == 0 {
		return &apitypes.MessageResponse{
			ID:         fmt.Sprintf("lc-%d", time.Now().UnixNano()),
			Model:      model,
			Role:       "assistant",
			StopReason: "end_turn",
		}
	}

	choice := resp.Choices[0]
	var blocks []apitypes.OutputContentBlock

	if choice.Content != "" {
		blocks = append(blocks, apitypes.OutputContentBlock{
			Kind: "text",
			Text: choice.Content,
		})
	}

	for i, tc := range choice.ToolCalls {
		var rawArgs json.RawMessage
		if tc.FunctionCall != nil {
			rawArgs = json.RawMessage(tc.FunctionCall.Arguments)
		} else {
			rawArgs = json.RawMessage("{}")
		}

		name := ""
		if tc.FunctionCall != nil {
			name = tc.FunctionCall.Name
		}

		blocks = append(blocks, apitypes.OutputContentBlock{
			Kind:  "tool_use",
			ID:    fmt.Sprintf("call_%d", i+1),
			Name:  name,
			Input: rawArgs,
		})
	}

	stopReason := "end_turn"
	if len(choice.ToolCalls) > 0 {
		stopReason = "tool_use"
	}

	return &apitypes.MessageResponse{
		ID:         fmt.Sprintf("lc-%d", time.Now().UnixNano()),
		Model:      model,
		Role:       "assistant",
		Content:    blocks,
		StopReason: stopReason,
	}
}
