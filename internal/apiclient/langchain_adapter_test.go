package apiclient

import (
	"context"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
	"github.com/tmc/langchaingo/llms"
)

type mockLangChainModel struct {
	generateContentFn func(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error)
}

func (m *mockLangChainModel) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	if m.generateContentFn != nil {
		return m.generateContentFn(ctx, messages, options...)
	}
	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{Content: "mock response"},
		},
	}, nil
}

func (m *mockLangChainModel) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return "mock call", nil
}

func TestLangChainModelAdapter_SendMessage(t *testing.T) {
	mock := &mockLangChainModel{
		generateContentFn: func(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
			return &llms.ContentResponse{
				Choices: []*llms.ContentChoice{
					{Content: "Hello from LangChain adapter!"},
				},
			}, nil
		},
	}

	adapter := NewLangChainModelAdapter(mock, ProviderOllama)
	if adapter.Kind() != ProviderOllama {
		t.Errorf("expected ProviderOllama, got %v", adapter.Kind())
	}

	req := apitypes.MessageRequest{
		Model: "llama3.3",
		Messages: []apitypes.InputMessage{
			apitypes.UserText("Hello!"),
		},
	}

	resp, err := adapter.SendMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("SendMessage failed: %v", err)
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "Hello from LangChain adapter!" {
		t.Errorf("unexpected content: %v", resp.Content)
	}
}

func TestLangChainModelAdapter_StreamMessage(t *testing.T) {
	mock := &mockLangChainModel{
		generateContentFn: func(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
			opts := llms.CallOptions{}
			for _, opt := range options {
				opt(&opts)
			}
			if opts.StreamingFunc != nil {
				_ = opts.StreamingFunc(ctx, []byte("Streaming chunk 1... "))
				_ = opts.StreamingFunc(ctx, []byte("chunk 2!"))
			}
			return &llms.ContentResponse{
				Choices: []*llms.ContentChoice{
					{
						Content: "Streaming chunk 1... chunk 2!",
						ToolCalls: []llms.ToolCall{
							{
								ID:   "call_123",
								Type: "function",
								FunctionCall: &llms.FunctionCall{
									Name:      "FileReadTool",
									Arguments: `{"path":"README.md"}`,
								},
							},
						},
					},
				},
			}, nil
		},
	}

	adapter := NewLangChainModelAdapter(mock, ProviderOpenAi)
	req := apitypes.MessageRequest{
		Model: "gpt-4o",
		Messages: []apitypes.InputMessage{
			apitypes.UserText("Read README.md"),
		},
	}

	eventCh, err := adapter.StreamMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("StreamMessage failed: %v", err)
	}

	var textChunks []string
	var foundToolCall bool

	for ev := range eventCh {
		if ev.Kind == "content_block_delta" && ev.BlockDelta != nil {
			textChunks = append(textChunks, ev.BlockDelta.Text)
		}
		if ev.Kind == "content_block_start" && ev.ContentBlock != nil && ev.ContentBlock.Kind == "tool_use" {
			foundToolCall = true
			if ev.ContentBlock.Name != "FileReadTool" {
				t.Errorf("expected FileReadTool, got %s", ev.ContentBlock.Name)
			}
		}
	}

	if len(textChunks) != 2 {
		t.Fatalf("expected 2 text chunks, got %d", len(textChunks))
	}
	if !foundToolCall {
		t.Errorf("expected tool_use event in stream, got none")
	}
}
