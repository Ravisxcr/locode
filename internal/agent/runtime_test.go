package agent

import (
	"context"
	"encoding/json"
	"sync"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/apiclient"
	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

func TestExtractFallbackTools_FencedJSON(t *testing.T) {
	text := "Let's start by reading the `README.md` file to get an overview of what this project is about.\n\n```json\n{\"name\": \"FileReadTool\", \"arguments\": {\"path\": \"README.md\"}}\n```"
	blocks := []apitypes.OutputContentBlock{
		{Kind: "text", Text: text},
	}

	tools := extractFallbackTools(blocks)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].name != "FileReadTool" {
		t.Errorf("expected tool name FileReadTool, got %q", tools[0].name)
	}

	var args map[string]interface{}
	if err := json.Unmarshal(tools[0].input, &args); err != nil {
		t.Fatalf("unmarshaling input: %v", err)
	}
	if args["path"] != "README.md" {
		t.Errorf("expected path README.md, got %v", args["path"])
	}
}

func TestExtractFallbackTools_DirectJSON(t *testing.T) {
	text := `{"name": "BashTool", "arguments": {"command": "go version"}}`
	blocks := []apitypes.OutputContentBlock{
		{Kind: "text", Text: text},
	}

	tools := extractFallbackTools(blocks)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].name != "BashTool" {
		t.Errorf("expected BashTool, got %q", tools[0].name)
	}
}

func TestExtractFallbackTools_DirectJSONArray(t *testing.T) {
	text := `[{"name": "FileReadTool", "arguments": {"path": "a.txt"}}, {"name": "FileReadTool", "arguments": {"path": "b.txt"}}]`
	blocks := []apitypes.OutputContentBlock{
		{Kind: "text", Text: text},
	}

	tools := extractFallbackTools(blocks)
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	if tools[0].name != "FileReadTool" || tools[1].name != "FileReadTool" {
		t.Errorf("unexpected tool names: %v, %v", tools[0].name, tools[1].name)
	}
}

func TestExtractFallbackTools_XML(t *testing.T) {
	text := `I will read the file. <tool_call>{"name": "FileReadTool", "arguments": {"path": "go.mod"}}</tool_call>`
	blocks := []apitypes.OutputContentBlock{
		{Kind: "text", Text: text},
	}

	tools := extractFallbackTools(blocks)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].name != "FileReadTool" {
		t.Errorf("expected FileReadTool, got %q", tools[0].name)
	}
}

func TestExtractFallbackTools_ArgsFirstRegex(t *testing.T) {
	text := `Here is the invocation: {"arguments": {"path": "main.go"}, "name": "FileReadTool"}`
	blocks := []apitypes.OutputContentBlock{
		{Kind: "text", Text: text},
	}

	tools := extractFallbackTools(blocks)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].name != "FileReadTool" {
		t.Errorf("expected FileReadTool, got %q", tools[0].name)
	}
	var args map[string]interface{}
	_ = json.Unmarshal(tools[0].input, &args)
	if args["path"] != "main.go" {
		t.Errorf("expected path main.go, got %v", args["path"])
	}
}

func TestPermissions_ReadOnlyAutoApproved(t *testing.T) {
	policy := &PermissionPolicy{
		Mode: WorkspaceWrite,
		Prompter: &mockPrompter{
			promptFn: func(toolName, operation string) (bool, error) {
				t.Fatalf("Prompter should NOT be called for read-only tool %s", toolName)
				return false, nil
			},
		},
	}

	readTools := []string{
		"FileReadTool", "file_read", "read_file",
		"GlobTool", "glob",
		"GrepTool", "grep",
		"ListDirectoryTool", "list_dir",
		"rag_search", "rag_code_context",
	}

	for _, tool := range readTools {
		allowed, reason := policy.Authorize(tool, "{}")
		if !allowed {
			t.Errorf("expected tool %s to be auto-approved, got denied: %s", tool, reason)
		}
	}
}

func TestPermissions_MutatingPrompts(t *testing.T) {
	promptCount := 0
	policy := &PermissionPolicy{
		Mode: WorkspaceWrite,
		Prompter: &mockPrompter{
			promptFn: func(toolName, operation string) (bool, error) {
				promptCount++
				return true, nil
			},
		},
	}

	allowed, _ := policy.Authorize("BashTool", `{"command":"rm -rf /"}`)
	if !allowed || promptCount != 1 {
		t.Errorf("expected BashTool to prompt once and be allowed, count=%d", promptCount)
	}

	allowed, _ = policy.Authorize("FileWriteTool", `{"path":"test.txt"}`)
	if !allowed || promptCount != 2 {
		t.Errorf("expected FileWriteTool to prompt and be allowed, count=%d", promptCount)
	}
}

type mockPrompter struct {
	promptFn func(toolName, operation string) (bool, error)
}

func (m *mockPrompter) Prompt(toolName, operation string) (bool, error) {
	if m.promptFn != nil {
		return m.promptFn(toolName, operation)
	}
	return true, nil
}

type mockProvider struct {
	mu           sync.Mutex
	turns        []func(req apitypes.MessageRequest) []apitypes.StreamEvent
	currentTurn  int
	receivedReqs []apitypes.MessageRequest
}

func (m *mockProvider) Kind() apiclient.ProviderKind {
	return apiclient.ProviderOpenAi
}

func (m *mockProvider) SendMessage(ctx context.Context, req apitypes.MessageRequest) (*apitypes.MessageResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.receivedReqs = append(m.receivedReqs, req)
	if m.currentTurn >= len(m.turns) {
		return &apitypes.MessageResponse{
			Role:    "assistant",
			Content: []apitypes.OutputContentBlock{{Kind: "text", Text: "Done"}},
		}, nil
	}
	events := m.turns[m.currentTurn](req)
	m.currentTurn++

	resp := &apitypes.MessageResponse{Role: "assistant"}
	for _, ev := range events {
		if ev.ContentBlock != nil {
			resp.Content = append(resp.Content, *ev.ContentBlock)
		}
	}
	return resp, nil
}

func (m *mockProvider) StreamMessage(ctx context.Context, req apitypes.MessageRequest) (<-chan apitypes.StreamEvent, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.receivedReqs = append(m.receivedReqs, req)
	ch := make(chan apitypes.StreamEvent, 10)

	var events []apitypes.StreamEvent
	if m.currentTurn < len(m.turns) {
		events = m.turns[m.currentTurn](req)
		m.currentTurn++
	} else {
		events = []apitypes.StreamEvent{
			{Kind: "content_block_delta", Index: 0, BlockDelta: &apitypes.ContentBlockDelta{Kind: "text_delta", Text: "Done"}},
			{Kind: "message_stop"},
		}
	}

	go func() {
		defer close(ch)
		for _, ev := range events {
			ch <- ev
		}
	}()
	return ch, nil
}

func TestStreamLoop_NativeToolUse_MultiTurn(t *testing.T) {
	toolExecCount := 0
	executor := NewStaticExecutor().Register("filereadtool", func(params map[string]interface{}) apitypes.ToolResult {
		toolExecCount++
		return apitypes.ToolResult{Output: "# Project Readme\nThis is a test project.", IsError: false}
	})

	prov := &mockProvider{
		turns: []func(req apitypes.MessageRequest) []apitypes.StreamEvent{
			// Turn 0: LLM emits native tool use
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind:  "content_block_start",
						Index: 0,
						ContentBlock: &apitypes.OutputContentBlock{
							Kind:  "tool_use",
							ID:    "call_native_1",
							Name:  "FileReadTool",
							Input: json.RawMessage(`{"path":"README.md"}`),
						},
					},
					{Kind: "message_stop"},
				}
			},
			// Turn 1: LLM receives tool output and gives final response
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind:  "content_block_delta",
						Index: 0,
						BlockDelta: &apitypes.ContentBlockDelta{
							Kind: "text_delta",
							Text: "This project is a test project based on the README.",
						},
					},
					{Kind: "message_stop"},
				}
			},
		},
	}

	rt := NewConversationRuntime(RuntimeOptions{
		Provider: prov,
		Executor: executor,
		Model:    "test-model",
	})

	eventCh, err := rt.StreamUserMessage(context.Background(), "Explain the code")
	if err != nil {
		t.Fatalf("StreamUserMessage error: %v", err)
	}

	var outputText string
	for ev := range eventCh {
		if ev.BlockDelta != nil && ev.BlockDelta.Kind == "text_delta" {
			outputText += ev.BlockDelta.Text
		}
	}

	if toolExecCount != 1 {
		t.Errorf("expected tool to execute 1 time, got %d", toolExecCount)
	}

	if outputText != "This project is a test project based on the README." {
		t.Errorf("unexpected output text: %q", outputText)
	}

	// Verify session history has correct turns
	session := rt.GetSession()
	if len(session) < 3 {
		t.Fatalf("expected at least 3 session messages (user, assistant, user-tool-results), got %d", len(session))
	}

	// Check turn 2 (tool results message)
	toolResultMsg := session[2]
	if toolResultMsg.Role != "user" {
		t.Errorf("expected tool results to have role 'user', got %q", toolResultMsg.Role)
	}
	if len(toolResultMsg.Content) != 1 || toolResultMsg.Content[0].Kind != "tool_result" {
		t.Errorf("expected 1 tool_result block in message, got %+v", toolResultMsg.Content)
	}
}

func TestStreamLoop_FallbackToolUse_MultiTurn(t *testing.T) {
	toolExecCount := 0
	executor := NewStaticExecutor().Register("filereadtool", func(params map[string]interface{}) apitypes.ToolResult {
		toolExecCount++
		return apitypes.ToolResult{Output: "# Project Readme\nThis is a test project.", IsError: false}
	})

	prov := &mockProvider{
		turns: []func(req apitypes.MessageRequest) []apitypes.StreamEvent{
			// Turn 0: LLM emits prose with fenced json (the user scenario)
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind:  "content_block_delta",
						Index: 0,
						BlockDelta: &apitypes.ContentBlockDelta{
							Kind: "text_delta",
							Text: "Let's start by reading the `README.md` file to get an overview of what this project is about.\n\n```json\n{\"name\": \"FileReadTool\", \"arguments\": {\"path\": \"README.md\"}}\n```",
						},
					},
					{Kind: "message_stop"},
				}
			},
			// Turn 1: LLM receives tool output and explains
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind:  "content_block_delta",
						Index: 0,
						BlockDelta: &apitypes.ContentBlockDelta{
							Kind: "text_delta",
							Text: "The codebase is a test project.",
						},
					},
					{Kind: "message_stop"},
				}
			},
		},
	}

	rt := NewConversationRuntime(RuntimeOptions{
		Provider: prov,
		Executor: executor,
		Model:    "test-model",
	})

	eventCh, err := rt.StreamUserMessage(context.Background(), "Explain the code")
	if err != nil {
		t.Fatalf("StreamUserMessage error: %v", err)
	}

	var outputText string
	for ev := range eventCh {
		if ev.BlockDelta != nil && ev.BlockDelta.Kind == "text_delta" {
			outputText += ev.BlockDelta.Text
		}
	}

	if toolExecCount != 1 {
		t.Errorf("expected tool to execute 1 time, got %d", toolExecCount)
	}

	if !contains(outputText, "The codebase is a test project.") {
		t.Errorf("expected final answer to be streamed, got %q", outputText)
	}
}

func TestSendLoop_NativeToolUse_MultiTurn(t *testing.T) {
	toolExecCount := 0
	executor := NewStaticExecutor().Register("filereadtool", func(params map[string]interface{}) apitypes.ToolResult {
		toolExecCount++
		return apitypes.ToolResult{Output: "# Project Readme\nThis is a test project.", IsError: false}
	})

	prov := &mockProvider{
		turns: []func(req apitypes.MessageRequest) []apitypes.StreamEvent{
			// Turn 0: LLM emits native tool use
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind: "content_block_start",
						ContentBlock: &apitypes.OutputContentBlock{
							Kind:  "tool_use",
							ID:    "call_send_1",
							Name:  "FileReadTool",
							Input: json.RawMessage(`{"path":"README.md"}`),
						},
					},
				}
			},
			// Turn 1: LLM receives tool output and gives response
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind: "content_block_start",
						ContentBlock: &apitypes.OutputContentBlock{
							Kind: "text",
							Text: "Project is documented in README.",
						},
					},
				}
			},
		},
	}

	rt := NewConversationRuntime(RuntimeOptions{
		Provider: prov,
		Executor: executor,
		Model:    "test-model",
	})

	resp, err := rt.SendUserMessage(context.Background(), "Explain the code")
	if err != nil {
		t.Fatalf("SendUserMessage error: %v", err)
	}

	if toolExecCount != 1 {
		t.Errorf("expected tool to execute 1 time, got %d", toolExecCount)
	}

	if len(resp.Content) == 0 || resp.Content[0].Text != "Project is documented in README." {
		t.Errorf("unexpected response content: %+v", resp.Content)
	}
}

func TestSendLoop_FallbackToolUse_MultiTurn(t *testing.T) {
	toolExecCount := 0
	executor := NewStaticExecutor().Register("filereadtool", func(params map[string]interface{}) apitypes.ToolResult {
		toolExecCount++
		return apitypes.ToolResult{Output: "# Project Readme\nThis is a test project.", IsError: false}
	})

	prov := &mockProvider{
		turns: []func(req apitypes.MessageRequest) []apitypes.StreamEvent{
			// Turn 0: LLM emits fallback json tool call in text
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind: "content_block_start",
						ContentBlock: &apitypes.OutputContentBlock{
							Kind: "text",
							Text: "Let me check the README.\n\n```json\n{\"name\": \"FileReadTool\", \"arguments\": {\"path\": \"README.md\"}}\n```",
						},
					},
				}
			},
			// Turn 1: LLM receives tool output and gives response
			func(req apitypes.MessageRequest) []apitypes.StreamEvent {
				return []apitypes.StreamEvent{
					{
						Kind: "content_block_start",
						ContentBlock: &apitypes.OutputContentBlock{
							Kind: "text",
							Text: "Project is documented in README.",
						},
					},
				}
			},
		},
	}

	rt := NewConversationRuntime(RuntimeOptions{
		Provider: prov,
		Executor: executor,
		Model:    "test-model",
	})

	resp, err := rt.SendUserMessage(context.Background(), "Explain the code")
	if err != nil {
		t.Fatalf("SendUserMessage error: %v", err)
	}

	if toolExecCount != 1 {
		t.Errorf("expected tool to execute 1 time, got %d", toolExecCount)
	}

	if len(resp.Content) == 0 || resp.Content[0].Text != "Project is documented in README." {
		t.Errorf("unexpected response content: %+v", resp.Content)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || (len(s) > 0 && len(substr) > 0 && stringContains(s, substr)))
}

func stringContains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
