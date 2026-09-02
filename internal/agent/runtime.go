package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/Ravisxcr/gocode-rag/internal/apiclient"
	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

// RuntimeOptions configures a ConversationRuntime.
type RuntimeOptions struct {
	Provider      apiclient.Provider
	Executor      ToolExecutor
	Model         string
	MaxTokens     int
	MaxIterations int
	SystemPrompt  string
	PermMode      PermissionMode
	Prompter      PermissionPrompter
	Trusted       *TrustedToolStore
	Hooks         HookRunner
	ToolCb        ToolCallback
}

// ToolCallback is called before and after tool execution for UI updates.
type ToolCallback interface {
	OnToolStart(name string, input map[string]interface{})
	OnToolEnd(name string, isError bool)
}

// DetailedToolCallback is an optional interface for tool callbacks that accept result output.
type DetailedToolCallback interface {
	OnToolEndWithResult(name string, success bool, output string)
}

// NoOpToolCallback does nothing.
type NoOpToolCallback struct{}

func (NoOpToolCallback) OnToolStart(string, map[string]interface{})             {}
func (NoOpToolCallback) OnToolEnd(string, bool)                                 {}
func (NoOpToolCallback) OnToolEndWithResult(string, bool, string)               {}

// ConversationRuntime orchestrates the agentic tool-use loop.
//  ConversationRuntime<C: ApiClient, T: ToolExecutor>.
type ConversationRuntime struct {
	provider     apiclient.Provider
	executor     ToolExecutor
	session      []apitypes.InputMessage
	model        string
	maxTokens    int
	maxIter      int
	systemPrompt string
	permPolicy   PermissionPolicy
	hooks        HookRunner
	usage        UsageTracker
	toolCb       ToolCallback
}

// NewConversationRuntime creates a new runtime from options.
func NewConversationRuntime(opts RuntimeOptions) *ConversationRuntime {
	hooks := opts.Hooks
	if hooks == nil {
		hooks = NoOpHookRunner{}
	}
	prompter := opts.Prompter
	if prompter == nil {
		prompter = AllowAllPrompter{}
	}
	maxIter := opts.MaxIterations
	if maxIter <= 0 {
		maxIter = 30
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}
	toolCb := opts.ToolCb
	if toolCb == nil {
		toolCb = NoOpToolCallback{}
	}
	return &ConversationRuntime{
		provider:     opts.Provider,
		executor:     opts.Executor,
		model:        opts.Model,
		maxTokens:    maxTokens,
		maxIter:      maxIter,
		systemPrompt: opts.SystemPrompt,
		permPolicy:   PermissionPolicy{Mode: opts.PermMode, Prompter: prompter, Trusted: opts.Trusted},
		hooks:        hooks,
		toolCb:       toolCb,
	}
}

// SendUserMessage runs the full agent loop: send prompt, execute tools, loop until done.
func (r *ConversationRuntime) SendUserMessage(ctx context.Context, text string) (*apitypes.MessageResponse, error) {
	r.session = append(r.session, apitypes.UserText(text))
	return r.sendLoop(ctx)
}

// SendWithMessage runs the full agent loop with a pre-built message (for multimodal input).
func (r *ConversationRuntime) SendWithMessage(ctx context.Context, msg apitypes.InputMessage) (*apitypes.MessageResponse, error) {
	r.session = append(r.session, msg)
	return r.sendLoop(ctx)
}

func (r *ConversationRuntime) sendLoop(ctx context.Context) (*apitypes.MessageResponse, error) {
	for iteration := 0; iteration < r.maxIter; iteration++ {
		req := r.buildRequest()
		resp, err := r.provider.SendMessage(ctx, req)
		if err != nil {
			return nil, err
		}
		r.usage.Add(resp.Usage)

		// Build assistant message from response content
		assistantMsg := apitypes.InputMessage{Role: "assistant"}
		var pendingTools []toolUseInfo
		for _, block := range resp.Content {
			switch block.Kind {
			case "text":
				assistantMsg.Content = append(assistantMsg.Content, apitypes.InputContentBlock{Kind: "text", Text: block.Text})
			case "tool_use":
				assistantMsg.Content = append(assistantMsg.Content, apitypes.InputContentBlock{
					Kind: "tool_use", ID: block.ID, Name: block.Name, Input: block.Input,
				})
				pendingTools = append(pendingTools, toolUseInfo{id: block.ID, name: block.Name, input: block.Input})
			}
		}

		// Fallback tool extraction for models that output tool calls in text
		if len(pendingTools) == 0 {
			pendingTools = extractFallbackTools(resp.Content)
			for _, tu := range pendingTools {
				hasUse := false
				for _, b := range assistantMsg.Content {
					if b.Kind == "tool_use" && b.ID == tu.id {
						hasUse = true
						break
					}
				}
				if !hasUse {
					assistantMsg.Content = append(assistantMsg.Content, apitypes.InputContentBlock{
						Kind: "tool_use", ID: tu.id, Name: tu.name, Input: tu.input,
					})
				}
			}
		}
		r.session = append(r.session, assistantMsg)

		// No tool calls — we're done
		if len(pendingTools) == 0 {
			return resp, nil
		}

		// Execute each tool and batch results into a single user message
		var toolResultBlocks []apitypes.InputContentBlock
		for _, tu := range pendingTools {
			result := r.executeTool(tu)
			toolResultBlocks = append(toolResultBlocks, apitypes.InputContentBlock{
				Kind:      "tool_result",
				ToolUseID: tu.id,
				Content:   result.Output,
				IsError:   result.IsError,
			})
		}
		r.session = append(r.session, apitypes.InputMessage{
			Role:    "user",
			Content: toolResultBlocks,
		})
	}

	// Max iterations exceeded
	return &apitypes.MessageResponse{
		Type:       "message",
		Role:       "assistant",
		StopReason: "max_iterations",
		Content:    []apitypes.OutputContentBlock{{Kind: "text", Text: fmt.Sprintf("Agent loop exceeded maximum of %d iterations", r.maxIter)}},
		Usage:      apitypes.Usage{},
	}, nil
}

// StreamUserMessage runs the agent loop with streaming responses.
// Returns a channel that emits StreamEvents for the current turn.
// For multi-turn tool loops, it internally handles tool execution and re-streams.
func (r *ConversationRuntime) StreamUserMessage(ctx context.Context, text string) (<-chan apitypes.StreamEvent, error) {
	r.session = append(r.session, apitypes.UserText(text))
	return r.streamLoop(ctx)
}

// StreamWithMessage runs the agent loop with streaming for a pre-built message (for multimodal input).
func (r *ConversationRuntime) StreamWithMessage(ctx context.Context, msg apitypes.InputMessage) (<-chan apitypes.StreamEvent, error) {
	r.session = append(r.session, msg)
	return r.streamLoop(ctx)
}

func (r *ConversationRuntime) streamLoop(ctx context.Context) (<-chan apitypes.StreamEvent, error) {
	outCh := make(chan apitypes.StreamEvent, 64)
	go func() {
		defer close(outCh)
		for iteration := 0; iteration < r.maxIter; iteration++ {
			req := r.buildRequest()
			eventCh, err := r.provider.StreamMessage(ctx, req)
			if err != nil {
				// Send an error event so the REPL can display it
				outCh <- apitypes.StreamEvent{
					Kind: "error",
					BlockDelta: &apitypes.ContentBlockDelta{
						Kind: "text_delta",
						Text: "Error: " + err.Error(),
					},
				}
				return
			}

			// Collect the full response while forwarding events
			var contentBlocks []apitypes.OutputContentBlock
			var pendingTools []toolUseInfo
			var currentUsage apitypes.Usage

			for ev := range eventCh {
				select {
				case outCh <- ev:
				case <-ctx.Done():
					return
				}
				// Track content blocks from stream events
				switch ev.Kind {
				case "content_block_start":
					if ev.ContentBlock != nil {
						// Ensure contentBlocks is large enough for the index
						for len(contentBlocks) <= ev.Index {
							contentBlocks = append(contentBlocks, apitypes.OutputContentBlock{})
						}
						contentBlocks[ev.Index] = *ev.ContentBlock
					}
				case "content_block_delta":
					if ev.BlockDelta != nil {
						// Grow contentBlocks if needed (OpenAI uses offset indices)
						for len(contentBlocks) <= ev.Index {
							contentBlocks = append(contentBlocks, apitypes.OutputContentBlock{})
						}
						block := &contentBlocks[ev.Index]
						switch ev.BlockDelta.Kind {
						case "text_delta":
							if block.Kind == "" {
								block.Kind = "text"
							}
							block.Text += ev.BlockDelta.Text
						case "input_json_delta":
							if block.Kind == "" {
								block.Kind = "tool_use"
							}
							block.Input = appendJSON(block.Input, ev.BlockDelta.PartialJSON)
						}
					}
				case "message_delta":
					if ev.DeltaUsage != nil {
						currentUsage = *ev.DeltaUsage
					}
				}
			}

			r.usage.Add(currentUsage)

			// 1. Collect native tool calls from contentBlocks
			for _, block := range contentBlocks {
				if block.Kind == "tool_use" {
					pendingTools = append(pendingTools, toolUseInfo{
						id:    block.ID,
						name:  block.Name,
						input: block.Input,
					})
				}
			}

			// 2. If no native tool calls, attempt fallback tool extraction from text blocks
			if len(pendingTools) == 0 {
				pendingTools = extractFallbackTools(contentBlocks)
			}

			// Build assistant message
			assistantMsg := apitypes.InputMessage{Role: "assistant"}
			for _, block := range contentBlocks {
				switch block.Kind {
				case "text":
					assistantMsg.Content = append(assistantMsg.Content, apitypes.InputContentBlock{Kind: "text", Text: block.Text})
				case "tool_use":
					assistantMsg.Content = append(assistantMsg.Content, apitypes.InputContentBlock{
						Kind: "tool_use", ID: block.ID, Name: block.Name, Input: block.Input,
					})
				}
			}

			// For fallback tools, ensure corresponding tool_use blocks exist in assistantMsg
			for _, tu := range pendingTools {
				hasUse := false
				for _, b := range assistantMsg.Content {
					if b.Kind == "tool_use" && b.ID == tu.id {
						hasUse = true
						break
					}
				}
				if !hasUse {
					assistantMsg.Content = append(assistantMsg.Content, apitypes.InputContentBlock{
						Kind: "tool_use", ID: tu.id, Name: tu.name, Input: tu.input,
					})
				}
			}
			r.session = append(r.session, assistantMsg)

			if len(pendingTools) == 0 {
				return
			}

			// Execute all tools and batch results into a single user message
			var toolResultBlocks []apitypes.InputContentBlock
			for _, tu := range pendingTools {
				result := r.executeTool(tu)
				toolResultBlocks = append(toolResultBlocks, apitypes.InputContentBlock{
					Kind:      "tool_result",
					ToolUseID: tu.id,
					Content:   result.Output,
					IsError:   result.IsError,
				})
			}
			r.session = append(r.session, apitypes.InputMessage{
				Role:    "user",
				Content: toolResultBlocks,
			})
		}
	}()
	return outCh, nil
}

var (
	xmlToolCallRegex = regexp.MustCompile("(?s)<tool_call>(?:```(?:json)?)?\\s*(\\{.*?\\})\\s*(?:```)?</tool_call>")
	fencedCodeRegex  = regexp.MustCompile("(?s)```(?:json)?\\s*([\\s\\S]*?)\\s*```")
	jsonNameFirstRe  = regexp.MustCompile(`(?s)\{\s*"(?:name|tool|tool_name)"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters|input)"\s*:\s*(\{.*?\})\s*\}`)
	jsonArgsFirstRe  = regexp.MustCompile(`(?s)\{\s*"(?:arguments|parameters|input)"\s*:\s*(\{.*?\})\s*,\s*"(?:name|tool|tool_name)"\s*:\s*"([^"]+)"\s*\}`)
)

func parseToolMap(obj map[string]interface{}, id string) *toolUseInfo {
	name, _ := obj["name"].(string)
	if name == "" {
		name, _ = obj["tool"].(string)
	}
	if name == "" {
		name, _ = obj["tool_name"].(string)
	}
	if name == "" {
		return nil
	}
	var args json.RawMessage
	if a, ok := obj["arguments"]; ok {
		args, _ = json.Marshal(a)
	} else if p, ok := obj["parameters"]; ok {
		args, _ = json.Marshal(p)
	} else if in, ok := obj["input"]; ok {
		args, _ = json.Marshal(in)
	} else {
		args = json.RawMessage("{}")
	}
	return &toolUseInfo{
		id:    id,
		name:  name,
		input: args,
	}
}

func extractFallbackTools(contentBlocks []apitypes.OutputContentBlock) []toolUseInfo {
	var tools []toolUseInfo
	for _, block := range contentBlocks {
		// Accept "text" or uninitialized "" (from streaming text deltas) with content
		if (block.Kind != "text" && block.Kind != "") || strings.TrimSpace(block.Text) == "" {
			continue
		}

		text := strings.TrimSpace(block.Text)

		// 1. Check direct JSON object or array
		var directObj map[string]interface{}
		if err := json.Unmarshal([]byte(text), &directObj); err == nil {
			if t := parseToolMap(directObj, fmt.Sprintf("call_%d", len(tools)+1)); t != nil {
				tools = append(tools, *t)
				continue
			}
		}
		var directArr []map[string]interface{}
		if err := json.Unmarshal([]byte(text), &directArr); err == nil {
			for _, obj := range directArr {
				if t := parseToolMap(obj, fmt.Sprintf("call_%d", len(tools)+1)); t != nil {
					tools = append(tools, *t)
				}
			}
			if len(tools) > 0 {
				continue
			}
		}

		// 2. Fenced code blocks ```json {...} ``` anywhere in text
		fencedMatches := fencedCodeRegex.FindAllStringSubmatch(text, -1)
		for _, fm := range fencedMatches {
			if len(fm) >= 2 {
				fencedBody := strings.TrimSpace(fm[1])
				var fObj map[string]interface{}
				if err := json.Unmarshal([]byte(fencedBody), &fObj); err == nil {
					if t := parseToolMap(fObj, fmt.Sprintf("call_%d", len(tools)+1)); t != nil {
						tools = append(tools, *t)
					}
				} else {
					var fArr []map[string]interface{}
					if err := json.Unmarshal([]byte(fencedBody), &fArr); err == nil {
						for _, obj := range fArr {
							if t := parseToolMap(obj, fmt.Sprintf("call_%d", len(tools)+1)); t != nil {
								tools = append(tools, *t)
							}
						}
					}
				}
			}
		}
		if len(tools) > 0 {
			continue
		}

		// 3. XML tags: <tool_call>...</tool_call>
		xmlMatches := xmlToolCallRegex.FindAllStringSubmatch(text, -1)
		for _, m := range xmlMatches {
			if len(m) >= 2 {
				var parsed map[string]interface{}
				if err := json.Unmarshal([]byte(m[1]), &parsed); err == nil {
					if t := parseToolMap(parsed, fmt.Sprintf("call_%d", len(tools)+1)); t != nil {
						tools = append(tools, *t)
					}
				}
			}
		}
		if len(tools) > 0 {
			continue
		}

		// 4. Embedded regex patterns
		// Regex 1: name first
		for _, m := range jsonNameFirstRe.FindAllStringSubmatch(text, -1) {
			if len(m) >= 3 {
				tools = append(tools, toolUseInfo{
					id:    fmt.Sprintf("call_%d", len(tools)+1),
					name:  m[1],
					input: json.RawMessage(m[2]),
				})
			}
		}
		// Regex 2: arguments first (m[1] is args, m[2] is name)
		if len(tools) == 0 {
			for _, m := range jsonArgsFirstRe.FindAllStringSubmatch(text, -1) {
				if len(m) >= 3 {
					tools = append(tools, toolUseInfo{
						id:    fmt.Sprintf("call_%d", len(tools)+1),
						name:  m[2],
						input: json.RawMessage(m[1]),
					})
				}
			}
		}
	}
	return tools
}

type toolUseInfo struct {
	id    string
	name  string
	input json.RawMessage
}

func (r *ConversationRuntime) executeTool(tu toolUseInfo) apitypes.ToolResult {
	// Parse input
	var inputMap map[string]interface{}
	if len(tu.input) > 0 {
		_ = json.Unmarshal(tu.input, &inputMap)
	}
	if inputMap == nil {
		inputMap = make(map[string]interface{})
	}
	inputStr, _ := json.Marshal(inputMap)

	// Notify UI that a tool is about to run (stops spinner, shows tool name)
	r.toolCb.OnToolStart(tu.name, inputMap)

	// Check permissions
	allowed, reason := r.permPolicy.Authorize(tu.name, string(inputStr))
	if !allowed {
		if dtc, ok := r.toolCb.(DetailedToolCallback); ok {
			dtc.OnToolEndWithResult(tu.name, false, reason)
		} else {
			r.toolCb.OnToolEnd(tu.name, false)
		}
		return apitypes.ToolResult{ToolUseID: tu.id, Output: reason, IsError: true}
	}

	// Pre-tool hook
	preResult := r.hooks.PreToolUse(tu.name, inputMap)
	if preResult.IsDenied() {
		if dtc, ok := r.toolCb.(DetailedToolCallback); ok {
			dtc.OnToolEndWithResult(tu.name, false, "denied by pre-tool hook")
		} else {
			r.toolCb.OnToolEnd(tu.name, false)
		}
		return ToolResultFromHookDenial(tu.id, tu.name, preResult)
	}

	// Hook requested user confirmation
	if preResult.Escalate {
		escalateInput, _ := json.Marshal(inputMap)
		allowed, reason := r.permPolicy.Authorize(tu.name, string(escalateInput))
		if !allowed {
			if dtc, ok := r.toolCb.(DetailedToolCallback); ok {
				dtc.OnToolEndWithResult(tu.name, false, reason)
			} else {
				r.toolCb.OnToolEnd(tu.name, false)
			}
			return apitypes.ToolResult{ToolUseID: tu.id, Output: reason, IsError: true}
		}
	}

	// Apply updated input from hook
	if preResult.UpdatedInput != nil {
		inputMap = preResult.UpdatedInput
	}

	// Execute
	result := r.executor.Execute(tu.name, inputMap)
	if dtc, ok := r.toolCb.(DetailedToolCallback); ok {
		dtc.OnToolEndWithResult(tu.name, !result.IsError, result.Output)
	} else {
		r.toolCb.OnToolEnd(tu.name, !result.IsError)
	}
	result.ToolUseID = tu.id
	result.Output = MergeHookFeedback(preResult.Messages, result.Output, false)

	// Post-tool hook
	postResult := r.hooks.PostToolUse(tu.name, inputMap, result.Output, result.IsError)
	if postResult.IsDenied() {
		result.IsError = true
	}
	result.Output = MergeHookFeedback(postResult.Messages, result.Output, postResult.IsDenied())

	return result
}

func (r *ConversationRuntime) buildRequest() apitypes.MessageRequest {
	return apitypes.MessageRequest{
		Model:     r.model,
		MaxTokens: r.maxTokens,
		Messages:  r.session,
		System:    r.systemPrompt,
		Tools:     r.executor.ListTools(),
		Stream:    false,
	}
}

// CompactSession keeps only the last N messages.
func (r *ConversationRuntime) CompactSession(preserveRecent int) {
	if preserveRecent >= len(r.session) {
		return
	}
	r.session = r.session[len(r.session)-preserveRecent:]
}

// EstimateSessionTokens returns a rough token count for the current session.
// Uses the heuristic: 1 token ≈ 4 characters.
func (r *ConversationRuntime) EstimateSessionTokens() int {
	totalChars := len(r.systemPrompt)
	for _, msg := range r.session {
		for _, block := range msg.Content {
			totalChars += len(block.Text)
			totalChars += len(block.Input)
			totalChars += len(block.Name) + len(block.ID)
		}
	}
	return totalChars / 4
}

// SummarizeAndReset compacts the session by asking the LLM to summarize,
// then starts fresh with just the summary as context.
func (r *ConversationRuntime) SummarizeAndReset(ctx context.Context) (string, error) {
	if len(r.session) < 4 {
		return "", nil
	}

	summaryPrompt := "Summarize the entire conversation so far in a concise paragraph. Include: what the user asked, what tools were used, what was accomplished, and any pending tasks. This summary will be used as context for a new session."

	r.session = append(r.session, apitypes.InputMessage{
		Role:    "user",
		Content: []apitypes.InputContentBlock{{Kind: "text", Text: summaryPrompt}},
	})

	resp, err := r.provider.SendMessage(ctx, r.buildRequest())
	if err != nil {
		// If even the summary request fails, do a hard reset
		r.session = nil
		return "Previous session was too large to summarize. Starting fresh.", nil
	}

	var summary string
	for _, block := range resp.Content {
		if block.Kind == "text" {
			summary += block.Text
		}
	}

	// Reset session with just the summary
	r.session = []apitypes.InputMessage{
		{
			Role: "user",
			Content: []apitypes.InputContentBlock{{Kind: "text",
				Text: "Here is a summary of our previous conversation:\n\n" + summary + "\n\nPlease continue from where we left off."}},
		},
		{
			Role: "assistant",
			Content: []apitypes.InputContentBlock{{Kind: "text",
				Text: "Got it, I have the context from our previous conversation. How can I help you next?"}},
		},
	}

	return summary, nil
}

// GetUsage returns the cumulative usage tracker.
func (r *ConversationRuntime) GetUsage() UsageTracker { return r.usage }

// GetToolCb returns the tool callback for external wiring (e.g., spinner integration).
func (r *ConversationRuntime) GetToolCb() ToolCallback { return r.toolCb }

// SetToolCb replaces the tool callback (e.g., for structured output collection).
func (r *ConversationRuntime) SetToolCb(cb ToolCallback) { r.toolCb = cb }

// GetSession returns the current conversation session.
func (r *ConversationRuntime) GetSession() []apitypes.InputMessage { return r.session }

// GetModel returns the model name configured for this runtime.
func (r *ConversationRuntime) GetModel() string { return r.model }

// GetProvider returns the LLM provider configured for this runtime.
func (r *ConversationRuntime) GetProvider() apiclient.Provider { return r.provider }

// RestoreSession replaces the current session with a saved one.
func (r *ConversationRuntime) RestoreSession(messages []apitypes.InputMessage) {
	r.session = messages
}

func appendJSON(existing json.RawMessage, partial string) json.RawMessage {
	if len(existing) == 0 || string(existing) == "{}" {
		return json.RawMessage(partial)
	}
	return json.RawMessage(string(existing) + partial)
}
