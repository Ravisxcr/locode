package mcp

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/AlleyBo55/gocode/data"
	"github.com/AlleyBo55/gocode/internal/commands"
	"github.com/AlleyBo55/gocode/internal/execution"
	"github.com/AlleyBo55/gocode/internal/runtime"
	"github.com/AlleyBo55/gocode/internal/session"
	"github.com/AlleyBo55/gocode/internal/toolimpl"
	"github.com/AlleyBo55/gocode/internal/tools"
)

func newTestServer(t *testing.T) *MCPServer {
	t.Helper()
	toolReg, err := tools.NewToolRegistry(data.ToolsJSON)
	if err != nil {
		t.Fatalf("failed to create tool registry: %v", err)
	}
	cmdReg, err := commands.NewCommandRegistry(data.CommandsJSON)
	if err != nil {
		t.Fatalf("failed to create command registry: %v", err)
	}
	toolImpl := toolimpl.NewRegistry()
	sessionStore := session.NewSessionStore("")
	execReg := execution.BuildExecutionRegistry(cmdReg, toolReg)
	rt := runtime.NewPortRuntime(cmdReg, toolReg, execReg, sessionStore)
	return NewMCPServer(toolReg, toolImpl, cmdReg, rt, sessionStore, "test")
}

func TestInitializeHandshake(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "test-client",
				"version": "1.0.0",
			},
		},
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}

	resultJSON, _ := json.Marshal(resp.Result)
	var result InitializeResult
	if err := json.Unmarshal(resultJSON, &result); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if result.ProtocolVersion != "2024-11-05" {
		t.Errorf("expected protocol version 2024-11-05, got %s", result.ProtocolVersion)
	}
	if result.ServerInfo.Name != "gocode" {
		t.Errorf("expected server name gocode, got %s", result.ServerInfo.Name)
	}
	if result.Capabilities.Tools == nil {
		t.Error("expected tools capability")
	}
}

func TestNotificationsInitialized(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}

	resp := s.HandleRequest(req)
	if resp != nil {
		t.Errorf("expected nil response for notification, got %+v", resp)
	}
}

func TestPing(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "ping",
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}
}

func TestToolsList(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/list",
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}

	// Verify the shape: result should have a "tools" array of objects with name, description, inputSchema
	resultJSON, _ := json.Marshal(resp.Result)
	var resultMap map[string]interface{}
	json.Unmarshal(resultJSON, &resultMap)

	toolsRaw, ok := resultMap["tools"]
	if !ok {
		t.Fatal("result missing 'tools' key")
	}

	toolsArr, ok := toolsRaw.([]interface{})
	if !ok {
		t.Fatal("'tools' is not an array")
	}

	if len(toolsArr) == 0 {
		t.Fatal("tools list is empty")
	}

	// Check first tool has required fields
	firstTool, ok := toolsArr[0].(map[string]interface{})
	if !ok {
		t.Fatal("tool entry is not an object")
	}

	for _, field := range []string{"name", "description", "inputSchema"} {
		if _, ok := firstTool[field]; !ok {
			t.Errorf("tool missing required field: %s", field)
		}
	}

	// Check inputSchema has "type"
	schema, ok := firstTool["inputSchema"].(map[string]interface{})
	if !ok {
		t.Fatal("inputSchema is not an object")
	}
	if schema["type"] != "object" {
		t.Errorf("inputSchema type should be 'object', got %v", schema["type"])
	}
}

func TestToolsCallBashTool(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name": "BashTool",
			"arguments": map[string]interface{}{
				"command": "echo hello-gocode",
			},
		},
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}

	resultJSON, _ := json.Marshal(resp.Result)
	var resultMap map[string]interface{}
	json.Unmarshal(resultJSON, &resultMap)

	// Verify content shape
	contentRaw, ok := resultMap["content"]
	if !ok {
		t.Fatal("result missing 'content' key")
	}

	contentArr, ok := contentRaw.([]interface{})
	if !ok {
		t.Fatal("'content' is not an array")
	}

	if len(contentArr) == 0 {
		t.Fatal("content array is empty")
	}

	firstBlock, ok := contentArr[0].(map[string]interface{})
	if !ok {
		t.Fatal("content block is not an object")
	}

	if firstBlock["type"] != "text" {
		t.Errorf("content type should be 'text', got %v", firstBlock["type"])
	}

	text, ok := firstBlock["text"].(string)
	if !ok {
		t.Fatal("text is not a string")
	}

	if !strings.Contains(text, "hello-gocode") {
		t.Errorf("expected output to contain 'hello-gocode', got: %s", text)
	}
}

func TestToolsCallFileReadTool(t *testing.T) {
	s := newTestServer(t)

	// Read a known file in the project (tests run from internal/mcp/)
	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name": "FileReadTool",
			"arguments": map[string]interface{}{
				"path":       "../../go.mod",
				"start_line": 1,
				"end_line":   3,
			},
		},
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}

	resultJSON, _ := json.Marshal(resp.Result)
	var resultMap map[string]interface{}
	json.Unmarshal(resultJSON, &resultMap)

	contentArr := resultMap["content"].([]interface{})
	text := contentArr[0].(map[string]interface{})["text"].(string)

	if !strings.Contains(text, "module") {
		t.Errorf("expected go.mod content, got: %s", text)
	}
}

func TestToolsCallListDirectoryTool(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name": "ListDirectoryTool",
			"arguments": map[string]interface{}{
				"path": ".",
			},
		},
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}

	resultJSON, _ := json.Marshal(resp.Result)
	var resultMap map[string]interface{}
	json.Unmarshal(resultJSON, &resultMap)

	contentArr := resultMap["content"].([]interface{})
	text := contentArr[0].(map[string]interface{})["text"].(string)

	if text == "" {
		t.Error("expected non-empty directory listing")
	}
}

func TestToolsCallNotFound(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name": "NonExistentTool",
			"arguments": map[string]interface{}{},
		},
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error == nil {
		t.Fatal("expected error for non-existent tool")
	}
	if resp.Error.Code != -32602 {
		t.Errorf("expected error code -32602, got %d", resp.Error.Code)
	}
}

func TestMethodNotFound(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "unknown/method",
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error == nil {
		t.Fatal("expected error for unknown method")
	}
	if resp.Error.Code != -32601 {
		t.Errorf("expected error code -32601, got %d", resp.Error.Code)
	}
}

func TestResourcesList(t *testing.T) {
	s := newTestServer(t)

	req := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "resources/list",
	}

	resp := s.HandleRequest(req)
	if resp == nil {
		t.Fatal("expected response, got nil")
	}
	if resp.Error != nil {
		t.Fatalf("unexpected error: %v", resp.Error.Message)
	}
}

func TestStdioRoundTrip(t *testing.T) {
	s := newTestServer(t)

	// Build a multi-line stdio session
	var input strings.Builder

	// 1. initialize
	initReq, _ := json.Marshal(MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "test",
				"version": "1.0",
			},
		},
	})
	input.Write(initReq)
	input.WriteString("\n")

	// 2. notifications/initialized (no response expected)
	initedReq, _ := json.Marshal(MCPRequest{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	})
	input.Write(initedReq)
	input.WriteString("\n")

	// 3. tools/list
	listReq, _ := json.Marshal(MCPRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/list",
	})
	input.Write(listReq)
	input.WriteString("\n")

	// 4. ping
	pingReq, _ := json.Marshal(MCPRequest{
		JSONRPC: "2.0",
		ID:      3,
		Method:  "ping",
	})
	input.Write(pingReq)
	input.WriteString("\n")

	var output bytes.Buffer
	err := s.ServeStdioWithReaderWriter(strings.NewReader(input.String()), &output)
	if err != nil {
		t.Fatalf("ServeStdio error: %v", err)
	}

	// Parse output lines — should get 3 responses (initialize, tools/list, ping)
	// notifications/initialized should NOT produce a response
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 response lines, got %d:\n%s", len(lines), output.String())
	}

	// Verify each response has the correct ID
	expectedIDs := []float64{1, 2, 3}
	for i, line := range lines {
		var resp MCPResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			t.Fatalf("line %d: failed to parse response: %v", i, err)
		}
		id, ok := resp.ID.(float64)
		if !ok {
			t.Fatalf("line %d: response ID is not a number: %v", i, resp.ID)
		}
		if id != expectedIDs[i] {
			t.Errorf("line %d: expected ID %v, got %v", i, expectedIDs[i], id)
		}
		if resp.Error != nil {
			t.Errorf("line %d: unexpected error: %s", i, resp.Error.Message)
		}
	}
}
