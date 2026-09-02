package gitcommit

import (
	"context"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/apiclient"
	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

func TestCleanCommitMessage(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "```\nfeat(agent): add loop circuit breaker\n```",
			expected: "feat(agent): add loop circuit breaker",
		},
		{
			input:    "```git\nfix(repl): handle slash commands\n\n- Details here\n```",
			expected: "fix(repl): handle slash commands\n\n- Details here",
		},
		{
			input:    "\"feat(cli): add commit command\"",
			expected: "feat(cli): add commit command",
		},
		{
			input:    "refactor(tools): normalize tool aliases",
			expected: "refactor(tools): normalize tool aliases",
		},
	}

	for _, tc := range tests {
		got := cleanCommitMessage(tc.input)
		if got != tc.expected {
			t.Errorf("cleanCommitMessage(%q) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}

func TestIsValidCommitHeader(t *testing.T) {
	valid := []string{
		"feat(agent): add loop detection",
		"fix: resolve nil pointer",
		"refactor(toolimpl): clean aliases",
		"docs: update readme",
		"test(gitcommit): add unit tests",
		"chore: update dependencies",
		"perf(rag): optimize vector search",
	}
	for _, s := range valid {
		if !isValidCommitHeader(s) {
			t.Errorf("expected valid header for %q", s)
		}
	}

	invalid := []string{
		"gocode: auto-commit",
		"Updated some files",
		"feat add something",
		"",
	}
	for _, s := range invalid {
		if isValidCommitHeader(s) {
			t.Errorf("expected invalid header for %q", s)
		}
	}
}

func TestHeuristicMessage(t *testing.T) {
	// Test docs detection
	docsStat := "docs/architecture.md | 10 ++\nREADME.md | 5 +"
	msg := HeuristicMessage(docsStat, "Update documentation details")
	if msg != "docs: update documentation and guides" {
		t.Errorf("unexpected docs message: %q", msg)
	}

	// Test test detection
	testStat := "internal/agent/runtime_test.go | 20 ++\ninternal/agent/agent_test.go | 15 +"
	msg = HeuristicMessage(testStat, "func TestSomething(t *testing.T)")
	if msg != "test(agent): add and update unit tests" {
		t.Errorf("unexpected test message: %q", msg)
	}

	// Test fix detection
	fixStat := "internal/agent/runtime.go | 12 +-"
	msg = HeuristicMessage(fixStat, "fix infinite loop and error handling")
	if msg != "fix(agent): resolve issues in agent component" {
		t.Errorf("unexpected fix message: %q", msg)
	}

	// Test feat detection
	featStat := "internal/repl/repl.go | 50 ++\ninternal/repl/input.go | 20 ++"
	msg = HeuristicMessage(featStat, "new file mode 100644\nadd commit command")
	if msg != "feat(repl): implement repl feature" {
		t.Errorf("unexpected feat message: %q", msg)
	}
}

type mockCommitProvider struct {
	response string
	err      error
}

func (m *mockCommitProvider) SendMessage(ctx context.Context, req apitypes.MessageRequest) (*apitypes.MessageResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &apitypes.MessageResponse{
		Role:    "assistant",
		Content: []apitypes.OutputContentBlock{{Kind: "text", Text: m.response}},
	}, nil
}

func (m *mockCommitProvider) StreamMessage(ctx context.Context, req apitypes.MessageRequest) (<-chan apitypes.StreamEvent, error) {
	ch := make(chan apitypes.StreamEvent, 1)
	close(ch)
	return ch, nil
}

func (m *mockCommitProvider) Kind() apiclient.ProviderKind {
	return apiclient.ProviderOpenAi
}

func TestGenerateMessage_WithProvider(t *testing.T) {
	prov := &mockCommitProvider{
		response: "feat(gitcommit): add intelligent commit message generator\n\n- Add LLM generator\n- Add fallback heuristics",
	}

	msg, err := GenerateMessage(context.Background(), prov, "test-model", "diff content", "stat content")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "feat(gitcommit): add intelligent commit message generator\n\n- Add LLM generator\n- Add fallback heuristics"
	if msg != expected {
		t.Errorf("got %q, want %q", msg, expected)
	}
}

func TestGenerateMessage_FallbackOnInvalidLLMOutput(t *testing.T) {
	prov := &mockCommitProvider{
		response: "Here is your commit message:\nI changed some files for you!",
	}

	// Should fall back to heuristic
	msg, err := GenerateMessage(context.Background(), prov, "test-model", "fix some bug", "internal/agent/runtime.go | 5 +-")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if msg != "fix(agent): resolve issues in agent component" {
		t.Errorf("expected heuristic fallback, got %q", msg)
	}
}
