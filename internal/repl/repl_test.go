package repl

import (
	"bytes"
	"strings"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/agent"
	"github.com/Ravisxcr/gocode-rag/internal/skills"
)

func TestREPL_SkillActivation(t *testing.T) {
	initialPrompt := "You are gocode."
	rt := agent.NewConversationRuntime(agent.RuntimeOptions{
		SystemPrompt: initialPrompt,
	})

	testSkills := []skills.Skill{
		{
			Name:         "loop",
			SystemPrompt: "You are in autonomous loop mode.",
		},
		{
			Name:         "remember",
			SystemPrompt: "Review user memory landscape.",
		},
	}

	var buf bytes.Buffer
	r := NewREPL(rt, &bytes.Buffer{}, &buf, REPLConfig{
		Model:    "test-model",
		MaxTurns: 10,
	}, testSkills)

	// 1. Activate skill via handleSkillCommand
	r.handleSkillCommand("/skill loop")

	out := buf.String()
	if !strings.Contains(out, "Skill") || !strings.Contains(out, "loop") || !strings.Contains(out, "activated.") {
		t.Errorf("expected activation output, got: %s", out)
	}

	// Verify runtime system prompt was updated
	activePrompt := rt.GetSystemPrompt()
	if !strings.Contains(activePrompt, "You are in autonomous loop mode.") {
		t.Errorf("expected runtime system prompt to contain skill prompt, got: %s", activePrompt)
	}

	// 2. Activate skill with leading slash stripped (/remember)
	r.handleSkillCommand("/skill /remember")
	activePrompt2 := rt.GetSystemPrompt()
	if !strings.Contains(activePrompt2, "Review user memory landscape.") {
		t.Errorf("expected runtime system prompt to contain remember skill, got: %s", activePrompt2)
	}
}
