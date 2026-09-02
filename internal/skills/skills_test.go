package skills

import (
	"sort"
	"testing"
)

func TestDefaultBuiltinsCount(t *testing.T) {
	builtins := defaultBuiltins()
	if got := len(builtins); got != 17 {
		t.Fatalf("expected 17 built-in skills, got %d", got)
	}

	ragSkill, ok := builtins["rag-code-architect"]
	if !ok {
		t.Fatalf("expected rag-code-architect skill to exist")
	}
	if ragSkill.SystemPrompt == "" {
		t.Fatalf("expected non-empty SystemPrompt for rag-code-architect")
	}
}

func TestWave2SkillsExistWithNonEmptyPrompts(t *testing.T) {
	wave2 := []string{"loop", "stuck", "debug", "verify", "simplify", "remember", "skillify", "batch", "rag-code-architect"}
	builtins := defaultBuiltins()

	for _, name := range wave2 {
		t.Run(name, func(t *testing.T) {
			skill, ok := builtins[name]
			if !ok {
				t.Fatalf("skill %q not found in defaultBuiltins", name)
			}
			if skill.SystemPrompt == "" {
				t.Fatalf("skill %q has empty SystemPrompt", name)
			}
			if len(skill.ToolPerms) == 0 {
				t.Fatalf("skill %q has no tool permissions", name)
			}
		})
	}
}

func TestSkillLoaderLoadsAllBuiltins(t *testing.T) {
	// Use a non-existent directory so only built-ins are loaded.
	loader := NewSkillLoader(t.TempDir())
	skills, errs := loader.LoadAll()
	if len(errs) > 0 {
		t.Fatalf("unexpected errors loading skills: %v", errs)
	}
	if got := len(skills); got != 17 {
		names := make([]string, len(skills))
		for i, s := range skills {
			names[i] = s.Name
		}
		sort.Strings(names)
		t.Fatalf("expected 17 skills from LoadAll, got %d: %v", got, names)
	}
}

func TestAllBuiltinSkillsPassValidation(t *testing.T) {
	builtins := defaultBuiltins()
	for name, skill := range builtins {
		t.Run(name, func(t *testing.T) {
			if err := Validate(skill); err != nil {
				t.Fatalf("built-in skill %q failed validation: %v", name, err)
			}
		})
	}
}
