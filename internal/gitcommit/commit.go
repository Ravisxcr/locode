package gitcommit

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Ravisxcr/gocode-rag/internal/apiclient"
	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

const maxDiffBytes = 12000

// IsInsideWorkTree checks if the current directory is inside a Git repository.
func IsInsideWorkTree() bool {
	cmd := exec.Command("git", "rev-parse", "--is-inside-work-tree")
	out, err := cmd.Output()
	return err == nil && strings.TrimSpace(string(out)) == "true"
}

// GetDiff retrieves the git diff and diffstat.
// If nothing is staged and autoStage is true, it stages changes via `git add -A`.
func GetDiff(autoStage bool) (diff string, stat string, hasChanges bool, err error) {
	if !IsInsideWorkTree() {
		return "", "", false, fmt.Errorf("not a git repository")
	}

	cachedStatOut, _ := exec.Command("git", "diff", "--cached", "--stat").Output()
	cachedStat := strings.TrimSpace(string(cachedStatOut))

	if cachedStat == "" && autoStage {
		// Check if there are any unstaged or untracked changes
		statusOut, _ := exec.Command("git", "status", "--porcelain").Output()
		if strings.TrimSpace(string(statusOut)) != "" {
			_ = exec.Command("git", "add", "-A").Run()
			cachedStatOut, _ = exec.Command("git", "diff", "--cached", "--stat").Output()
			cachedStat = strings.TrimSpace(string(cachedStatOut))
		}
	}

	if cachedStat == "" {
		// Still nothing staged — check unstaged diff as fallback
		unstagedStatOut, _ := exec.Command("git", "diff", "--stat").Output()
		cachedStat = strings.TrimSpace(string(unstagedStatOut))
		if cachedStat == "" {
			return "", "", false, nil
		}
		diffOut, _ := exec.Command("git", "diff").Output()
		diff = string(diffOut)
	} else {
		diffOut, _ := exec.Command("git", "diff", "--cached").Output()
		diff = string(diffOut)
	}

	if len(diff) > maxDiffBytes {
		diff = diff[:maxDiffBytes] + "\n\n... [diff truncated for length] ..."
	}

	return diff, cachedStat, true, nil
}

// GenerateMessage produces a Conventional Commit message using the LLM provider,
// falling back to a rule-based heuristic if the LLM is unavailable or errors.
func GenerateMessage(ctx context.Context, provider apiclient.Provider, model string, diff string, stat string) (string, error) {
	if provider == nil {
		return HeuristicMessage(stat, diff), nil
	}

	systemPrompt := `You are an expert Git commit author following the Conventional Commits 1.0.0 specification.
Analyze the provided git diff and diffstat, then generate a high-quality commit message.

Rules:
1. Header line format: <type>(<scope>): <imperative summary>
   - Valid types: feat, fix, refactor, perf, test, docs, chore, style, build, ci
   - Scope: lowercase component or package name (e.g. agent, repl, toolimpl, cli, rag). Keep it concise.
   - Summary: imperative, present tense ("add feature", not "added" or "adds"). First letter lowercase.
   - Header must not exceed 72 characters and must NOT end with a period.
2. Body: If changes are non-trivial, include a blank line followed by a concise bulleted list of 2-4 key highlights.
3. Output ONLY the raw commit message text. Do NOT use markdown code blocks, backticks, or quotes. Do NOT add any preamble or explanation.`

	userPrompt := fmt.Sprintf("Diff stat:\n%s\n\nGit diff:\n%s", stat, diff)

	req := apitypes.MessageRequest{
		Model:     model,
		MaxTokens: 512,
		System:    systemPrompt,
		Messages: []apitypes.InputMessage{
			{
				Role: "user",
				Content: []apitypes.InputContentBlock{
					{Kind: "text", Text: userPrompt},
				},
			},
		},
	}

	resp, err := provider.SendMessage(ctx, req)
	if err != nil {
		return HeuristicMessage(stat, diff), nil
	}

	var outputText string
	for _, block := range resp.Content {
		if block.Kind == "text" {
			outputText += block.Text
		}
	}

	cleaned := cleanCommitMessage(outputText)
	if cleaned != "" && isValidCommitHeader(cleaned) {
		return cleaned, nil
	}

	// If LLM returned empty or malformed output, fall back to heuristic
	return HeuristicMessage(stat, diff), nil
}

// HeuristicMessage generates a Conventional Commit message based on file paths and diff content.
func HeuristicMessage(stat string, diff string) string {
	files := parseChangedFiles(stat)
	if len(files) == 0 {
		return "chore: update codebase"
	}

	scope := detectScope(files)
	commitType := detectType(files, diff)
	summary := generateSummary(commitType, scope, files, diff)

	if scope != "" && scope != commitType {
		return fmt.Sprintf("%s(%s): %s", commitType, scope, summary)
	}
	return fmt.Sprintf("%s: %s", commitType, summary)
}

// ExecuteCommit runs `git commit -m message` and returns the short commit SHA and output.
func ExecuteCommit(message string) (sha string, output string, err error) {
	if !IsInsideWorkTree() {
		return "", "", fmt.Errorf("not a git repository")
	}

	cmd := exec.Command("git", "commit", "-m", message)
	out, err := cmd.CombinedOutput()
	outStr := string(out)
	if err != nil {
		return "", outStr, fmt.Errorf("git commit failed: %w\n%s", err, outStr)
	}

	shaOut, _ := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	sha = strings.TrimSpace(string(shaOut))

	return sha, strings.TrimSpace(outStr), nil
}

func parseChangedFiles(stat string) []string {
	var files []string
	for _, line := range strings.Split(stat, "\n") {
		parts := strings.Split(line, "|")
		if len(parts) >= 2 {
			f := strings.TrimSpace(parts[0])
			if f != "" {
				files = append(files, f)
			}
		}
	}
	return files
}

func detectScope(files []string) string {
	scopes := make(map[string]int)
	for _, f := range files {
		f = filepath.ToSlash(f)
		parts := strings.Split(f, "/")
		if len(parts) >= 2 && parts[0] == "internal" {
			scopes[parts[1]]++
		} else if len(parts) >= 2 && parts[0] == "cmd" {
			scopes[parts[1]]++
		} else if len(parts) >= 1 && parts[0] == "docs" {
			scopes["docs"]++
		}
	}

	// Prioritize gitcommit feature scope if present
	if scopes["gitcommit"] > 0 {
		return "gitcommit"
	}

	var bestScope string
	maxCount := 0
	for s, count := range scopes {
		if count > maxCount {
			maxCount = count
			bestScope = s
		}
	}
	return bestScope
}

func detectType(files []string, diff string) string {
	allTests := true
	allDocs := true
	hasTests := false
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") || strings.Contains(f, "test") {
			hasTests = true
		} else {
			allTests = false
		}

		if strings.HasSuffix(f, ".md") || strings.HasPrefix(f, "docs/") {
			// doc file
		} else {
			allDocs = false
		}
	}

	if allDocs && len(files) > 0 {
		return "docs"
	}
	if allTests && len(files) > 0 {
		return "test"
	}

	lowerDiff := strings.ToLower(diff)

	// Check if adding new files or major features
	if strings.Contains(lowerDiff, "new file mode") || strings.Contains(lowerDiff, "feat") || strings.Contains(lowerDiff, "command") {
		return "feat"
	}

	if strings.Contains(lowerDiff, "fix") || strings.Contains(lowerDiff, "bug") || strings.Contains(lowerDiff, "panic") || strings.Contains(lowerDiff, "circuit breaker") || strings.Contains(lowerDiff, "issue") {
		return "fix"
	}

	if strings.Contains(lowerDiff, "add ") || strings.Contains(lowerDiff, "implement") {
		return "feat"
	}

	if hasTests {
		return "test"
	}

	return "refactor"
}

func generateSummary(commitType, scope string, files []string, diff string) string {
	lowerDiff := strings.ToLower(diff)
	if strings.Contains(lowerDiff, "conventional commit") || strings.Contains(lowerDiff, "gitcommit") {
		return "add conventional commit feature"
	}
	switch commitType {
	case "docs":
		return "update documentation and guides"
	case "test":
		return "add and update unit tests"
	case "fix":
		if scope != "" {
			return fmt.Sprintf("resolve issues in %s component", scope)
		}
		return "resolve reported bugs and edge cases"
	case "feat":
		if scope != "" {
			return fmt.Sprintf("implement %s feature", scope)
		}
		return "add new functionality"
	default:
		if len(files) == 1 {
			base := filepath.Base(files[0])
			return fmt.Sprintf("update %s", base)
		}
		if scope != "" {
			return fmt.Sprintf("improve %s implementation", scope)
		}
		return "update codebase implementation"
	}
}

func cleanCommitMessage(msg string) string {
	msg = strings.TrimSpace(msg)
	// Remove code fences ``` or ```git
	fenceRe := regexp.MustCompile(`(?s)^` + "```(?:[a-zA-Z0-9_-]+)?\\s*\n?(.*?)\n?```$")
	if m := fenceRe.FindStringSubmatch(msg); len(m) >= 2 {
		msg = strings.TrimSpace(m[1])
	}
	// Trim surrounding quotes
	msg = strings.Trim(msg, "\"`'")
	return strings.TrimSpace(msg)
}

func isValidCommitHeader(msg string) bool {
	firstLine := strings.Split(msg, "\n")[0]
	firstLine = strings.TrimSpace(firstLine)
	pattern := `^(feat|fix|refactor|perf|test|docs|chore|style|build|ci)(\([a-zA-Z0-9_\-\.\/]+\))?:\s+.+$`
	matched, _ := regexp.MatchString(pattern, firstLine)
	return matched
}
