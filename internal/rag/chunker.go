package rag

import (
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// CodeChunk represents a chunk of indexed code with location metadata.
type CodeChunk struct {
	ID         string `json:"id"`
	FilePath   string `json:"file_path"`
	Language   string `json:"language"`
	Content    string `json:"content"`
	StartLine  int    `json:"start_line"`
	EndLine    int    `json:"end_line"`
	TokenCount int    `json:"token_count"`
	ChunkIndex int    `json:"chunk_index"`
	Checksum   string `json:"checksum"`
}

// ChunkOptions configures code splitting parameters.
type ChunkOptions struct {
	MaxChunkTokens int // target max tokens per chunk (default: 350)
	OverlapLines   int // overlap lines between adjacent chunks (default: 3)
	MinChunkTokens int // minimum tokens to avoid tiny chunks (default: 20)
}

// DefaultChunkOptions returns standard chunking options.
func DefaultChunkOptions() ChunkOptions {
	return ChunkOptions{
		MaxChunkTokens: 350,
		OverlapLines:   3,
		MinChunkTokens: 20,
	}
}

// CodeChunker splits source files into semantically meaningful chunks.
type CodeChunker struct {
	opts ChunkOptions
}

// NewCodeChunker creates a new CodeChunker.
func NewCodeChunker(opts ChunkOptions) *CodeChunker {
	if opts.MaxChunkTokens <= 0 {
		opts.MaxChunkTokens = 350
	}
	if opts.OverlapLines < 0 {
		opts.OverlapLines = 3
	}
	if opts.MinChunkTokens <= 0 {
		opts.MinChunkTokens = 20
	}
	return &CodeChunker{opts: opts}
}

// DetectLanguage returns the programming language name based on file extension.
func DetectLanguage(filePath string) string {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".go":
		return "go"
	case ".ts", ".tsx":
		return "typescript"
	case ".js", ".jsx", ".mjs", ".cjs":
		return "javascript"
	case ".py":
		return "python"
	case ".rs":
		return "rust"
	case ".java":
		return "java"
	case ".c", ".h":
		return "c"
	case ".cpp", ".cc", ".cxx", ".hpp":
		return "cpp"
	case ".cs":
		return "csharp"
	case ".rb":
		return "ruby"
	case ".php":
		return "php"
	case ".swift":
		return "swift"
	case ".kt", ".kts":
		return "kotlin"
	case ".sql":
		return "sql"
	case ".sh", ".bash", ".zsh":
		return "bash"
	case ".md", ".markdown":
		return "markdown"
	case ".json":
		return "json"
	case ".yaml", ".yml":
		return "yaml"
	case ".toml":
		return "toml"
	case ".html", ".htm":
		return "html"
	case ".css", ".scss", ".sass":
		return "css"
	default:
		return "text"
	}
}

// EstimateTokens provides a fast approximation of token count (~4 chars per token).
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	words := len(strings.Fields(text))
	chars := len(text) / 4
	if words > chars {
		return words
	}
	return chars
}

// topLevelDeclRegex matches top-level functions, methods, classes, and types in common languages.
var topLevelDeclRegex = regexp.MustCompile(`(?m)^(?:func\s+|type\s+|def\s+|class\s+|export\s+class\s+|export\s+function\s+|export\s+interface\s+|interface\s+|pub\s+fn\s+|fn\s+|async\s+def\s+|public\s+class\s+|struct\s+|enum\s+|#+\s+)`)

// ChunkFile splits a file content into a slice of CodeChunk.
func (c *CodeChunker) ChunkFile(filePath, content string) []CodeChunk {
	trimmed := strings.TrimRight(content, "\r\n")
	if strings.TrimSpace(trimmed) == "" {
		return nil
	}

	lines := strings.Split(trimmed, "\n")
	if len(lines) == 0 {
		return nil
	}

	lang := DetectLanguage(filePath)
	totalTokens := EstimateTokens(content)

	// If entire file fits in one chunk, return it as single chunk
	if totalTokens <= c.opts.MaxChunkTokens {
		checksum := fmt.Sprintf("%x", sha256.Sum256([]byte(content)))
		return []CodeChunk{
			{
				ID:         fmt.Sprintf("%s#L1-L%d", filePath, len(lines)),
				FilePath:   filePath,
				Language:   lang,
				Content:    content,
				StartLine:  1,
				EndLine:    len(lines),
				TokenCount: totalTokens,
				ChunkIndex: 0,
				Checksum:   checksum,
			},
		}
	}

	// Semantic chunking: find logical block boundaries (function declarations, headers, blank line separators)
	var boundaryLines []int
	boundaryLines = append(boundaryLines, 0) // start of file

	for i, line := range lines {
		if i == 0 {
			continue
		}
		if topLevelDeclRegex.MatchString(line) {
			boundaryLines = append(boundaryLines, i)
		}
	}
	boundaryLines = append(boundaryLines, len(lines))

	var chunks []CodeChunk
	currentStartLine := 0
	var currentChunkLines []string

	for bIdx := 0; bIdx < len(boundaryLines)-1; bIdx++ {
		start := boundaryLines[bIdx]
		end := boundaryLines[bIdx+1]

		sectionLines := lines[start:end]
		sectionTokens := EstimateTokens(strings.Join(sectionLines, "\n"))

		// If this section alone is larger than MaxChunkTokens, split line-by-line with overlap
		if sectionTokens > c.opts.MaxChunkTokens {
			// Flush current accumulator first if non-empty
			if len(currentChunkLines) > 0 {
				chunks = append(chunks, c.makeChunk(filePath, lang, currentStartLine+1, currentChunkLines, len(chunks)))
				currentChunkLines = nil
			}

			// Sub-split large section
			subChunks := c.splitLinesWithOverlap(filePath, lang, start+1, sectionLines, len(chunks))
			chunks = append(chunks, subChunks...)
			currentStartLine = end
			continue
		}

		// Check if accumulating this section exceeds limit
		accTokens := EstimateTokens(strings.Join(currentChunkLines, "\n")) + sectionTokens
		if accTokens > c.opts.MaxChunkTokens && len(currentChunkLines) > 0 {
			chunks = append(chunks, c.makeChunk(filePath, lang, currentStartLine+1, currentChunkLines, len(chunks)))

			// Start new accumulator with overlap from previous chunk
			overlapStart := max(0, len(currentChunkLines)-c.opts.OverlapLines)
			overlap := currentChunkLines[overlapStart:]
			currentStartLine = start - len(overlap)
			currentChunkLines = append(append([]string{}, overlap...), sectionLines...)
		} else {
			if len(currentChunkLines) == 0 {
				currentStartLine = start
			}
			currentChunkLines = append(currentChunkLines, sectionLines...)
		}
	}

	// Flush remaining accumulated lines
	if len(currentChunkLines) > 0 {
		chunks = append(chunks, c.makeChunk(filePath, lang, currentStartLine+1, currentChunkLines, len(chunks)))
	}

	return chunks
}

func (c *CodeChunker) splitLinesWithOverlap(filePath, lang string, baseStartLine int, lines []string, startingIndex int) []CodeChunk {
	var chunks []CodeChunk
	var currentLines []string
	startLine := baseStartLine

	for i, line := range lines {
		currentLines = append(currentLines, line)
		tokens := EstimateTokens(strings.Join(currentLines, "\n"))

		if tokens >= c.opts.MaxChunkTokens {
			chunks = append(chunks, c.makeChunk(filePath, lang, startLine, currentLines, startingIndex+len(chunks)))

			// Keep overlap
			overlapCount := min(c.opts.OverlapLines, len(currentLines))
			overlap := currentLines[len(currentLines)-overlapCount:]
			startLine = baseStartLine + i - overlapCount + 1
			currentLines = append([]string{}, overlap...)
		}
	}

	if len(currentLines) > 0 {
		tokens := EstimateTokens(strings.Join(currentLines, "\n"))
		if tokens >= c.opts.MinChunkTokens || len(chunks) == 0 {
			chunks = append(chunks, c.makeChunk(filePath, lang, startLine, currentLines, startingIndex+len(chunks)))
		}
	}

	return chunks
}

func (c *CodeChunker) makeChunk(filePath, lang string, startLine int, lines []string, chunkIndex int) CodeChunk {
	content := strings.Join(lines, "\n")
	endLine := startLine + len(lines) - 1
	if endLine < startLine {
		endLine = startLine
	}
	checksum := fmt.Sprintf("%x", sha256.Sum256([]byte(content)))

	return CodeChunk{
		ID:         fmt.Sprintf("%s#L%d-L%d", filePath, startLine, endLine),
		FilePath:   filePath,
		Language:   lang,
		Content:    content,
		StartLine:  startLine,
		EndLine:    endLine,
		TokenCount: EstimateTokens(content),
		ChunkIndex: chunkIndex,
		Checksum:   checksum,
	}
}
