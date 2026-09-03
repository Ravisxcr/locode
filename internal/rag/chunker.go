package rag

import (
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/tmc/langchaingo/textsplitter"
)

// Language-specific separator hierarchies for AST/boundary-aware splitting.
var languageSeparators = map[string][]string{
	"go": {
		"\nfunc ",
		"\ntype ",
		"\nvar ",
		"\nconst ",
		"\n// ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"python": {
		"\ndef ",
		"\nclass ",
		"\nasync def ",
		"\n# ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"typescript": {
		"\nexport function ",
		"\nexport class ",
		"\nexport interface ",
		"\nexport type ",
		"\nexport const ",
		"\nfunction ",
		"\nclass ",
		"\ninterface ",
		"\ntype ",
		"\n// ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"javascript": {
		"\nexport function ",
		"\nexport class ",
		"\nexport const ",
		"\nfunction ",
		"\nclass ",
		"\nconst ",
		"\nlet ",
		"\nvar ",
		"\n// ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"rust": {
		"\npub fn ",
		"\nfn ",
		"\npub struct ",
		"\nstruct ",
		"\npub enum ",
		"\nenum ",
		"\nimpl ",
		"\npub trait ",
		"\ntrait ",
		"\n// ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"java": {
		"\npublic class ",
		"\nclass ",
		"\npublic interface ",
		"\ninterface ",
		"\npublic ",
		"\nprivate ",
		"\nprotected ",
		"\n// ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"cpp": {
		"\nclass ",
		"\nstruct ",
		"\nnamespace ",
		"\n// ",
		"\n\n",
		"\n",
		" ",
		"",
	},
	"c": {
		"\nstruct ",
		"\ntypedef ",
		"\n// ",
		"\n/*",
		"\n\n",
		"\n",
		" ",
		"",
	},
}

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

// ChunkFile splits a file content into a slice of CodeChunk using langchaingo's textsplitter.
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

	// If entire file fits in one chunk, return it as a single chunk
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

	// Approximate character limits based on token limits (~4 chars per token)
	chunkCharSize := c.opts.MaxChunkTokens * 4
	overlapCharSize := c.opts.OverlapLines * 40

	var splitter textsplitter.TextSplitter
	if lang == "markdown" {
		splitter = textsplitter.NewMarkdownTextSplitter(
			textsplitter.WithChunkSize(chunkCharSize),
			textsplitter.WithChunkOverlap(overlapCharSize),
		)
	} else if seps, ok := languageSeparators[lang]; ok {
		splitter = textsplitter.NewRecursiveCharacter(
			textsplitter.WithSeparators(seps),
			textsplitter.WithChunkSize(chunkCharSize),
			textsplitter.WithChunkOverlap(overlapCharSize),
			textsplitter.WithKeepSeparator(true),
		)
	} else {
		splitter = textsplitter.NewRecursiveCharacter(
			textsplitter.WithChunkSize(chunkCharSize),
			textsplitter.WithChunkOverlap(overlapCharSize),
			textsplitter.WithKeepSeparator(true),
		)
	}

	pieces, err := splitter.SplitText(content)
	if err != nil || len(pieces) == 0 {
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

	var chunks []CodeChunk
	searchPos := 0

	for _, piece := range pieces {
		pieceTrimmed := strings.TrimSpace(piece)
		if pieceTrimmed == "" {
			continue
		}
		if EstimateTokens(pieceTrimmed) < c.opts.MinChunkTokens && len(pieces) > 1 && len(chunks) > 0 {
			continue
		}

		// Locate the piece in content to determine exact line numbers
		idx := strings.Index(content[searchPos:], piece)
		var startLine int
		if idx != -1 {
			actualPos := searchPos + idx
			startLine = strings.Count(content[:actualPos], "\n") + 1
			searchPos = actualPos + len(piece)/2
		} else {
			if len(chunks) > 0 {
				startLine = chunks[len(chunks)-1].EndLine + 1
			} else {
				startLine = 1
			}
		}

		lineCount := strings.Count(piece, "\n")
		endLine := startLine + lineCount
		if endLine < startLine {
			endLine = startLine
		}

		checksum := fmt.Sprintf("%x", sha256.Sum256([]byte(piece)))
		chunks = append(chunks, CodeChunk{
			ID:         fmt.Sprintf("%s#L%d-L%d", filePath, startLine, endLine),
			FilePath:   filePath,
			Language:   lang,
			Content:    piece,
			StartLine:  startLine,
			EndLine:    endLine,
			TokenCount: EstimateTokens(piece),
			ChunkIndex: len(chunks),
			Checksum:   checksum,
		})
	}

	return chunks
}
