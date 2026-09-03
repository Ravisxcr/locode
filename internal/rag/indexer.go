package rag

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// IndexOptions configures workspace indexing.
type IndexOptions struct {
	RootPath     string
	IndexPath    string // path to index file (default: .gocode/rag/index.json)
	ChunkOpts    ChunkOptions
	ForceReindex bool
	MaxFileSize  int64 // max file size to index (default: 512KB)
	ProgressCb   func(file string, current, total int)
}

// DefaultIndexOptions returns standard indexing options.
func DefaultIndexOptions(rootPath string) IndexOptions {
	if rootPath == "" {
		rootPath, _ = os.Getwd()
	}
	return IndexOptions{
		RootPath:    rootPath,
		IndexPath:   filepath.Join(rootPath, ".gocode", "rag", "index.json"),
		ChunkOpts:   DefaultChunkOptions(),
		MaxFileSize: 512 * 1024, // 512KB
	}
}

// IndexReport summarizes indexing execution results.
type IndexReport struct {
	TotalFilesScanned int           `json:"total_files_scanned"`
	FilesIndexed      int           `json:"files_indexed"`
	FilesSkipped      int           `json:"files_skipped"`
	TotalChunks       int           `json:"total_chunks"`
	Duration          time.Duration `json:"duration"`
	EmbedderModel     string        `json:"embedder_model"`
	IndexPath         string        `json:"index_path"`
}

// Indexer coordinates workspace traversal, chunking, embedding, and vector persistence.
type Indexer struct {
	chunker  *CodeChunker
	embedder Embedder
	store    *VectorStore
}

// NewIndexer creates a new Indexer.
func NewIndexer(chunker *CodeChunker, embedder Embedder, store *VectorStore) *Indexer {
	if chunker == nil {
		chunker = NewCodeChunker(DefaultChunkOptions())
	}
	if store == nil {
		store = NewVectorStore(embedder.ModelName(), embedder.Dimension())
	}
	return &Indexer{
		chunker:  chunker,
		embedder: embedder,
		store:    store,
	}
}

// GetStore returns the underlying vector store.
func (idx *Indexer) GetStore() *VectorStore {
	return idx.store
}

var defaultIgnoredDirs = map[string]bool{
	".git":         true,
	".gocode":      true,
	"node_modules": true,
	"vendor":       true,
	"bin":          true,
	"dist":         true,
	"build":        true,
	".idea":        true,
	".vscode":      true,
	"__pycache__":  true,
	".pytest_cache": true,
	".next":        true,
	"target":       true,
}

var ignoredFileExts = map[string]bool{
	".png":   true,
	".jpg":   true,
	".jpeg":  true,
	".gif":   true,
	".ico":   true,
	".svg":   true,
	".pdf":   true,
	".zip":   true,
	".tar":   true,
	".gz":    true,
	".exe":   true,
	".bin":   true,
	".dll":   true,
	".so":    true,
	".dylib": true,
	".wasm":  true,
	".pyc":   true,
	".class": true,
	".o":     true,
	".a":     true,
	".sum":   true,
}

// ScanFiles returns all indexable files in the workspace.
func ScanFiles(rootPath string, maxFileSize int64) ([]string, error) {
	if maxFileSize <= 0 {
		maxFileSize = 512 * 1024
	}

	var files []string
	err := filepath.WalkDir(rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}

		name := d.Name()
		if d.IsDir() {
			if defaultIgnoredDirs[name] || strings.HasPrefix(name, ".") && name != "." {
				return filepath.SkipDir
			}
			return nil
		}

		ext := strings.ToLower(filepath.Ext(name))
		if ignoredFileExts[ext] {
			return nil
		}

		info, err := d.Info()
		if err != nil || info.Size() > maxFileSize || info.Size() == 0 {
			return nil
		}

		relPath, err := filepath.Rel(rootPath, path)
		if err == nil {
			files = append(files, relPath)
		} else {
			files = append(files, path)
		}
		return nil
	})

	return files, err
}

// IndexWorkspace scans the workspace, embeds new or modified files, and persists the index.
func (idx *Indexer) IndexWorkspace(ctx context.Context, opts IndexOptions) (*IndexReport, error) {
	startTime := time.Now()

	if opts.IndexPath == "" {
		opts.IndexPath = filepath.Join(opts.RootPath, ".gocode", "rag", "index.json")
	}

	// Try loading existing index if present
	_ = idx.store.Load(opts.IndexPath)

	files, err := ScanFiles(opts.RootPath, opts.MaxFileSize)
	if err != nil {
		return nil, fmt.Errorf("scanning workspace: %w", err)
	}

	currentFileSet := make(map[string]struct{})
	for _, f := range files {
		currentFileSet[f] = struct{}{}
	}

	// Prune deleted files from index
	idx.store.mu.RLock()
	var toDelete []string
	for indexedPath := range idx.store.FileHashes {
		if _, exists := currentFileSet[indexedPath]; !exists {
			toDelete = append(toDelete, indexedPath)
		}
	}
	idx.store.mu.RUnlock()

	for _, delPath := range toDelete {
		idx.store.RemoveFileChunks(delPath)
	}

	filesIndexed := 0
	filesSkipped := 0

	for i, relPath := range files {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if opts.ProgressCb != nil {
			opts.ProgressCb(relPath, i+1, len(files))
		}

		fullPath := filepath.Join(opts.RootPath, relPath)
		data, err := os.ReadFile(fullPath)
		if err != nil {
			filesSkipped++
			continue
		}

		content := string(data)
		checksum := fmt.Sprintf("%x", sha256.Sum256(data))

		// Check if file unchanged
		if !opts.ForceReindex {
			if existingHash, ok := idx.store.GetFileHash(relPath); ok && existingHash == checksum {
				filesSkipped++
				continue
			}
		}

		// Re-index this file: remove old chunks first
		idx.store.RemoveFileChunks(relPath)

		chunks := idx.chunker.ChunkFile(relPath, content)
		if len(chunks) == 0 {
			idx.store.SetFileHash(relPath, checksum)
			filesSkipped++
			continue
		}

		// Extract chunk texts for embedding
		chunkTexts := make([]string, len(chunks))
		for cIdx, c := range chunks {
			chunkTexts[cIdx] = fmt.Sprintf("File: %s\n%s", c.FilePath, c.Content)
		}

		embs, err := idx.embedder.EmbedDocuments(ctx, chunkTexts)
		if err != nil {
			return nil, fmt.Errorf("embedding file %s: %w", relPath, err)
		}

		if err := idx.store.AddChunks(chunks, embs); err != nil {
			return nil, fmt.Errorf("adding chunks to vector store: %w", err)
		}

		idx.store.SetFileHash(relPath, checksum)
		filesIndexed++
	}

	// Persist index to disk
	if err := idx.store.Save(opts.IndexPath); err != nil {
		return nil, fmt.Errorf("saving index to %s: %w", opts.IndexPath, err)
	}

	stats := idx.store.Stats()

	return &IndexReport{
		TotalFilesScanned: len(files),
		FilesIndexed:      filesIndexed,
		FilesSkipped:      filesSkipped,
		TotalChunks:       stats.TotalChunks,
		Duration:          time.Since(startTime),
		EmbedderModel:     idx.embedder.ModelName(),
		IndexPath:         opts.IndexPath,
	}, nil
}

