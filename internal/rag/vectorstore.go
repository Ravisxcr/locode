package rag

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// VectorDoc combines a CodeChunk with its dense vector embedding.
type VectorDoc struct {
	Chunk     CodeChunk `json:"chunk"`
	Embedding []float32 `json:"embedding"`
}

// StoreStats summarizes the vector store contents.
type StoreStats struct {
	TotalChunks int       `json:"total_chunks"`
	TotalFiles  int       `json:"total_files"`
	ModelName   string    `json:"model_name"`
	Dimension   int       `json:"dimension"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// VectorStore holds code chunks and embeddings in memory with disk persistence.
type VectorStore struct {
	mu         sync.RWMutex
	Docs       []VectorDoc       `json:"docs"`
	FileHashes map[string]string `json:"file_hashes"` // filePath -> content sha256
	ModelName  string            `json:"model_name"`
	Dimension  int               `json:"dimension"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// NewVectorStore creates an empty VectorStore.
func NewVectorStore(modelName string, dimension int) *VectorStore {
	return &VectorStore{
		Docs:       make([]VectorDoc, 0),
		FileHashes: make(map[string]string),
		ModelName:  modelName,
		Dimension:  dimension,
		UpdatedAt:  time.Now(),
	}
}

// CosineSimilarity computes the cosine similarity between two float32 vectors.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA <= 0 || normB <= 0 {
		return 0
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// AddDocuments adds a list of chunks and their corresponding embeddings to the store.
func (vs *VectorStore) AddDocuments(chunks []CodeChunk, embeddings [][]float32) error {
	if len(chunks) != len(embeddings) {
		return fmt.Errorf("chunk count (%d) does not match embedding count (%d)", len(chunks), len(embeddings))
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	for i := range chunks {
		vs.Docs = append(vs.Docs, VectorDoc{
			Chunk:     chunks[i],
			Embedding: embeddings[i],
		})
	}
	vs.UpdatedAt = time.Now()
	return nil
}

// SetFileHash records the SHA256 checksum for a file.
func (vs *VectorStore) SetFileHash(filePath, hash string) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	if vs.FileHashes == nil {
		vs.FileHashes = make(map[string]string)
	}
	vs.FileHashes[filePath] = hash
}

// GetFileHash returns the recorded checksum for a file.
func (vs *VectorStore) GetFileHash(filePath string) (string, bool) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	if vs.FileHashes == nil {
		return "", false
	}
	h, ok := vs.FileHashes[filePath]
	return h, ok
}

// RemoveFileChunks removes all chunks associated with a specific file path.
func (vs *VectorStore) RemoveFileChunks(filePath string) {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	filtered := make([]VectorDoc, 0, len(vs.Docs))
	for _, doc := range vs.Docs {
		if doc.Chunk.FilePath != filePath {
			filtered = append(filtered, doc)
		}
	}
	vs.Docs = filtered
	delete(vs.FileHashes, filePath)
	vs.UpdatedAt = time.Now()
}

// VectorSearchResult holds a matched chunk and its cosine similarity score.
type VectorSearchResult struct {
	Chunk CodeChunk
	Score float32
}

// Search retrieves the top-K chunks most similar to the query vector.
func (vs *VectorStore) Search(queryVec []float32, topK int, minScore float32) []VectorSearchResult {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if len(vs.Docs) == 0 || len(queryVec) == 0 {
		return nil
	}

	type scoredDoc struct {
		doc   *VectorDoc
		score float32
	}

	scored := make([]scoredDoc, 0, len(vs.Docs))
	for i := range vs.Docs {
		score := CosineSimilarity(queryVec, vs.Docs[i].Embedding)
		if score >= minScore {
			scored = append(scored, scoredDoc{doc: &vs.Docs[i], score: score})
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	limit := min(topK, len(scored))
	results := make([]VectorSearchResult, limit)
	for i := 0; i < limit; i++ {
		results[i] = VectorSearchResult{
			Chunk: scored[i].doc.Chunk,
			Score: scored[i].score,
		}
	}

	return results
}

// Stats returns store statistics.
func (vs *VectorStore) Stats() StoreStats {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	files := make(map[string]struct{})
	for _, doc := range vs.Docs {
		files[doc.Chunk.FilePath] = struct{}{}
	}

	return StoreStats{
		TotalChunks: len(vs.Docs),
		TotalFiles:  len(files),
		ModelName:   vs.ModelName,
		Dimension:   vs.Dimension,
		UpdatedAt:   vs.UpdatedAt,
	}
}

// Save serializes the vector store to a JSON file.
func (vs *VectorStore) Save(targetPath string) error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
		return fmt.Errorf("creating index directory: %w", err)
	}

	data, err := json.MarshalIndent(vs, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling vector store: %w", err)
	}

	tmpFile := targetPath + ".tmp"
	if err := os.WriteFile(tmpFile, data, 0o644); err != nil {
		return fmt.Errorf("writing temporary index file: %w", err)
	}

	if err := os.Rename(tmpFile, targetPath); err != nil {
		return fmt.Errorf("persisting index file: %w", err)
	}

	return nil
}

// Load loads the vector store from a JSON file.
func (vs *VectorStore) Load(targetPath string) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	data, err := os.ReadFile(targetPath)
	if err != nil {
		return err
	}

	var loaded VectorStore
	if err := json.Unmarshal(data, &loaded); err != nil {
		return fmt.Errorf("unmarshaling vector store: %w", err)
	}

	vs.Docs = loaded.Docs
	vs.FileHashes = loaded.FileHashes
	vs.ModelName = loaded.ModelName
	vs.Dimension = loaded.Dimension
	vs.UpdatedAt = loaded.UpdatedAt

	return nil
}

