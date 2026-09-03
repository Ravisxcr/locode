package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
)

// Ensure VectorStore satisfies the langchaingo vectorstores.VectorStore interface.
var _ vectorstores.VectorStore = (*VectorStore)(nil)

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

// VectorStore holds code chunks and embeddings in memory with disk persistence,
// implementing github.com/tmc/langchaingo/vectorstores.VectorStore.
type VectorStore struct {
	mu         sync.RWMutex
	Docs       []VectorDoc       `json:"docs"`
	FileHashes map[string]string `json:"file_hashes"` // filePath -> content sha256
	ModelName  string            `json:"model_name"`
	Dimension  int               `json:"dimension"`
	UpdatedAt  time.Time         `json:"updated_at"`

	embedder embeddings.Embedder
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

// SetEmbedder sets the default embedder for the VectorStore.
func (vs *VectorStore) SetEmbedder(emb embeddings.Embedder) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.embedder = emb
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

// AddChunks adds a list of chunks and their corresponding precomputed embeddings to the store.
func (vs *VectorStore) AddChunks(chunks []CodeChunk, embeddings [][]float32) error {
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

// AddDocuments adds a list of schema.Document items to the store, implementing langchaingo vectorstores.VectorStore.
func (vs *VectorStore) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) ([]string, error) {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}

	emb := opts.Embedder
	if emb == nil {
		emb = vs.embedder
	}

	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}

	var vectors [][]float32
	if emb != nil {
		var err error
		vectors, err = emb.EmbedDocuments(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("embedding documents: %w", err)
		}
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	ids := make([]string, len(docs))
	for i, doc := range docs {
		chunk := DocumentToChunk(doc)
		if chunk.ID == "" {
			chunk.ID = fmt.Sprintf("doc_%d_%d", time.Now().UnixNano(), i)
		}
		ids[i] = chunk.ID

		var vec []float32
		if i < len(vectors) {
			vec = vectors[i]
		}

		vs.Docs = append(vs.Docs, VectorDoc{
			Chunk:     chunk,
			Embedding: vec,
		})
	}
	vs.UpdatedAt = time.Now()
	return ids, nil
}

// SimilaritySearch retrieves the top matching documents for a query text, implementing langchaingo vectorstores.VectorStore.
func (vs *VectorStore) SimilaritySearch(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
	opts := vectorstores.Options{}
	for _, opt := range options {
		opt(&opts)
	}

	emb := opts.Embedder
	if emb == nil {
		emb = vs.embedder
	}
	if emb == nil {
		return nil, fmt.Errorf("no embedder configured for SimilaritySearch")
	}

	queryVec, err := emb.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	results := vs.Search(queryVec, numDocuments, opts.ScoreThreshold)
	docs := make([]schema.Document, len(results))
	for i, res := range results {
		doc := ChunkToDocument(res.Chunk)
		doc.Score = res.Score
		docs[i] = doc
	}
	return docs, nil
}

// ToRetriever returns a langchaingo Retriever wrapping this VectorStore.
func (vs *VectorStore) ToRetriever(numDocuments int, options ...vectorstores.Option) vectorstores.Retriever {
	return vectorstores.ToRetriever(vs, numDocuments, options...)
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

// ChunkToDocument converts an internal CodeChunk to a langchaingo schema.Document.
func ChunkToDocument(c CodeChunk) schema.Document {
	return schema.Document{
		PageContent: c.Content,
		Metadata: map[string]any{
			"id":          c.ID,
			"file_path":   c.FilePath,
			"language":    c.Language,
			"start_line":  c.StartLine,
			"end_line":    c.EndLine,
			"token_count": c.TokenCount,
			"chunk_index": c.ChunkIndex,
			"checksum":    c.Checksum,
		},
	}
}

// DocumentToChunk converts a langchaingo schema.Document back to an internal CodeChunk.
func DocumentToChunk(doc schema.Document) CodeChunk {
	c := CodeChunk{
		Content: doc.PageContent,
	}
	if doc.Metadata != nil {
		if v, ok := doc.Metadata["id"].(string); ok {
			c.ID = v
		}
		if v, ok := doc.Metadata["file_path"].(string); ok {
			c.FilePath = v
		}
		if v, ok := doc.Metadata["language"].(string); ok {
			c.Language = v
		}
		if v, ok := doc.Metadata["start_line"].(int); ok {
			c.StartLine = v
		} else if v, ok := doc.Metadata["start_line"].(float64); ok {
			c.StartLine = int(v)
		}
		if v, ok := doc.Metadata["end_line"].(int); ok {
			c.EndLine = v
		} else if v, ok := doc.Metadata["end_line"].(float64); ok {
			c.EndLine = int(v)
		}
		if v, ok := doc.Metadata["token_count"].(int); ok {
			c.TokenCount = v
		} else if v, ok := doc.Metadata["token_count"].(float64); ok {
			c.TokenCount = int(v)
		}
		if v, ok := doc.Metadata["chunk_index"].(int); ok {
			c.ChunkIndex = v
		} else if v, ok := doc.Metadata["chunk_index"].(float64); ok {
			c.ChunkIndex = int(v)
		}
		if v, ok := doc.Metadata["checksum"].(string); ok {
			c.Checksum = v
		}
	}
	return c
}
