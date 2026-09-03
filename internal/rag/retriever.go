package rag

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/langchaingo/schema"
)

// Ensure Retriever satisfies the langchaingo schema.Retriever interface.
var _ schema.Retriever = (*Retriever)(nil)

// SearchResult represents a retrieved chunk with similarity scoring.
type SearchResult struct {
	Chunk      CodeChunk `json:"chunk"`
	Score      float64   `json:"score"`
	DenseRank  int       `json:"dense_rank,omitempty"`
	SparseRank int       `json:"sparse_rank,omitempty"`
}

// Retriever performs hybrid (dense + sparse) retrieval over indexed code chunks.
type Retriever struct {
	store    *VectorStore
	embedder Embedder
}

// NewRetriever creates a new Retriever.
func NewRetriever(store *VectorStore, embedder Embedder) *Retriever {
	return &Retriever{
		store:    store,
		embedder: embedder,
	}
}

// GetRelevantDocuments satisfies the langchaingo schema.Retriever interface.
func (r *Retriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
	results, err := r.Retrieve(ctx, query, 5, "")
	if err != nil {
		return nil, err
	}
	docs := make([]schema.Document, len(results))
	for i, res := range results {
		d := ChunkToDocument(res.Chunk)
		d.Score = float32(res.Score)
		docs[i] = d
	}
	return docs, nil
}

// Retrieve queries the vector store and sparse lexical index, combining them with Reciprocal Rank Fusion.
func (r *Retriever) Retrieve(ctx context.Context, query string, limit int, pathFilter string) ([]SearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 5
	}

	// 1. Dense Semantic Search
	queryVec, err := r.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("generating query embedding: %w", err)
	}

	denseResults := r.store.Search(queryVec, limit*3, 0.0)

	// Filter by path if requested
	if pathFilter != "" {
		filtered := make([]VectorSearchResult, 0, len(denseResults))
		for _, dr := range denseResults {
			if matchesPathFilter(dr.Chunk.FilePath, pathFilter) {
				filtered = append(filtered, dr)
			}
		}
		denseResults = filtered
	}

	// 2. Sparse Lexical Search (Keyword / Code Token overlap)
	sparseResults := r.sparseSearch(query, limit*3, pathFilter)

	// 3. Reciprocal Rank Fusion (RRF)
	// Score = 1 / (60 + rank_dense) + 1 / (60 + rank_sparse)
	type fusionEntry struct {
		chunk      CodeChunk
		denseRank  int
		sparseRank int
		rrfScore   float64
	}

	chunkMap := make(map[string]*fusionEntry)

	for rank, dr := range denseResults {
		id := dr.Chunk.ID
		if _, ok := chunkMap[id]; !ok {
			chunkMap[id] = &fusionEntry{
				chunk:      dr.Chunk,
				denseRank:  rank + 1,
				sparseRank: 0,
			}
		} else {
			chunkMap[id].denseRank = rank + 1
		}
	}

	for rank, sr := range sparseResults {
		id := sr.Chunk.ID
		if _, ok := chunkMap[id]; !ok {
			chunkMap[id] = &fusionEntry{
				chunk:      sr.Chunk,
				denseRank:  0,
				sparseRank: rank + 1,
			}
		} else {
			chunkMap[id].sparseRank = rank + 1
		}
	}

	// Compute RRF scores
	const rrfK = 60.0
	results := make([]SearchResult, 0, len(chunkMap))
	for _, entry := range chunkMap {
		var score float64
		if entry.denseRank > 0 {
			score += 1.0 / (rrfK + float64(entry.denseRank))
		}
		if entry.sparseRank > 0 {
			score += 1.0 / (rrfK + float64(entry.sparseRank))
		}
		entry.rrfScore = score

		results = append(results, SearchResult{
			Chunk:      entry.chunk,
			Score:      score,
			DenseRank:  entry.denseRank,
			SparseRank: entry.sparseRank,
		})
	}

	// Sort descending by RRF score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// sparseSearch computes lexical token overlap between query and all stored chunks.
func (r *Retriever) sparseSearch(query string, topK int, pathFilter string) []SearchResult {
	queryTokens := TokenizeCode(query)
	if len(queryTokens) == 0 {
		return nil
	}

	querySet := make(map[string]struct{})
	for _, t := range queryTokens {
		querySet[t] = struct{}{}
	}

	r.store.mu.RLock()
	defer r.store.mu.RUnlock()

	type scoredSparse struct {
		chunk CodeChunk
		score float64
	}

	var candidates []scoredSparse
	for i := range r.store.Docs {
		chunk := r.store.Docs[i].Chunk
		if pathFilter != "" && !matchesPathFilter(chunk.FilePath, pathFilter) {
			continue
		}

		chunkTokens := TokenizeCode(chunk.Content)
		if len(chunkTokens) == 0 {
			continue
		}

		matchedCount := 0
		for _, ct := range chunkTokens {
			if _, ok := querySet[ct]; ok {
				matchedCount++
			}
		}

		if matchedCount > 0 {
			score := float64(matchedCount) / float64(len(queryTokens))
			candidates = append(candidates, scoredSparse{chunk: chunk, score: score})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	limit := min(topK, len(candidates))
	results := make([]SearchResult, limit)
	for i := 0; i < limit; i++ {
		results[i] = SearchResult{
			Chunk: candidates[i].chunk,
			Score: candidates[i].score,
		}
	}
	return results
}

func matchesPathFilter(filePath, filter string) bool {
	matched, err := filepath.Match(filter, filePath)
	if err == nil && matched {
		return true
	}
	return strings.Contains(strings.ToLower(filePath), strings.ToLower(filter))
}

// FormatContext formats a list of search results into Markdown code blocks for LLM context or tool responses.
func FormatContext(results []SearchResult) string {
	if len(results) == 0 {
		return "No relevant code snippets found in RAG index."
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Found %d relevant code snippet(s):\n\n", len(results)))

	for i, r := range results {
		c := r.Chunk
		sb.WriteString(fmt.Sprintf("### %d. `%s` (Lines %d-%d, %s)\n", i+1, c.FilePath, c.StartLine, c.EndLine, c.Language))
		sb.WriteString(fmt.Sprintf("```%s\n%s\n```\n\n", c.Language, c.Content))
	}

	return strings.TrimSpace(sb.String())
}

