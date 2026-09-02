package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/Ravisxcr/gocode-rag/internal/apiclient"
	"github.com/Ravisxcr/gocode-rag/internal/apitypes"
)

// Embedder generates vector embeddings for text and code chunks.
type Embedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
	EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)
	Dimension() int
	ModelName() string
}

// EmbedderConfig holds configuration for resolving the embedder.
type EmbedderConfig struct {
	Provider string // "ollama", "openai", "local", or "auto"
	Model    string // e.g. "nomic-embed-text", "text-embedding-3-small"
	Host     string // custom host URL (e.g. http://localhost:11434)
	APIKey   string
}

// ResolveEmbedder automatically selects and returns the most suitable Embedder.
func ResolveEmbedder(cfg EmbedderConfig) Embedder {
	prov := strings.ToLower(strings.TrimSpace(cfg.Provider))

	switch prov {
	case "local", "bm25", "tfidf":
		return NewLocalBM25Embedder(256)
	case "openai":
		model := cfg.Model
		if model == "" {
			model = "text-embedding-3-small"
		}
		apiKey := cfg.APIKey
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
		baseURL := os.Getenv("OPENAI_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.openai.com/v1"
		}
		return NewOpenAIEmbedder(baseURL, apiKey, model, 1536)
	case "ollama":
		model := cfg.Model
		if model == "" {
			model = "nomic-embed-text"
		}
		host := cfg.Host
		if host == "" {
			host = apiclient.DefaultOllamaHost()
		}
		return NewOllamaEmbedder(host, model)
	default:
		// "auto" detection: check Ollama first, then OpenAI env var, then fallback to LocalBM25
		host := cfg.Host
		if host == "" {
			host = apiclient.DefaultOllamaHost()
		}

		// Quick probe Ollama
		ctx, cancel := context.WithTimeout(context.Background(), 800*time.Millisecond)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, host+"/api/tags", nil)
		if err == nil {
			if resp, err := (&http.Client{Timeout: 800 * time.Millisecond}).Do(req); err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					model := cfg.Model
					if model == "" {
						model = "nomic-embed-text"
					}
					return NewOllamaEmbedder(host, model)
				}
			}
		}

		if key := os.Getenv("OPENAI_API_KEY"); key != "" {
			return NewOpenAIEmbedder("https://api.openai.com/v1", key, "text-embedding-3-small", 1536)
		}

		// Default fallback
		return NewLocalBM25Embedder(256)
	}
}

// --- Ollama Embedder ---

type OllamaEmbedder struct {
	client    *apiclient.OllamaProvider
	model     string
	dimension int
}

// NewOllamaEmbedder creates an Ollama-backed embedder.
func NewOllamaEmbedder(host, model string) *OllamaEmbedder {
	if model == "" {
		model = "nomic-embed-text"
	}
	prov := apiclient.NewOllamaProvider(host, apitypes.AuthSource{})
	return &OllamaEmbedder{
		client:    prov,
		model:     model,
		dimension: 768, // standard for nomic-embed-text; updated dynamically on first call
	}
}

func (e *OllamaEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	emb, err := e.client.GenerateEmbedding(ctx, e.model, text)
	if err != nil {
		return nil, err
	}
	if len(emb) > 0 {
		e.dimension = len(emb)
	}
	return emb, nil
}

func (e *OllamaEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	embs, err := e.client.GenerateEmbeddingsBatch(ctx, e.model, texts)
	if err != nil {
		return nil, err
	}
	if len(embs) > 0 && len(embs[0]) > 0 {
		e.dimension = len(embs[0])
	}
	return embs, nil
}

func (e *OllamaEmbedder) Dimension() int   { return e.dimension }
func (e *OllamaEmbedder) ModelName() string { return "ollama/" + e.model }

// --- OpenAI Embedder ---

type OpenAIEmbedder struct {
	baseURL   string
	apiKey    string
	model     string
	dimension int
	client    *http.Client
}

// NewOpenAIEmbedder creates an OpenAI-compatible embedder.
func NewOpenAIEmbedder(baseURL, apiKey, model string, dimension int) *OpenAIEmbedder {
	if dimension <= 0 {
		dimension = 1536
	}
	return &OpenAIEmbedder{
		baseURL:   strings.TrimRight(baseURL, "/"),
		apiKey:    apiKey,
		model:     model,
		dimension: dimension,
		client:    &http.Client{Timeout: 60 * time.Second},
	}
}

func (e *OpenAIEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	embs, err := e.EmbedDocuments(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embs) == 0 {
		return nil, fmt.Errorf("empty embedding response from OpenAI")
	}
	return embs[0], nil
}

func (e *OpenAIEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	type embRequest struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	type embData struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	type embResponse struct {
		Data []embData `json:"data"`
	}

	payload, err := json.Marshal(embRequest{Model: e.model, Input: texts})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embeddings", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai embeddings API returned HTTP %d", resp.StatusCode)
	}

	var res embResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}

	result := make([][]float32, len(texts))
	for _, item := range res.Data {
		if item.Index >= 0 && item.Index < len(result) {
			result[item.Index] = item.Embedding
		}
	}
	return result, nil
}

func (e *OpenAIEmbedder) Dimension() int   { return e.dimension }
func (e *OpenAIEmbedder) ModelName() string { return "openai/" + e.model }

// --- Local BM25 / Term-Frequency Embedder (Zero Dependencies) ---

// LocalBM25Embedder computes hashed term-frequency vectors in pure Go.
type LocalBM25Embedder struct {
	dim int
}

// NewLocalBM25Embedder creates a pure Go hash-based vectorizer.
func NewLocalBM25Embedder(dim int) *LocalBM25Embedder {
	if dim <= 0 {
		dim = 256
	}
	return &LocalBM25Embedder{dim: dim}
}

var tokenSplitRegex = regexp.MustCompile(`[A-Z][a-z]+|[A-Z]+|[a-z]+|\d+`)

// TokenizeCode extracts normalized tokens (camelCase, snake_case, paths, symbols) from code.
func TokenizeCode(text string) []string {
	if text == "" {
		return nil
	}

	matches := tokenSplitRegex.FindAllString(text, -1)
	tokens := make([]string, 0, len(matches))
	seen := make(map[string]struct{})

	for _, m := range matches {
		m = strings.ToLower(strings.TrimSpace(m))
		if len(m) > 1 {
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				tokens = append(tokens, m)
			}
		}
	}
	return tokens
}

func (e *LocalBM25Embedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return e.vectorize(text), nil
}

func (e *LocalBM25Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	res := make([][]float32, len(texts))
	for i, t := range texts {
		res[i] = e.vectorize(t)
	}
	return res, nil
}

func (e *LocalBM25Embedder) vectorize(text string) []float32 {
	vec := make([]float32, e.dim)
	tokens := TokenizeCode(text)
	if len(tokens) == 0 {
		return vec
	}

	tf := make(map[string]int)
	for _, t := range tokens {
		tf[t]++
	}

	for word, count := range tf {
		h := fnv.New32a()
		_, _ = h.Write([]byte(word))
		bucket := int(h.Sum32() % uint32(e.dim))
		// Log-frequency weighting
		weight := float32(1.0 + math.Log(float64(count)))
		vec[bucket] += weight
	}

	// L2 normalization
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	if norm > 0 {
		norm = float32(math.Sqrt(float64(norm)))
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec
}

func (e *LocalBM25Embedder) Dimension() int   { return e.dim }
func (e *LocalBM25Embedder) ModelName() string { return "local/bm25-tfidf" }
