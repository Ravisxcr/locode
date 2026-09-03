package toolimpl

import (
	"os"
	"strings"
	"testing"

	"github.com/Ravisxcr/gocode-rag/internal/memdir"
)

func TestMemoryTool_StoreAndRecall(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "memtool-test-*")
	if err != nil {
		t.Fatalf("creating temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	store := memdir.NewStoreWithDirs(tmpDir, tmpDir, tmpDir)
	tool := &MemoryTool{store: store}

	// 1. Store a memory
	res := tool.Execute(map[string]interface{}{
		"action":  "store",
		"content": "The codebase uses standard library net/http for microservices",
		"scope":   "project",
	})
	if !res.Success {
		t.Fatalf("expected store success, got error: %s", res.Error)
	}

	// 2. Recall memory with query match
	res = tool.Execute(map[string]interface{}{
		"action": "recall",
		"query":  "net/http microservices",
	})
	if !res.Success {
		t.Fatalf("expected recall success, got error: %s", res.Error)
	}
	if !strings.Contains(res.Output, "net/http") {
		t.Errorf("expected recall output to contain net/http, got: %s", res.Output)
	}

	// 3. List memories
	res = tool.Execute(map[string]interface{}{
		"action": "list",
	})
	if !res.Success {
		t.Fatalf("expected list success, got error: %s", res.Error)
	}
	if !strings.Contains(res.Output, "net/http") {
		t.Errorf("expected list output to contain stored memory, got: %s", res.Output)
	}
}
