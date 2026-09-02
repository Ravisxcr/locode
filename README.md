# gocode

An open-source, terminal-native AI coding agent built in Go. Inspired by Claude Code, engineered for speed, and powered by **Ollama**, **LangChain-style Codebase RAG**, and **11+ LLM providers**.

---

## Key Features

- **Local & Cloud LLMs**: Native support for **Ollama** (`llama3.3`, `qwen2.5-coder`, `deepseek-r1`, `phi4`), **Anthropic**, **OpenAI**, **Google Gemini**, **DeepSeek**, **Groq**, **xAI**, **Mistral**, and **OpenRouter**.
- **Vector Codebase RAG**: Syntax-aware code chunking, in-memory vector store, and hybrid dense + BM25 sparse retrieval with Reciprocal Rank Fusion (RRF).
- **RAG-Grounded Code Synthesis**: Context-aware code generation using `rag_code_context` to match repository patterns, reuse existing utilities, and satisfy interface contracts.
- **Claude Code Developer Experience**:
  - Interactive REPL with streaming tokens and spinner status.
  - Multi-turn agentic loop with autonomous tool execution.
  - Interactive slash commands: `/index`, `/rag`, `/plan`, `/diff`, `/undo`, `/compact`, `/skill`, `/doctor`.
- **Fast & Dependency-Free**: Single Go binary with sub-10ms startup time, zero Python/Node.js runtime requirements.

---

## Installation

### Prerequisites
- [Go 1.22+](https://golang.org/dl/)

### Build from Source

```bash
git clone https://github.com/Ravisxcr/gocode-rag.git
cd gocode-rag
make build
```

Binary will be generated at `./bin/gocode`.

### Install Globally

```bash
make install
```

---

## Quickstart

### 1. Index Your Codebase for RAG

Index your workspace to enable semantic search and vector-grounded code generation:

```bash
# Index using pure-Go local vectorizer (zero setup required)
gocode index

# Or index using Ollama with local embeddings
gocode index --embedder ollama --model nomic-embed-text
```

### 2. Search Your Codebase Semantically

```bash
# Query the codebase via CLI
gocode rag search "session persistence and recovery"

# Check a file for duplicate patterns and related test suites
gocode rag check internal/session/session.go
```

### 3. Start Interactive Agent Chat

```bash
# Run with local Ollama
gocode chat --provider ollama --model llama3.3 --rag

# Run with Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
gocode chat --model sonnet --rag

# Run with OpenAI
export OPENAI_API_KEY="sk-..."
gocode chat --model gpt-4o --rag
```

### 4. One-Shot Command Prompt

```bash
gocode prompt --provider ollama --model llama3.3 --rag \
  "Implement a token verification middleware following existing auth patterns"
```

---

## CLI & REPL Commands

### CLI Commands

| Command | Description |
|---------|-------------|
| `gocode chat` | Start interactive agent REPL |
| `gocode prompt "<text>"` | Run a single one-shot agent prompt |
| `gocode index [path]` | Index codebase workspace for vector RAG |
| `gocode rag search <query>` | Query the indexed codebase semantically |
| `gocode rag check <file>` | Analyze file against index for duplicates and patterns |
| `gocode rag status` | Display RAG vector store statistics |
| `gocode ollama list` | List locally installed Ollama models |
| `gocode ollama status` | Check Ollama server connectivity and latency |
| `gocode doctor` | Validate environment, API keys, and RAG status |

### REPL Slash Commands

| Slash Command | Description |
|---------------|-------------|
| `/index` | Re-index workspace directly from chat session |
| `/rag <query>` | Perform semantic codebase search |
| `/plan` | Start an interactive architecture planning session |
| `/diff` | Show git diff of changes made during the session |
| `/undo [N]` | Revert changes to a checkpoint |
| `/compact` | Compact conversation history to free context window |
| `/skill <name>` | Activate domain-specific skill (`rag-code-architect`, `golang-best-practices`, etc.) |
| `/doctor` | Run environment diagnostics |
| `/exit` | Exit the interactive session |

---

## Architecture

```
gocode/
├── cmd/gocode/          # CLI entrypoint & subcommand routing
├── data/                # Embedded tool and command registries
└── internal/
    ├── agent/           # Autonomous conversation runtime & tool executor
    ├── apiclient/       # Ollama, Anthropic, OpenAI, Gemini, & OpenRouter providers
    ├── rag/             # Chunker, vector store, hybrid retriever, indexer, & tools
    ├── repl/            # Interactive REPL, prompt builder, & slash command parser
    ├── skills/          # Domain skills (including rag-code-architect)
    ├── session/         # Session persistence and recovery
    └── toolimpl/        # Built-in tool implementations (file, bash, git, RAG)
```

---

## License

MIT
