BINARY_NAME := gocode
VERSION     ?= v0.9.0
LDFLAGS     := -ldflags "-s -w -X main.version=$(VERSION)"
MODEL       ?= qwen2.5-coder:7b
OLLAMA_HOST ?= 192.168.1.6

.PHONY: all build test clean install index doctor rag

all: build

build:
	go build $(LDFLAGS) -o bin/$(BINARY_NAME) ./cmd/gocode

test:
	go test ./...

clean:
	rm -rf bin/ dist/

install:
	go install $(LDFLAGS) ./cmd/gocode

index: build
	./bin/$(BINARY_NAME) index

doctor: build
	./bin/$(BINARY_NAME) doctor

rag: build
	export OLLAMA_HOST=$(OLLAMA_HOST) && ./bin/$(BINARY_NAME) chat --provider ollama --model $(MODEL) --rag
