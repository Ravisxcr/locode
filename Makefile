BINARY_NAME := gocode
VERSION     ?= v0.9.0
LDFLAGS     := -ldflags "-s -w -X main.version=$(VERSION)"

.PHONY: all build test clean install index doctor

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
