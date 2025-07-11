.PHONY: build test run clean demo examples

# Build the demo application
build:
	go build -o bin/demo cmd/demo/main.go

# Run tests
test:
	go test ./...

# Run tests with verbose output
test-v:
	go test -v ./...

# Run benchmarks
bench:
	go test -bench=. ./...

# Run the demo application
run: build
	./bin/demo

# Run specific demo examples
xor: build
	./bin/demo -example=xor -epochs=1000 -lr=0.01 -optimizer=adam

classification: build
	./bin/demo -example=classification -epochs=500 -lr=0.01 -optimizer=adam

regression: build
	./bin/demo -example=regression -epochs=1000 -lr=0.01 -optimizer=adam

visualization: build
	./bin/demo -example=visualization -epochs=100 -lr=0.1 -optimizer=sgd

# Run with visualization
visualize: build
	./bin/demo -example=xor -epochs=100 -lr=0.1 -optimizer=sgd -visualize

# Run examples
examples:
	go run examples/xor_example.go

# Clean build artifacts
clean:
	rm -rf bin/
	rm -f *.json
	rm -f *.model
	rm -f *.weights

# Install dependencies
deps:
	go mod tidy
	go mod download

# Format code
fmt:
	go fmt ./...

# Lint code
lint:
	golangci-lint run

# Generate documentation
docs:
	godoc -http=:6060

# Build for different platforms
build-all: clean
	GOOS=linux GOARCH=amd64 go build -o bin/demo-linux-amd64 cmd/demo/main.go
	GOOS=windows GOARCH=amd64 go build -o bin/demo-windows-amd64.exe cmd/demo/main.go
	GOOS=darwin GOARCH=amd64 go build -o bin/demo-darwin-amd64 cmd/demo/main.go

# Run all examples
all-examples: xor classification regression visualization

# Help
help:
	@echo "Available commands:"
	@echo "  build        - Build the demo application"
	@echo "  test         - Run tests"
	@echo "  test-v       - Run tests with verbose output"
	@echo "  bench        - Run benchmarks"
	@echo "  run          - Run the demo application"
	@echo "  xor          - Run XOR example"
	@echo "  classification - Run classification example"
	@echo "  regression   - Run regression example"
	@echo "  visualization - Run visualization example"
	@echo "  visualize    - Run with real-time visualization"
	@echo "  examples     - Run standalone examples"
	@echo "  clean        - Clean build artifacts"
	@echo "  deps         - Install dependencies"
	@echo "  fmt          - Format code"
	@echo "  lint         - Lint code"
	@echo "  docs         - Generate documentation"
	@echo "  build-all    - Build for all platforms"
	@echo "  all-examples - Run all examples"
	@echo "  help         - Show this help" 