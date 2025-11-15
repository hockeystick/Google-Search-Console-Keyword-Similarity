.PHONY: help install run build clean

help:
	@echo "Keyword Similarity Analyzer - macOS App Build Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies for macOS app"
	@echo "  make run        - Run the app in development mode"
	@echo "  make build      - Build the macOS .app bundle"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make test       - Run tests"
	@echo ""

install:
	@echo "Installing macOS app dependencies..."
	pip install -r requirements-macos.txt

run:
	@echo "Running app in development mode..."
	python keyword_analyzer_mac.py

build:
	@echo "Building macOS .app bundle..."
	@echo "Note: Requires macOS and py2app"
	python setup.py py2app
	@echo "App bundle created in dist/Keyword Similarity Analyzer.app"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test:
	@echo "Running tests..."
	pytest tests/ -v
