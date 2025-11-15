#!/bin/bash
# Launcher script for Keyword Similarity Analyzer macOS app

echo "🚀 Starting Keyword Similarity Analyzer..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or later."
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import PyQt6" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements-macos.txt
    echo ""
fi

# Run the app
echo "✅ Launching application..."
python3 keyword_analyzer_mac.py
