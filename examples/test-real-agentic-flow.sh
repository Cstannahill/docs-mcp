#!/bin/bash
# examples/test-real-agentic-flow.sh
# Test script for real agentic flow implementation

set -e

echo "🚀 Testing Real Agentic Flow Implementation"
echo "=========================================="

# Check if Ollama is running
echo "📋 Checking Ollama availability..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running on localhost:11434"
    echo "   Please start Ollama first: ollama serve"
    echo "   Then pull a model: ollama pull codellama:7b"
    exit 1
fi

echo "✅ Ollama is running"

# Check if codellama model is available
echo "📋 Checking for codellama model..."
if ! curl -s http://localhost:11434/api/tags | grep -q "codellama"; then
    echo "⚠️  codellama model not found. Pulling it now..."
    ollama pull codellama:7b
fi

echo "✅ codellama model is available"

# Compile the project
echo "🔨 Building project..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build successful"

# Run the integration test
echo "🧪 Running real agentic flow test..."
cargo run --bin test-real-agentic-flow

echo "🎉 Test completed successfully!"
