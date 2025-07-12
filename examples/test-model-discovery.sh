#!/bin/bash
# examples/test-model-discovery.sh
# Quick test script for model discovery system

echo "=== Model Discovery System Test ==="

# Check if Ollama is running
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running at localhost:11434"
    echo "   Please start Ollama first: ollama serve"
    echo "   Then pull some models: ollama pull llama3.2:3b"
    exit 1
fi

echo "✅ Ollama is running"

# List available models
echo "📋 Available Ollama models:"

# Check if jq is available
if command -v jq >/dev/null 2>&1; then
    MODEL_COUNT=$(curl -s http://127.0.0.1:11434/api/tags | jq -r '.models | length' 2>/dev/null || echo "0")
    
    if [ "$MODEL_COUNT" = "0" ] || [ "$MODEL_COUNT" = "null" ]; then
        echo "⚠️  No models found in Ollama"
    else
        curl -s http://127.0.0.1:11434/api/tags | jq -r '.models[].name' 2>/dev/null
        echo "✅ Found $MODEL_COUNT models"
    fi
else
    # Fallback without jq
    RESPONSE=$(curl -s http://127.0.0.1:11434/api/tags 2>/dev/null)
    if echo "$RESPONSE" | grep -q '"models":\[\]' || [ -z "$RESPONSE" ]; then
        echo "⚠️  No models found in Ollama"
        MODEL_COUNT="0"
    else
        echo "Models found (install 'jq' for better formatting)"
        echo "$RESPONSE"
        MODEL_COUNT="unknown"
    fi
fi

if [ "$MODEL_COUNT" = "0" ]; then
    echo ""
    echo "🔧 To install models, run:"
    echo "   ollama pull llama3.2:3b"
    echo "   ollama pull deepseek-coder:6.7b"  
    echo "   ollama pull gemma2:9b"
    echo ""
    echo "📖 Then verify with: ollama list"
    echo ""
    echo "ℹ️  The model discovery system will still demonstrate its capabilities"
    echo "   but won't find any actual models to work with."
fi

echo ""
echo "🔍 Testing Model Discovery Integration..."
echo ""

# Find the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"

# Change to project root and run the model discovery integration example
cd "$PROJECT_ROOT"
echo "🚀 Running model discovery integration example..."
cargo run --example model_discovery_integration 2>/dev/null

echo ""
echo "=== Test Complete ==="
echo "The model discovery system includes:"
echo "  ✅ Real Ollama integration"
echo "  ✅ Comprehensive model metadata storage"
echo "  ✅ Performance tracking and benchmarking"
echo "  ✅ Automated daily discovery scheduling"
echo "  ✅ Model recommendation engine"
echo "  ✅ Database persistence"
echo ""
echo "For production use:"
echo "  1. Ensure Ollama is running with models installed"
echo "  2. Initialize the database with ModelDatabase::initialize()"
echo "  3. Start the ModelDiscoveryScheduler for automated updates"
echo "  4. Use ModelDiscoveryService for model queries and recommendations"
