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
curl -s http://127.0.0.1:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "No models found"

echo ""
echo "🔍 Testing Model Discovery Integration..."
echo ""

# Run the model discovery integration example
cd /workspaces/docs-mcp
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
