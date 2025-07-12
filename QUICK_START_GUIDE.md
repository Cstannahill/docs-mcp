# Model Discovery System - Quick Start Guide

## Prerequisites

1. **Install Ollama** (if not already installed):

   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh

   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Start Ollama server**:

   ```bash
   ollama serve
   ```

3. **Install some models** (in another terminal):

   ```bash
   # Lightweight model (good for testing)
   ollama pull llama3.2:3b

   # Code-focused model
   ollama pull deepseek-coder:6.7b

   # Larger general-purpose model
   ollama pull gemma2:9b
   ```

4. **Verify models are installed**:
   ```bash
   ollama list
   ```

## Running the Model Discovery System

### Option 1: Quick Test

```bash
# From the project root
./examples/test-model-discovery.sh
```

### Option 2: Full Integration Example

```bash
# From the project root
cargo run --example model_discovery_integration
```

### Option 3: Custom Configuration

```bash
# Set custom Ollama URL
export OLLAMA_URL="http://localhost:11434"

# Set custom database location
export DATABASE_URL="sqlite:my_models.db"

# Run with custom settings
cargo run --example model_discovery_integration
```

## Expected Output

When working correctly, you should see:

- ✅ Ollama connection successful
- 📋 List of discovered models
- 🔍 Model capabilities and performance metrics
- 💾 Database storage confirmation
- 🎯 Model recommendations for different use cases

## Troubleshooting

### "Ollama is not running"

- Make sure `ollama serve` is running
- Check if port 11434 is accessible: `curl http://localhost:11434/api/tags`

### "No models found"

- Install models: `ollama pull llama3.2:3b`
- Verify: `ollama list`

### "Connection refused"

- Check Ollama is running on the correct port
- Try: `ollama ps` to see running models

### "Database errors"

- Ensure the `test_data/` directory exists
- Check write permissions in the project directory

## Environment Variables

- `OLLAMA_URL`: Ollama server URL (default: `http://127.0.0.1:11434`)
- `DATABASE_URL`: Database location (default: `sqlite:test_data/model_discovery.db`)

## Next Steps

Once the system is working:

1. Set up automated scheduling for daily model discovery
2. Integrate with your existing agentic flow system
3. Configure additional model providers (OpenAI, Anthropic, etc.)
4. Customize model recommendation criteria for your use cases
