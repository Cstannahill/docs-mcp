# Ollama Integration Examples

This document shows how to use the Enhanced Documentation MCP Server with Ollama models and other LLMs that don't have native MCP tool support.

## Quick Start

### 1. Start the HTTP Server

```bash
# Start with basic features
cargo run --release -- --http-server --port 3000

# Start with OpenAI enhanced features
cargo run --release -- --http-server --port 3000 --openai-api-key your_key_here

# Or use environment variable
export OPENAI_API_KEY=your_key_here
cargo run --release -- --http-server --port 3000
```

### 2. Basic Usage Examples

#### Simple Documentation Search

```bash
curl "http://localhost:3000/chat?message=search%20for%20rust%20async%20programming"
```

#### Learning Session Creation

```bash
curl "http://localhost:3000/chat?message=teach%20me%20TypeScript%20basics"
```

#### Get Personalized Recommendations

```bash
curl "http://localhost:3000/chat?message=get%20recommendations&session_id=user123"
```

## Integration with Ollama

### 1. Using curl with Ollama

```bash
# Ask Ollama to use our documentation server
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Please search for information about Rust closures using this API: http://localhost:3000/chat?message=search%20for%20rust%20closures. Then explain what you find.",
  "stream": false
}'
```

### 2. Python Integration Script

```python
import requests
import json

class DocsChatBot:
    def __init__(self, docs_url="http://localhost:3000", ollama_url="http://localhost:11434"):
        self.docs_url = docs_url
        self.ollama_url = ollama_url
        self.session_id = "ollama_session"

    def search_docs(self, query):
        """Search documentation using natural language"""
        response = requests.get(f"{self.docs_url}/chat", params={
            "message": f"search for {query}",
            "session_id": self.session_id
        })
        return response.json()

    def create_learning_path(self, topic, level="intermediate"):
        """Create a learning path for a topic"""
        response = requests.get(f"{self.docs_url}/chat", params={
            "message": f"create {level} tutorial for {topic}",
            "session_id": self.session_id
        })
        return response.json()

    def ask_ollama_with_docs(self, question, model="llama3.2"):
        """Ask Ollama a question enhanced with documentation search"""
        # First, search for relevant documentation
        docs_result = self.search_docs(question)

        if docs_result.get("success"):
            docs_info = docs_result["data"]["response"]

            # Create enhanced prompt for Ollama
            enhanced_prompt = f"""
Based on the following documentation search results, please answer the question: "{question}"

Documentation Results:
{docs_info}

Please provide a comprehensive answer using this documentation and your knowledge.
"""

            # Send to Ollama
            ollama_response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": model,
                "prompt": enhanced_prompt,
                "stream": False
            })

            return ollama_response.json()
        else:
            return {"error": "Failed to retrieve documentation"}

# Usage example
bot = DocsChatBot()

# Search for Rust async programming
result = bot.ask_ollama_with_docs("How do I use async/await in Rust?")
print(result.get("response", "No response"))

# Create a learning path
learning_path = bot.create_learning_path("Python web development", "beginner")
print(learning_path["data"]["response"])
```

### 3. Node.js Integration

```javascript
const axios = require("axios");

class DocsAssistant {
  constructor(
    docsUrl = "http://localhost:3000",
    ollamaUrl = "http://localhost:11434"
  ) {
    this.docsUrl = docsUrl;
    this.ollamaUrl = ollamaUrl;
    this.sessionId = "nodejs_session";
  }

  async searchDocs(query) {
    try {
      const response = await axios.get(`${this.docsUrl}/chat`, {
        params: {
          message: `search for ${query}`,
          session_id: this.sessionId,
        },
      });
      return response.data;
    } catch (error) {
      console.error("Documentation search failed:", error);
      return null;
    }
  }

  async askWithDocs(question, model = "llama3.2") {
    // Search documentation first
    const docsResult = await this.searchDocs(question);

    if (docsResult && docsResult.success) {
      const docsInfo = docsResult.data.response;

      const enhancedPrompt = `
Based on the following documentation search results, answer: "${question}"

Documentation:
${docsInfo}

Provide a detailed answer using this documentation.
`;

      try {
        const ollamaResponse = await axios.post(
          `${this.ollamaUrl}/api/generate`,
          {
            model: model,
            prompt: enhancedPrompt,
            stream: false,
          }
        );

        return ollamaResponse.data.response;
      } catch (error) {
        console.error("Ollama request failed:", error);
        return "Failed to get response from Ollama";
      }
    }

    return "Failed to retrieve documentation";
  }
}

// Usage
async function main() {
  const assistant = new DocsAssistant();

  // Ask about TypeScript
  const answer = await assistant.askWithDocs(
    "How do I use TypeScript interfaces?"
  );
  console.log(answer);

  // Create learning session
  const learning = await assistant.searchDocs(
    "create tutorial for React hooks"
  );
  console.log(learning.data.response);
}

main();
```

## Advanced Features

### 1. Session Persistence

Use the same `session_id` across requests to build learning profiles:

```bash
# First interaction
curl "http://localhost:3000/chat?message=search%20for%20rust%20basics&session_id=user123"

# Later - get personalized recommendations
curl "http://localhost:3000/chat?message=get%20recommendations&session_id=user123"
```

### 2. Learning Path Creation

```bash
# Create structured learning path
curl "http://localhost:3000/chat?message=create%20beginner%20course%20for%20Python%20web%20development%20in%2060%20minutes"

# Interactive tutorial
curl "http://localhost:3000/chat?message=make%20interactive%20tutorial%20for%20Rust%20ownership"
```

### 3. Progress Tracking

```bash
# Track interaction with content
curl -X POST "http://localhost:3000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "bookmark page rust_async_programming_123",
    "session_id": "user123"
  }'

# Check progress
curl "http://localhost:3000/chat?message=show%20my%20progress&session_id=user123"
```

## Natural Language Commands

The system understands these types of natural language commands:

### Search Commands

- "search for rust async programming"
- "find documentation about TypeScript interfaces"
- "semantic search functional programming concepts"
- "look for Python web frameworks"

### Learning Commands

- "teach me React hooks"
- "create tutorial for Rust ownership"
- "help me learn TypeScript at beginner level"
- "make a 30-minute course for Python basics"

### Recommendation Commands

- "get recommendations"
- "what should I learn next?"
- "suggest related topics"
- "show my learning progress"

### Content Management

- "bookmark [page_id]"
- "rate this content highly"
- "find related to async programming"
- "explain closures for beginners"

## API Endpoints

### GET /chat

- `message` (required): Natural language command
- `session_id` (optional): User session for personalization
- `context` (optional): Additional context

### POST /chat

Same parameters as GET but in JSON body.

### GET /search

- `q` (required): Search query
- `type` (optional): semantic, keyword, hybrid, learning
- `session_id` (optional): For personalized results

### GET /info

Returns server capabilities and documentation coverage.

## Coverage

- **Rust**: 2,303 pages (The Rust Book, std library, Cargo)
- **TypeScript**: 686 pages (Handbook, npm documentation)
- **Python**: 3,374 pages (Official docs, pip packages)
- **React**: Complete documentation and guides
- **Tauri**: 117 pages (Framework documentation)

**Total: 8,563+ pages** with semantic search, learning paths, and AI-powered recommendations.

## Benefits for Ollama Models

1. **Rich Context**: Get relevant, up-to-date documentation for any query
2. **Structured Learning**: Create personalized learning paths
3. **Progress Tracking**: Build learning profiles over time
4. **Natural Interface**: Use conversational commands
5. **Multi-Language Support**: Comprehensive coverage across languages
6. **Semantic Understanding**: Vector-based similarity search
7. **Quality Ranking**: AI-powered content relevance scoring

This integration allows Ollama models to provide much more accurate, comprehensive, and up-to-date information about programming languages and frameworks!
