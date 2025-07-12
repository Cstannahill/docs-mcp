use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::{
    cors::CorsLayer,
    services::ServeDir,
};
use tracing::{info, error};

use crate::chat_interface::{ChatInterface, ChatRequest, ChatResponse};
use crate::enhanced_search::EnhancedSearchSystem;
use crate::database::Database;

pub struct HttpServer {
    chat_interface: ChatInterface,
    enhanced_search: EnhancedSearchSystem,
    db: Database,
}

#[derive(Debug, Deserialize)]
pub struct ChatQueryParams {
    pub message: String,
    pub session_id: Option<String>,
    pub context: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatPostRequest {
    pub message: String,
    pub session_id: Option<String>,
    pub context: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub features: Vec<String>,
    pub endpoints: Vec<EndpointInfo>,
    pub total_documents: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct EndpointInfo {
    pub path: String,
    pub method: String,
    pub description: String,
    pub parameters: Vec<String>,
}

type AppState = Arc<HttpServer>;

impl HttpServer {
    pub async fn new(database_url: &str, openai_api_key: Option<String>) -> Result<Self> {
        let db = Database::new(database_url).await?;
        
        // Run migrations
        db.run_advanced_features_migration().await?;
        
        let enhanced_search = EnhancedSearchSystem::new(db.clone(), openai_api_key).await?;
        let chat_interface = ChatInterface::new(enhanced_search.clone()).await?;

        Ok(Self {
            chat_interface,
            enhanced_search,
            db,
        })
    }

    pub async fn start_server(&self, port: u16) -> Result<()> {
        let state = Arc::new(self.clone());
        
        let app = Router::new()
            .route("/", get(root_handler))
            .route("/health", get(health_handler))
            .route("/info", get(info_handler))
            .route("/chat", get(chat_get_handler))
            .route("/chat", post(chat_post_handler))
            .route("/search", get(search_handler))
            .route("/docs", get(docs_handler))
            .nest_service("/static", ServeDir::new("static"))
            .fallback_service(ServeDir::new("static"))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let addr = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&addr).await?;
        
        info!("🚀 HTTP Server starting on http://{}", addr);
        info!("📚 Documentation chat interface available at http://{}:{}/chat", "localhost", port);
        info!("📖 API documentation available at http://{}:{}/docs", "localhost", port);
        
        axum::serve(listener, app).await?;
        
        Ok(())
    }
}

impl Clone for HttpServer {
    fn clone(&self) -> Self {
        Self {
            chat_interface: self.chat_interface.clone(),
            enhanced_search: self.enhanced_search.clone(),
            db: self.db.clone(),
        }
    }
}

// Route handlers
async fn root_handler() -> Json<ApiResponse<ServerInfo>> {
    let server_info = ServerInfo {
        name: "Enhanced Documentation MCP Server".to_string(),
        version: "1.0.0".to_string(),
        description: "AI-powered documentation system with semantic search, learning paths, and natural language interface".to_string(),
        features: vec![
            "Natural Language Chat Interface".to_string(),
            "Semantic Search with Vector Embeddings".to_string(),
            "Personalized Learning Paths".to_string(),
            "Interactive Tutorials".to_string(),
            "Multi-factor Content Ranking".to_string(),
            "User Interaction Tracking".to_string(),
            "Adaptive Recommendations".to_string(),
            "8,563+ Documentation Pages".to_string(),
        ],
        endpoints: vec![
            EndpointInfo {
                path: "/chat".to_string(),
                method: "GET/POST".to_string(),
                description: "Natural language chat interface for documentation queries".to_string(),
                parameters: vec!["message".to_string(), "session_id (optional)".to_string(), "context (optional)".to_string()],
            },
            EndpointInfo {
                path: "/search".to_string(),
                method: "GET".to_string(),
                description: "Direct search interface with advanced filtering".to_string(),
                parameters: vec!["query".to_string(), "type (semantic|keyword|hybrid)".to_string()],
            },
            EndpointInfo {
                path: "/info".to_string(),
                method: "GET".to_string(),
                description: "Server information and capabilities".to_string(),
                parameters: vec![],
            },
        ],
        total_documents: None, // Would be populated from database
    };

    Json(ApiResponse {
        success: true,
        data: Some(server_info),
        error: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

async fn health_handler() -> Json<ApiResponse<HashMap<String, String>>> {
    let mut health = HashMap::new();
    health.insert("status".to_string(), "healthy".to_string());
    health.insert("uptime".to_string(), "active".to_string());

    Json(ApiResponse {
        success: true,
        data: Some(health),
        error: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

async fn info_handler(State(state): State<AppState>) -> Json<ApiResponse<ServerInfo>> {
    let server_info = ServerInfo {
        name: "Enhanced Documentation MCP Server".to_string(),
        version: "1.0.0".to_string(),
        description: "AI-powered documentation system with comprehensive language support".to_string(),
        features: vec![
            "🤖 Natural Language Chat Interface".to_string(),
            "🔍 Semantic Search with Vector Embeddings".to_string(),
            "📚 Personalized Learning Paths".to_string(),
            "🎯 Interactive Tutorials".to_string(),
            "⭐ Multi-factor Content Ranking".to_string(),
            "📊 User Interaction Tracking".to_string(),
            "🎨 Adaptive Recommendations".to_string(),
            "📖 8,563+ Documentation Pages".to_string(),
            "🦀 Rust (Book, std, Cargo)".to_string(),
            "🟦 TypeScript & npm".to_string(),
            "🐍 Python & pip".to_string(),
            "⚛️ React Documentation".to_string(),
            "🦋 Tauri Framework".to_string(),
        ],
        endpoints: vec![
            EndpointInfo {
                path: "/chat".to_string(),
                method: "GET/POST".to_string(),
                description: "Chat with the documentation using natural language".to_string(),
                parameters: vec!["message (required)".to_string(), "session_id (optional)".to_string()],
            },
            EndpointInfo {
                path: "/search".to_string(),
                method: "GET".to_string(),
                description: "Advanced search with AI ranking".to_string(),
                parameters: vec!["q (query)".to_string(), "type (search type)".to_string()],
            },
        ],
        total_documents: None,
    };

    Json(ApiResponse {
        success: true,
        data: Some(server_info),
        error: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

async fn chat_get_handler(
    Query(params): Query<ChatQueryParams>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<ChatResponse>>, StatusCode> {
    let session_id = params.session_id.unwrap_or_else(|| "default".to_string());
    
    let chat_request = ChatRequest {
        message: params.message,
        session_id,
        context: params.context,
    };

    match state.chat_interface.process_chat(chat_request).await {
        Ok(response) => Ok(Json(ApiResponse {
            success: true,
            data: Some(response),
            error: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })),
        Err(e) => {
            error!("Chat processing error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn chat_post_handler(
    State(state): State<AppState>,
    Json(payload): Json<ChatPostRequest>,
) -> Result<Json<ApiResponse<ChatResponse>>, StatusCode> {
    let session_id = payload.session_id.unwrap_or_else(|| "default".to_string());
    
    let chat_request = ChatRequest {
        message: payload.message,
        session_id,
        context: payload.context,
    };

    match state.chat_interface.process_chat(chat_request).await {
        Ok(response) => Ok(Json(ApiResponse {
            success: true,
            data: Some(response),
            error: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })),
        Err(e) => {
            error!("Chat processing error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    q: String,
    #[serde(rename = "type")]
    search_type: Option<String>,
    session_id: Option<String>,
}

async fn search_handler(
    Query(params): Query<SearchParams>,
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<serde_json::Value>>, StatusCode> {
    let search_type = match params.search_type.as_deref() {
        Some("semantic") => crate::enhanced_search::SearchType::Semantic,
        Some("keyword") => crate::enhanced_search::SearchType::Keyword,
        Some("learning") => crate::enhanced_search::SearchType::LearningFocused,
        _ => crate::enhanced_search::SearchType::Hybrid,
    };

    let search_request = crate::enhanced_search::EnhancedSearchRequest {
        query: params.q,
        session_id: params.session_id.unwrap_or_else(|| "default".to_string()),
        user_context: None,
        search_type,
        filters: crate::enhanced_search::SearchFilters {
            doc_types: Vec::new(),
            difficulty_levels: Vec::new(),
            content_types: Vec::new(),
            date_range: None,
            exclude_completed: false,
        },
        include_suggestions: true,
        include_learning_paths: true,
    };

    match state.enhanced_search.enhanced_search(search_request).await {
        Ok(response) => {
            let response_json = serde_json::to_value(response).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(Json(ApiResponse {
                success: true,
                data: Some(response_json),
                error: None,
                timestamp: chrono::Utc::now().to_rfc3339(),
            }))
        }
        Err(e) => {
            error!("Search error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn docs_handler() -> &'static str {
    r#"
# Enhanced Documentation MCP Server API

## Chat Interface (Natural Language)

### GET /chat
**Natural language documentation queries**
- `message` (required): Your question or command
- `session_id` (optional): Session identifier for personalization
- `context` (optional): Additional context

**Examples:**
- `/chat?message=search for rust async programming`
- `/chat?message=teach me TypeScript basics&session_id=user123`
- `/chat?message=create tutorial for Python web development`
- `/chat?message=get my learning recommendations`

### POST /chat
Same as GET but with JSON body:
```json
{
  "message": "explain closures in rust",
  "session_id": "user123",
  "context": "I'm a beginner programmer"
}
```

## Direct Search Interface

### GET /search
**Advanced search with AI ranking**
- `q` (required): Search query
- `type` (optional): semantic, keyword, hybrid, or learning
- `session_id` (optional): For personalized results

**Example:**
- `/search?q=async await&type=semantic`

## Natural Language Commands

The chat interface understands these types of commands:

**Search & Discovery:**
- "search for [topic]"
- "find documentation about [topic]"
- "semantic search [concept]"
- "learn about [topic]"

**Learning & Tutorials:**
- "create tutorial for [topic]"
- "teach me [topic]"
- "help me learn [topic] at beginner level"
- "make a course for [topic]"

**Recommendations:**
- "get recommendations"
- "what should I learn next?"
- "show my progress"

**Content Management:**
- "bookmark [page]"
- "rate [content]"
- "find related to [topic]"

## Features

- 🤖 **AI-Powered Chat**: Natural language interface for all features
- 🔍 **Semantic Search**: Vector embeddings for conceptual similarity
- 📚 **Learning Paths**: Personalized, adaptive learning journeys
- 🎯 **Interactive Tutorials**: Step-by-step guided learning
- ⭐ **Smart Ranking**: Multi-factor relevance scoring
- 📊 **Progress Tracking**: User interaction analytics
- 🎨 **Recommendations**: AI-powered content suggestions

## Documentation Coverage

- **Rust**: 2,303 pages (Book, std library, Cargo)
- **TypeScript**: 686 pages (Handbook, npm docs)
- **Python**: 3,374 pages (Official docs, pip packages)
- **React**: Documentation and guides
- **Tauri**: 117 pages (Framework documentation)

**Total: 8,563+ pages** across all languages and frameworks.

## Usage Examples

### For Ollama Models
```bash
# Simple chat query
curl "http://localhost:3000/chat?message=search%20for%20rust%20closures"

# Learning session
curl "http://localhost:3000/chat?message=teach%20me%20async%20programming%20in%20rust"

# Get recommendations
curl "http://localhost:3000/chat?message=get%20recommendations&session_id=user123"
```

### Integration with AI Models
```python
import requests

# Chat with the documentation
response = requests.get("http://localhost:3000/chat", params={
    "message": "explain rust ownership model for beginners",
    "session_id": "my_session"
})

result = response.json()
print(result["data"]["response"])
```

The system learns from interactions to provide increasingly personalized recommendations!
    "#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        // Test server initialization
    }
}
