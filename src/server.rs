use anyhow::Result;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use tracing::{info, error, debug};

use crate::database::{Database, SearchQuery, SearchFilters, RankingPreferences, DocType};
use crate::cache::DocumentCache;
use crate::ai_integration::{AIIntegrationEngine, AIContext, SkillLevel, ExplanationStyle};
use crate::enhanced_search::{EnhancedSearchSystem, EnhancedSearchRequest, SearchType, LearningSessionRequest, LearningFormat};
use crate::database::{InteractionType, DifficultyLevel, UserContext};
use crate::chat_interface::{ChatInterface, ChatRequest};

pub struct McpServer {
    db: Database,
    cache: DocumentCache,
    ai_engine: AIIntegrationEngine,
    enhanced_search: EnhancedSearchSystem,
    chat_interface: ChatInterface,
}

#[derive(serde::Deserialize)]
struct McpRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(serde::Serialize)]
struct McpResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
}

#[derive(serde::Serialize)]
struct McpError {
    code: i32,
    message: String,
}

impl McpServer {
    pub async fn new(database_url: &str, openai_api_key: Option<String>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let database = Database::new(database_url).await?;
        
        // Initialize all database schemas during startup  
        // 1. Basic tables are already created in Database::new()
        // 2. Initialize model discovery tables if needed
        if let Err(e) = database.initialize_model_discovery().await {
            tracing::info!("Model discovery initialization skipped: {}", e);
        }
        
        // 3. Run advanced features migration
        if let Err(e) = database.run_advanced_features_migration().await {
            tracing::info!("Advanced features migration failed, continuing with basic functionality: {}", e);
        }
        
        let enhanced_search = EnhancedSearchSystem::new(database.clone(), openai_api_key).await
            .map_err(|e| format!("Failed to initialize enhanced search system: {}", e))?;
        
        let chat_interface = ChatInterface::new(enhanced_search.clone()).await
            .map_err(|e| format!("Failed to initialize chat interface: {}", e))?;
        
        Ok(Self { 
            db: database.clone(),
            cache: DocumentCache::new(1000, 500),
            ai_engine: AIIntegrationEngine::new(database.clone()),
            enhanced_search,
            chat_interface,
        })
    }

    pub async fn run(&self) -> Result<()> {
        info!("Starting MCP server on stdio");
        
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            debug!("Received request: {}", line);

            match serde_json::from_str::<McpRequest>(&line) {
                Ok(request) => {
                    let response = self.handle_request(request).await;
                    let response_json = serde_json::to_string(&response)?;
                    println!("{}", response_json);
                    stdout.flush()?;
                }
                Err(e) => {
                    error!("Failed to parse request: {}", e);
                    let error_response = McpResponse {
                        jsonrpc: "2.0".to_string(),
                        id: None,
                        result: None,
                        error: Some(McpError {
                            code: -32700,
                            message: "Parse error".to_string(),
                        }),
                    };
                    let response_json = serde_json::to_string(&error_response)?;
                    println!("{}", response_json);
                    stdout.flush()?;
                }
            }
        }

        Ok(())
    }

    async fn handle_request(&self, request: McpRequest) -> McpResponse {
        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params).await,
            "tools/list" => self.handle_list_tools().await,
            "tools/call" => self.handle_tool_call(request.params).await,
            "resources/list" => self.handle_list_resources().await,
            "resources/read" => self.handle_read_resource(request.params).await,
            _ => Err(anyhow::anyhow!("Unknown method: {}", request.method)),
        };

        match result {
            Ok(result) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: Some(result),
                error: None,
            },
            Err(e) => McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(McpError {
                    code: -32603,
                    message: e.to_string(),
                }),
            },
        }
    }

    async fn handle_initialize(&self, _params: Option<Value>) -> Result<Value> {
        Ok(json!({
            "protocol_version": "1.0.0",
            "server_info": {
                "name": "docs-mcp-server",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": {},
                "resources": {}
            }
        }))
    }

    async fn handle_list_tools(&self) -> Result<Value> {
        Ok(json!({
            "tools": [
                {
                    "name": "search_documentation",
                    "description": "Enhanced search across all documentation sources with smart ranking and caching",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for documentation content"
                            },
                            "source": {
                                "type": "string",
                                "enum": ["rust", "tauri", "react", "typescript", "python", "tailwind", "shadcn"],
                                "description": "Optional: Limit search to specific documentation source"
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["api", "tutorial", "example", "guide"],
                                "description": "Optional: Filter by content type (API reference, tutorials, code examples, etc.)"
                            },
                            "limit": {
                                "type": "integer", 
                                "default": 10,
                                "description": "Maximum number of results to return"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_documentation_page",
                    "description": "Get a specific documentation page by path",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "Documentation source ID (e.g., 'rust-std', 'react-docs')"
                            },
                            "path": {
                                "type": "string",
                                "description": "Path to the specific documentation page"
                            }
                        },
                        "required": ["source_id", "path"]
                    }
                },
                {
                    "name": "list_documentation_sources",
                    "description": "List all available documentation sources, their status, and cache performance",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "get_cache_stats",
                    "description": "Get cache performance statistics and hit ratios",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "get_contextual_help",
                    "description": "Get AI-powered contextual help with adaptive explanations based on your skill level and project context",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Your question or topic you need help with"
                            },
                            "context": {
                                "type": "object",
                                "properties": {
                                    "current_language": {
                                        "type": "string",
                                        "description": "Programming language you're currently working with"
                                    },
                                    "project_type": {
                                        "type": "string",
                                        "description": "Type of project (web app, CLI tool, library, etc.)"
                                    },
                                    "skill_level": {
                                        "type": "string",
                                        "enum": ["beginner", "intermediate", "advanced", "expert"],
                                        "default": "intermediate"
                                    },
                                    "explanation_style": {
                                        "type": "string",
                                        "enum": ["concise", "detailed", "example_focused", "conceptual"],
                                        "default": "detailed"
                                    },
                                    "current_file_path": {
                                        "type": "string",
                                        "description": "Path to file you're currently working on"
                                    }
                                }
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_quality_analytics",
                    "description": "Get content quality analytics and usage statistics for documentation sources",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_type": {
                                "type": "string",
                                "enum": ["rust", "react", "typescript", "python", "tauri", "tailwind", "shadcn"],
                                "description": "Optional: Get analytics for specific documentation type"
                            }
                        }
                    }
                },
                {
                    "name": "enhanced_search",
                    "description": "Perform AI-powered semantic search with advanced ranking and personalized recommendations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "sessionId": {
                                "type": "string",
                                "description": "User session ID for personalization"
                            },
                            "searchType": {
                                "type": "string",
                                "enum": ["semantic", "keyword", "hybrid", "learning"],
                                "description": "Type of search to perform"
                            },
                            "includeSuggestions": {
                                "type": "boolean",
                                "description": "Include content suggestions in response"
                            },
                            "includeLearningPaths": {
                                "type": "boolean",
                                "description": "Include learning path recommendations"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "create_learning_session",
                    "description": "Create a personalized learning session with adaptive content recommendations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Learning topic"
                            },
                            "sessionId": {
                                "type": "string",
                                "description": "User session ID"
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Target difficulty level"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["tutorial", "path", "explore", "recommendations"],
                                "description": "Preferred learning format"
                            },
                            "timeMinutes": {
                                "type": "integer",
                                "description": "Available learning time in minutes"
                            }
                        },
                        "required": ["topic"]
                    }
                },
                {
                    "name": "track_interaction",
                    "description": "Track user interaction with content for personalized recommendations",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sessionId": {
                                "type": "string",
                                "description": "User session ID"
                            },
                            "pageId": {
                                "type": "string",
                                "description": "Document page ID"
                            },
                            "interactionType": {
                                "type": "string",
                                "enum": ["view", "bookmark", "copy", "rate"],
                                "description": "Type of interaction"
                            },
                            "durationSeconds": {
                                "type": "integer",
                                "description": "Time spent on page in seconds"
                            }
                        },
                        "required": ["sessionId", "pageId"]
                    }
                },
                {
                    "name": "get_learning_recommendations",
                    "description": "Get personalized learning recommendations based on user history",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sessionId": {
                                "type": "string",
                                "description": "User session ID"
                            }
                        }
                    }
                },
                {
                    "name": "chat",
                    "description": "Natural language interface for interacting with the documentation system. Supports search, learning, recommendations, and more through conversational commands.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Natural language message or command"
                            },
                            "sessionId": {
                                "type": "string",
                                "description": "User session ID for personalization and context"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for the conversation"
                            }
                        },
                        "required": ["message"]
                    }
                }
            ]
        }))
    }

    async fn handle_tool_call(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let tool_name = params.get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        match tool_name {
            "search_documentation" => self.handle_search_documentation(arguments).await,
            "get_documentation_page" => self.handle_get_documentation_page(arguments).await,
            "list_documentation_sources" => self.handle_list_documentation_sources().await,
            "get_cache_stats" => self.handle_get_cache_stats().await,
            "get_contextual_help" => self.handle_get_contextual_help(arguments).await,
            "get_quality_analytics" => self.handle_get_quality_analytics(arguments).await,
            "enhanced_search" => {
                    let query = arguments.get("query")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Query parameter is required"))?;
                    
                    let session_id = arguments.get("sessionId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default_session")
                        .to_string();
                    
                    let search_type = arguments.get("searchType")
                        .and_then(|v| v.as_str())
                        .map(|s| match s {
                            "semantic" => SearchType::Semantic,
                            "keyword" => SearchType::Keyword,
                            "learning" => SearchType::LearningFocused,
                            _ => SearchType::Hybrid,
                        })
                        .unwrap_or(SearchType::Hybrid);
                    
                    let include_suggestions = arguments.get("includeSuggestions")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    let include_learning_paths = arguments.get("includeLearningPaths")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    let request = EnhancedSearchRequest {
                        query: query.to_string(),
                        session_id,
                        user_context: None, // Could be populated from session data
                        search_type,
                        filters: crate::enhanced_search::SearchFilters {
                            doc_types: vec![],
                            difficulty_levels: vec![],
                            content_types: vec![],
                            date_range: None,
                            exclude_completed: false,
                        },
                        include_suggestions,
                        include_learning_paths,
                    };
                    
                    match self.enhanced_search.enhanced_search(request).await {
                        Ok(response) => {
                            Ok(serde_json::to_value(response)?)
                        }
                        Err(e) => {
                            Err(anyhow::anyhow!("Enhanced search failed: {}", e))
                        }
                    }
                }
                
                "create_learning_session" => {
                    let topic = arguments.get("topic")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Topic parameter is required"))?;
                    
                    let session_id = arguments.get("sessionId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default_session")
                        .to_string();
                    
                    let difficulty = arguments.get("difficulty")
                        .and_then(|v| v.as_str())
                        .map(|s| match s {
                            "beginner" => DifficultyLevel::Beginner,
                            "advanced" => DifficultyLevel::Advanced,
                            _ => DifficultyLevel::Intermediate,
                        })
                        .unwrap_or(DifficultyLevel::Intermediate);
                    
                    let format = arguments.get("format")
                        .and_then(|v| v.as_str())
                        .map(|s| match s {
                            "tutorial" => LearningFormat::InteractiveTutorial,
                            "path" => LearningFormat::StructuredPath,
                            "explore" => LearningFormat::ExploratorySearch,
                            _ => LearningFormat::PersonalizedRecommendations,
                        })
                        .unwrap_or(LearningFormat::PersonalizedRecommendations);
                    
                    let request = LearningSessionRequest {
                        session_id,
                        topic: topic.to_string(),
                        target_difficulty: difficulty,
                        learning_goals: Vec::new(),
                        available_time_minutes: arguments.get("timeMinutes")
                            .and_then(|v| v.as_i64())
                            .map(|t| t as i32),
                        preferred_format: format,
                    };
                    
                    match self.enhanced_search.create_learning_session(request).await {
                        Ok(response) => {
                            Ok(serde_json::to_value(response)?)
                        }
                        Err(e) => {
                            Err(anyhow::anyhow!("Learning session creation failed: {}", e))
                        }
                    }
                }
                
                "track_interaction" => {
                    let session_id = arguments.get("sessionId")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Session ID is required"))?;
                    
                    let page_id = arguments.get("pageId")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Page ID is required"))?;
                    
                    let interaction_type = arguments.get("interactionType")
                        .and_then(|v| v.as_str())
                        .map(|s| match s {
                            "view" => InteractionType::View,
                            "bookmark" => InteractionType::Bookmark,
                            "copy" => InteractionType::Copy,
                            "rate" => InteractionType::Rate,
                            _ => InteractionType::View,
                        })
                        .unwrap_or(InteractionType::View);
                    
                    let duration = arguments.get("durationSeconds")
                        .and_then(|v| v.as_i64())
                        .map(|d| d as i32);
                    
                    match self.enhanced_search.track_interaction(
                        session_id,
                        page_id,
                        interaction_type,
                        duration,
                        None,
                    ).await {
                        Ok(_) => {
                            Ok(serde_json::json!({"status": "success", "message": "Interaction tracked"}))
                        }
                        Err(e) => {
                            Err(anyhow::anyhow!("Failed to track interaction: {}", e))
                        }
                    }
                }
                
                "get_learning_recommendations" => {
                    let session_id = arguments.get("sessionId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default_session");
                    
                    // Build basic user context
                    let user_context = UserContext {
                        session_id: session_id.to_string(),
                        skill_level: Some(DifficultyLevel::Intermediate),
                        preferred_doc_types: vec![],
                        current_learning_paths: vec![],
                        recent_interactions: vec![],
                    };
                    
                    match self.enhanced_search.generate_recommendations(&user_context).await {
                        Ok(recommendations) => {
                            Ok(serde_json::to_value(recommendations)?)
                        }
                        Err(e) => {
                            Err(anyhow::anyhow!("Failed to generate recommendations: {}", e))
                        }
                    }
                }
                
                "chat" => {
                    let message = arguments.get("message")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Message parameter is required"))?;
                    
                    let session_id = arguments.get("sessionId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("default_session")
                        .to_string();
                    
                    let context = arguments.get("context")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    
                    let chat_request = ChatRequest {
                        message: message.to_string(),
                        session_id,
                        context,
                    };
                    
                    match self.chat_interface.process_chat(chat_request).await {
                        Ok(response) => {
                            Ok(serde_json::to_value(response)?)
                        }
                        Err(e) => {
                            Err(anyhow::anyhow!("Chat processing failed: {}", e))
                        }
                    }
                }
            _ => Err(anyhow::anyhow!("Unknown tool: {}", tool_name)),
        }
    }

    async fn handle_search_documentation(&self, arguments: Value) -> Result<Value> {
        let query_text = arguments.get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;

        // Parse optional parameters for enhanced search
        let source = arguments.get("source").and_then(|v| v.as_str());
        let content_type = arguments.get("content_type").and_then(|v| v.as_str());
        let limit = arguments.get("limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(10) as usize;

        // Build enhanced search query
        let mut search_query = SearchQuery {
            query: query_text.to_string(),
            filters: SearchFilters {
                doc_types: None,
                content_types: content_type.map(|ct| vec![ct.to_string()]),
                language: None,
                difficulty_level: None,
                last_updated_after: None,
            },
            ranking_preferences: RankingPreferences {
                prioritize_recent: true,
                prioritize_official: true,
                prioritize_examples: content_type == Some("example"),
                context_similarity_weight: 1.0,
            },
        };

        // Add source filter if specified
        if let Some(source_str) = source {
            let doc_type = match source_str {
                "rust" => Some(DocType::Rust),
                "react" => Some(DocType::React),
                "typescript" => Some(DocType::TypeScript),
                "python" => Some(DocType::Python),
                "tauri" => Some(DocType::Tauri),
                "tailwind" => Some(DocType::Tailwind),
                "shadcn" => Some(DocType::Shadcn),
                _ => None,
            };
            if let Some(dt) = doc_type {
                search_query.filters.doc_types = Some(vec![dt]);
            }
        }

        // Check cache first
        if let Some(cached_results) = self.cache.get_search_results(&search_query) {
            debug!("Returning cached search results for query: {}", query_text);
            let limited_results: Vec<_> = cached_results.into_iter().take(limit).collect();
            return Ok(self.format_search_results(&limited_results, query_text));
        }

        // Perform enhanced search
        let results = self.db.enhanced_search(&search_query).await?;
        
        // Cache the results
        self.cache.cache_search_results(&search_query, results.clone());
        
        let limited_results: Vec<_> = results.into_iter().take(limit).collect();

        Ok(self.format_search_results(&limited_results, query_text))
    }

    fn format_search_results(&self, results: &[crate::database::SearchResult], query: &str) -> Value {
        let content = if results.is_empty() {
            format!("No documentation found for query: '{}'", query)
        } else {
            let mut response = format!("Found {} enhanced documentation result(s) for '{}':\n\n", results.len(), query);
            
            for result in results.iter() {
                response.push_str(&format!(
                    "## {} (Relevance: {:.2})\n**Source**: {} | **URL**: {}\n",
                    result.page.title,
                    result.relevance_score,
                    result.page.source_id,
                    result.page.url
                ));

                // Add matched snippets
                if !result.matched_snippets.is_empty() {
                    response.push_str("\n**Key Excerpts**:\n");
                    for snippet in &result.matched_snippets {
                        response.push_str(&format!("- {}\n", 
                            snippet.content.chars().take(200).collect::<String>()
                        ));
                    }
                }

                response.push_str(&format!("\n**Content Preview**:\n{}\n\n", 
                    result.page.markdown_content.chars().take(400).collect::<String>()
                ));

                // Add related pages
                if !result.related_pages.is_empty() {
                    response.push_str(&format!("**Related Pages**: {}\n", 
                        result.related_pages.join(", ")
                    ));
                }

                response.push_str("---\n\n");
            }
            
            response
        };

        json!({
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })
    }

    async fn handle_get_documentation_page(&self, arguments: Value) -> Result<Value> {
        let source_id = arguments.get("source_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: source_id"))?;

        let path = arguments.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

        if let Some(doc) = self.db.get_document_by_path(source_id, path).await? {
            Ok(json!({
                "content": [
                    {
                        "type": "text",
                        "text": format!(
                            "# {}\n\n**Source**: {} | **URL**: {}\n**Last Updated**: {}\n\n{}",
                            doc.title,
                            doc.source_id,
                            doc.url,
                            doc.last_updated.format("%Y-%m-%d %H:%M:%S UTC"),
                            doc.markdown_content
                        )
                    }
                ]
            }))
        } else {
            Err(anyhow::anyhow!("Documentation page not found: {} at path {}", source_id, path))
        }
    }

    async fn handle_list_documentation_sources(&self) -> Result<Value> {
        let sources = self.db.get_sources().await?;
        let cache_stats = self.cache.get_stats();
        let (page_hit_ratio, query_hit_ratio) = self.cache.get_hit_ratio();
        
        let mut content = "# Available Documentation Sources\n\n".to_string();
        
        // Add cache performance summary
        content.push_str(&format!(
            "## Cache Performance\n\
            - **Page Cache Hit Ratio**: {:.1}%\n\
            - **Query Cache Hit Ratio**: {:.1}%\n\
            - **Total Cached Pages**: {}\n\
            - **Total Cached Queries**: {}\n\n",
            page_hit_ratio * 100.0,
            query_hit_ratio * 100.0,
            cache_stats.total_pages_cached,
            cache_stats.total_queries_cached
        ));

        content.push_str("## Documentation Sources\n\n");
        
        for source in sources {
            // Get document count for this source
            let doc_count = self.db.search_documents("", Some(&source.id)).await?
                .into_iter()
                .filter(|doc| doc.source_id == source.id)
                .count();

            let status = if source.last_updated.is_some() { 
                "✅ Active" 
            } else { 
                "⚠️ Not Updated" 
            };

            content.push_str(&format!(
                "### {} {}\n\
                - **ID**: `{}`\n\
                - **Type**: {}\n\
                - **Base URL**: {}\n\
                - **Last Updated**: {}\n\
                - **Version**: {}\n\
                - **Documents**: {} pages\n\
                - **Search Priority**: {}\n\n",
                source.name,
                status,
                source.id,
                source.doc_type.as_str(),
                source.base_url,
                source.last_updated
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                    .unwrap_or_else(|| "Never".to_string()),
                source.version.unwrap_or_else(|| "Latest".to_string()),
                doc_count,
                match source.doc_type {
                    crate::database::DocType::React | crate::database::DocType::Shadcn => "High",
                    crate::database::DocType::Rust | crate::database::DocType::TypeScript => "High", 
                    crate::database::DocType::Tailwind => "Medium",
                    _ => "Standard"
                }
            ));
        }

        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        }))
    }

    async fn handle_list_resources(&self) -> Result<Value> {
        Ok(json!({
            "resources": [
                {
                    "uri": "docs://rust/std",
                    "name": "Rust Standard Library Documentation",
                    "description": "Complete Rust standard library documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://rust/book",
                    "name": "The Rust Programming Language Book",
                    "description": "The official Rust programming language book",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://tauri",
                    "name": "Tauri Documentation",
                    "description": "Tauri framework documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://react",
                    "name": "React Documentation",
                    "description": "React library documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://typescript",
                    "name": "TypeScript Documentation",
                    "description": "TypeScript language documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://python",
                    "name": "Python Documentation",
                    "description": "Python language documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://tailwind",
                    "name": "Tailwind CSS Documentation",
                    "description": "Tailwind CSS utility-first framework documentation",
                    "mimeType": "text/markdown"
                },
                {
                    "uri": "docs://shadcn",
                    "name": "shadcn/ui Documentation", 
                    "description": "shadcn/ui component library documentation",
                    "mimeType": "text/markdown"
                }
            ]
        }))
    }

    async fn handle_read_resource(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow::anyhow!("Missing parameters"))?;
        let uri = params.get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing URI parameter"))?;

        // Parse the URI to determine what documentation to return
        let (doc_type, section) = match uri {
            uri if uri.starts_with("docs://rust/std") => ("rust", Some("rust-std")),
            uri if uri.starts_with("docs://rust/book") => ("rust", Some("rust-book")),
            uri if uri.starts_with("docs://tauri") => ("tauri", None),
            uri if uri.starts_with("docs://react") => ("react", None),
            uri if uri.starts_with("docs://typescript") => ("typescript", None),
            uri if uri.starts_with("docs://python") => ("python", None),
            uri if uri.starts_with("docs://tailwind") => ("tailwind", None),
            uri if uri.starts_with("docs://shadcn") => ("shadcn", None),
            _ => return Err(anyhow::anyhow!("Unknown resource URI: {}", uri)),
        };

        // Get all documents for the requested type/section
        let results = if let Some(section_id) = section {
            // Get documents for specific source ID
            self.db.search_documents("", Some(doc_type)).await?
                .into_iter()
                .filter(|doc| doc.source_id == section_id)
                .collect()
        } else {
            // Get all documents for doc type
            self.db.search_documents("", Some(doc_type)).await?
        };

        let mut content = format!("# {} Documentation\n\n", doc_type.to_uppercase());
        
        for doc in results {
            content.push_str(&format!(
                "## {}\n**URL**: {}\n\n{}\n\n---\n\n",
                doc.title,
                doc.url,
                doc.markdown_content
            ));
        }

        Ok(json!({
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": content
                }
            ]
        }))
    }

    async fn handle_get_cache_stats(&self) -> Result<Value> {
        let stats = self.cache.get_stats();
        let (page_hit_ratio, query_hit_ratio) = self.cache.get_hit_ratio();

        let content = format!(
            "# Cache Performance Statistics\n\n\
            ## Page Cache\n\
            - **Pages Cached**: {}\n\
            - **Cache Hits**: {}\n\
            - **Cache Misses**: {}\n\
            - **Hit Ratio**: {:.2}%\n\n\
            ## Query Cache\n\
            - **Queries Cached**: {}\n\
            - **Cache Hits**: {}\n\
            - **Cache Misses**: {}\n\
            - **Hit Ratio**: {:.2}%\n\n\
            ## Performance Impact\n\
            - Reduced database queries: {}\n\
            - Faster response times for cached content\n\
            - Memory efficiency with LRU eviction\n",
            stats.total_pages_cached,
            stats.page_hits,
            stats.page_misses,
            page_hit_ratio * 100.0,
            stats.total_queries_cached,
            stats.query_hits,
            stats.query_misses,
            query_hit_ratio * 100.0,
            stats.page_hits + stats.query_hits
        );

        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        }))
    }

    async fn handle_get_contextual_help(&self, arguments: Value) -> Result<Value> {
        let query = arguments.get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing query parameter"))?;

        // Parse context information
        let context_obj = arguments.get("context").cloned().unwrap_or(json!({}));
        let context = self.parse_ai_context(context_obj)?;

        info!("Generating contextual help for query: {} with skill level: {:?}", 
              query, context.user_skill_level);

        // Generate AI-powered contextual response
        let enhanced_response = self.ai_engine.generate_contextual_response(query, &context).await?;

        // Format the response
        let mut content = format!("# 🤖 AI-Powered Help: {}\n\n", enhanced_response.primary_content.title);
        
        // Add difficulty notes if present
        if let Some(difficulty_notes) = &enhanced_response.difficulty_notes {
            content.push_str(&format!("## 🎯 Skill Level Notes\n{}\n\n", difficulty_notes));
        }
        
        // Add primary content
        content.push_str(&format!("## 📖 Main Content\n{}\n\n", 
                                 enhanced_response.primary_content.markdown_content));
        
        // Add code examples if available
        if !enhanced_response.related_examples.is_empty() {
            content.push_str("## 💻 Code Examples\n\n");
            for (idx, example) in enhanced_response.related_examples.iter().enumerate() {
                content.push_str(&format!(
                    "### {}\n**Complexity**: {:?} | **Language**: {}\n\n{}\n\n```{}\n{}\n```\n\n",
                    example.title,
                    example.complexity_level,
                    example.language,
                    example.explanation,
                    example.language,
                    example.code
                ));
            }
        }
        
        // Add concept map
        if !enhanced_response.conceptual_context.is_empty() {
            content.push_str("## 🔗 Related Concepts\n\n");
            for concept_link in &enhanced_response.conceptual_context {
                content.push_str(&format!(
                    "- **{}** → {} ({:?})\n",
                    concept_link.concept,
                    concept_link.related_page_id,
                    concept_link.relationship_type
                ));
            }
            content.push_str("\n");
        }
        
        // Add common pitfalls
        if !enhanced_response.common_pitfalls.is_empty() {
            content.push_str("## ⚠️ Common Pitfalls\n\n");
            for pitfall in &enhanced_response.common_pitfalls {
                content.push_str(&format!("- {}\n", pitfall));
            }
            content.push_str("\n");
        }
        
        // Add best practices
        if !enhanced_response.best_practices.is_empty() {
            content.push_str("## ✅ Best Practices\n\n");
            for practice in &enhanced_response.best_practices {
                content.push_str(&format!("- {}\n", practice));
            }
            content.push_str("\n");
        }
        
        // Add source reference
        content.push_str(&format!("\n---\n\n📄 **Source**: [{}]({})\n", 
                                 enhanced_response.primary_content.title,
                                 enhanced_response.primary_content.url));

        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        }))
    }
    
    async fn handle_get_quality_analytics(&self, arguments: Value) -> Result<Value> {
        info!("Generating quality analytics report");
        
        let specific_doc_type = arguments.get("doc_type")
            .and_then(|v| v.as_str())
            .and_then(|s| self.parse_doc_type(s));
        
        let mut content = String::from("# 📊 Documentation Quality Analytics\n\n");
        
        let doc_types = if let Some(doc_type) = specific_doc_type {
            vec![doc_type]
        } else {
            vec![DocType::Rust, DocType::React, DocType::TypeScript, DocType::Python, 
                 DocType::Tauri, DocType::Tailwind, DocType::Shadcn]
        };
        
        for doc_type in doc_types {
            content.push_str(&format!("## {:?} Documentation\n\n", doc_type));
            
            // Get quality metrics (simplified version since we don't have full scheduler integration yet)
            let page_count = self.db.get_page_count_for_type(&doc_type).await.unwrap_or(0);
            let last_update = self.db.get_last_update_time_for_type(&doc_type).await
                .unwrap_or(None);
            
            // Calculate basic metrics
            let freshness_score = if let Some(last_update) = last_update {
                let hours_since_update = (chrono::Utc::now() - last_update).num_hours();
                let expected_hours = match doc_type {
                    DocType::React | DocType::Shadcn => 6,
                    DocType::TypeScript | DocType::Rust | DocType::Tailwind => 12,
                    _ => 24,
                };
                ((expected_hours * 2 - hours_since_update.min(expected_hours * 2)) as f32 / (expected_hours * 2) as f32).max(0.0)
            } else {
                0.0
            };
            
            let expected_pages = match doc_type {
                DocType::Rust => 500,
                DocType::React => 200,
                DocType::TypeScript => 300,
                DocType::Python => 600,
                DocType::Tauri => 100,
                DocType::Tailwind => 150,
                DocType::Shadcn => 80,
            };
            let completeness_score = (page_count as f32 / expected_pages as f32).min(1.0);
            
            content.push_str(&format!(
                "### Quality Metrics\n\
                - **📄 Pages Indexed**: {}\n\
                - **📊 Completeness**: {:.1}% ({}/{})\n\
                - **🕒 Freshness**: {:.1}%\n\
                - **📅 Last Updated**: {}\n\n",
                page_count,
                completeness_score * 100.0,
                page_count,
                expected_pages,
                freshness_score * 100.0,
                last_update.map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                          .unwrap_or_else(|| "Never".to_string())
            ));
            
            // Quality indicators
            let overall_quality = (freshness_score + completeness_score) / 2.0;
            let quality_indicator = if overall_quality > 0.8 {
                "🟢 Excellent"
            } else if overall_quality > 0.6 {
                "🟡 Good"
            } else if overall_quality > 0.4 {
                "🟠 Needs Attention"
            } else {
                "🔴 Poor"
            };
            
            content.push_str(&format!("**Overall Quality**: {} ({:.1}%)\n\n", 
                                     quality_indicator, overall_quality * 100.0));
        }
        
        // Add recommendations
        content.push_str("## 🎯 Recommendations\n\n");
        content.push_str("- 🔄 High-usage documentation should be updated more frequently\n");
        content.push_str("- 📝 Focus on expanding incomplete documentation sections\n");
        content.push_str("- 🧹 Regular maintenance to fix broken links and outdated examples\n");
        content.push_str("- 📊 Monitor user feedback and search patterns for priority areas\n");
        
        Ok(json!({
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        }))
    }
    
    fn parse_ai_context(&self, context_obj: Value) -> Result<AIContext> {
        let current_language = context_obj.get("current_language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let project_type = context_obj.get("project_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let current_file_path = context_obj.get("current_file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let skill_level = context_obj.get("skill_level")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "beginner" => SkillLevel::Beginner,
                "intermediate" => SkillLevel::Intermediate,
                "advanced" => SkillLevel::Advanced,
                "expert" => SkillLevel::Expert,
                _ => SkillLevel::Intermediate,
            })
            .unwrap_or(SkillLevel::Intermediate);
        
        let explanation_style = context_obj.get("explanation_style")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "concise" => ExplanationStyle::Concise,
                "detailed" => ExplanationStyle::Detailed,
                "example_focused" => ExplanationStyle::ExampleFocused,
                "conceptual" => ExplanationStyle::Conceptual,
                _ => ExplanationStyle::Detailed,
            })
            .unwrap_or(ExplanationStyle::Detailed);
        
        Ok(AIContext {
            current_language,
            project_type,
            current_file_path,
            recent_queries: Vec::new(), // Could be populated from session history
            user_skill_level: skill_level,
            preferred_explanation_style: explanation_style,
        })
    }
    
    fn parse_doc_type(&self, doc_type_str: &str) -> Option<DocType> {
        match doc_type_str.to_lowercase().as_str() {
            "rust" => Some(DocType::Rust),
            "react" => Some(DocType::React),
            "typescript" => Some(DocType::TypeScript),
            "python" => Some(DocType::Python),
            "tauri" => Some(DocType::Tauri),
            "tailwind" => Some(DocType::Tailwind),
            "shadcn" => Some(DocType::Shadcn),
            _ => None,
        }
    }
}
