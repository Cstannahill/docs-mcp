use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;

use crate::enhanced_search::{EnhancedSearchSystem, EnhancedSearchRequest, SearchType, LearningSessionRequest, LearningFormat};
use crate::database::{InteractionType, DifficultyLevel, UserContext, DocType};
use crate::web_search::{WebSearchEngine, SearchRequest, SearchType as WebSearchType, SearchFilters as WebSearchFilters};
use crate::file_manager::FileManager;

#[derive(Clone)]
pub struct ChatInterface {
    enhanced_search: EnhancedSearchSystem,
    web_search: WebSearchEngine,
    file_manager: FileManager,
    command_patterns: HashMap<String, Regex>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub session_id: String,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub response: String,
    pub action_taken: Option<String>,
    pub results: Option<serde_json::Value>,
    pub suggestions: Vec<String>,
    pub session_id: String,
}

#[derive(Debug, Clone)]
pub enum ParsedCommand {
    Search {
        query: String,
        search_type: SearchType,
        include_suggestions: bool,
    },
    CreateLearning {
        topic: String,
        difficulty: DifficultyLevel,
        format: LearningFormat,
        time_minutes: Option<i32>,
    },
    GetRecommendations,
    TrackInteraction {
        page_id: String,
        interaction_type: InteractionType,
        duration: Option<i32>,
    },
    Help,
    ListDocuments {
        doc_type: Option<DocType>,
        limit: Option<i32>,
    },
    ExplainTopic {
        topic: String,
        level: DifficultyLevel,
    },
    FindRelated {
        page_id: String,
    },
    GetLearningProgress,
    // Web search commands
    WebSearch {
        query: String,
        search_type: WebSearchType,
        max_results: Option<usize>,
    },
    WebSearchProgramming {
        query: String,
        language: Option<String>,
    },
    WebSearchDocs {
        query: String,
        technology: Option<String>,
    },
    GetPageSummary {
        url: String,
    },
    // File operation commands
    ReadFile {
        path: String,
    },
    WriteFile {
        path: String,
        content: String,
        append: bool,
    },
    ListDirectory {
        path: String,
        recursive: bool,
    },
    FindFiles {
        pattern: String,
        directory: Option<String>,
    },
    SearchFileContent {
        query: String,
        directory: Option<String>,
        file_extensions: Option<Vec<String>>,
    },
    Unknown(String),
}

impl ChatInterface {
    pub async fn new(enhanced_search: EnhancedSearchSystem) -> Result<Self> {
        let mut command_patterns = HashMap::new();
        
        // Search patterns
        command_patterns.insert("search".to_string(), 
            Regex::new(r"(?i)(search|find|look for|query)\s+(?:for\s+)?(.+)").unwrap());
        command_patterns.insert("semantic_search".to_string(), 
            Regex::new(r"(?i)(semantic|meaning|similar)\s+search\s+(?:for\s+)?(.+)").unwrap());
        command_patterns.insert("learning_search".to_string(), 
            Regex::new(r"(?i)(learn|tutorial|guide)\s+(?:about\s+)?(.+)").unwrap());
        
        // Web search patterns
        command_patterns.insert("web_search".to_string(), 
            Regex::new(r"(?i)(web search|google|search web|search online)\s+(?:for\s+)?(.+)").unwrap());
        command_patterns.insert("programming_search".to_string(), 
            Regex::new(r"(?i)(programming|code)\s+search\s+(?:for\s+)?(.+)").unwrap());
        command_patterns.insert("docs_search".to_string(), 
            Regex::new(r"(?i)(documentation|docs)\s+search\s+(?:for\s+)?(.+)").unwrap());
        command_patterns.insert("page_summary".to_string(), 
            Regex::new(r"(?i)(summarize|summary of)\s+(https?://\S+)").unwrap());
        
        // File operation patterns
        command_patterns.insert("read_file".to_string(), 
            Regex::new(r"(?i)(read|show|display|cat)\s+(?:file\s+)?(.+)").unwrap());
        command_patterns.insert("write_file".to_string(), 
            Regex::new(r"(?i)(write|create|save)\s+(?:file\s+)?(.+)").unwrap());
        command_patterns.insert("list_dir".to_string(), 
            Regex::new(r"(?i)(list|ls|dir|show)\s+(?:directory|folder|dir)\s+(.+)").unwrap());
        command_patterns.insert("find_files".to_string(), 
            Regex::new(r"(?i)(find|locate)\s+(?:files?\s+)?(.+)").unwrap());
        command_patterns.insert("search_files".to_string(), 
            Regex::new(r"(?i)(search|grep)\s+(?:in\s+)?files?\s+(?:for\s+)?(.+)").unwrap());
        
        // Learning patterns
        command_patterns.insert("create_tutorial".to_string(), 
            Regex::new(r"(?i)(create|make|generate)\s+(tutorial|learning path|course)\s+(?:for\s+|about\s+)?(.+)").unwrap());
        command_patterns.insert("learn_topic".to_string(), 
            Regex::new(r"(?i)(teach me|help me learn|i want to learn)\s+(?:about\s+)?(.+)").unwrap());
        
        // Interaction patterns
        command_patterns.insert("bookmark".to_string(), 
            Regex::new(r"(?i)(bookmark|save|mark)\s+(.+)").unwrap());
        command_patterns.insert("rate".to_string(), 
            Regex::new(r"(?i)(rate|review)\s+(.+)").unwrap());
        
        // Information patterns
        command_patterns.insert("list_docs".to_string(), 
            Regex::new(r"(?i)(list|show|what)\s+(documents|docs|pages)").unwrap());
        command_patterns.insert("explain".to_string(), 
            Regex::new(r"(?i)(explain|what is|define)\s+(.+)").unwrap());
        command_patterns.insert("help".to_string(), 
            Regex::new(r"(?i)(help|commands|what can you do)").unwrap());
        
        // Progress patterns
        command_patterns.insert("progress".to_string(), 
            Regex::new(r"(?i)(progress|status|where am i)").unwrap());
        command_patterns.insert("recommendations".to_string(), 
            Regex::new(r"(?i)(recommend|suggest|what should i|next)").unwrap());

        Ok(Self {
            enhanced_search,
            web_search: WebSearchEngine::new(),
            file_manager: FileManager::new("/workspaces/docs-mcp".into()),
            command_patterns,
        })
    }

    pub async fn process_chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let parsed_command = self.parse_message(&request.message);
        
        match parsed_command {
            ParsedCommand::Search { query, search_type, include_suggestions } => {
                self.handle_search(request.session_id, query, search_type, include_suggestions).await
            }
            ParsedCommand::CreateLearning { topic, difficulty, format, time_minutes } => {
                self.handle_create_learning(request.session_id, topic, difficulty, format, time_minutes).await
            }
            ParsedCommand::GetRecommendations => {
                self.handle_get_recommendations(request.session_id).await
            }
            ParsedCommand::TrackInteraction { page_id, interaction_type, duration } => {
                self.handle_track_interaction(request.session_id, page_id, interaction_type, duration).await
            }
            ParsedCommand::Help => {
                self.handle_help().await
            }
            ParsedCommand::ListDocuments { doc_type, limit } => {
                self.handle_list_documents(doc_type, limit).await
            }
            ParsedCommand::ExplainTopic { topic, level } => {
                self.handle_explain_topic(request.session_id, topic, level).await
            }
            ParsedCommand::FindRelated { page_id } => {
                self.handle_find_related(page_id).await
            }
            ParsedCommand::GetLearningProgress => {
                self.handle_get_progress(request.session_id).await
            }
            // Web search commands
            ParsedCommand::WebSearch { query, search_type, max_results } => {
                self.handle_web_search(query, search_type, max_results).await
            }
            ParsedCommand::WebSearchProgramming { query, language } => {
                self.handle_web_search_programming(query, language).await
            }
            ParsedCommand::WebSearchDocs { query, technology } => {
                self.handle_web_search_docs(query, technology).await
            }
            ParsedCommand::GetPageSummary { url } => {
                self.handle_page_summary(url).await
            }
            // File operation commands
            ParsedCommand::ReadFile { path } => {
                self.handle_read_file(path).await
            }
            ParsedCommand::WriteFile { path, content, append } => {
                self.handle_write_file(path, content, append).await
            }
            ParsedCommand::ListDirectory { path, recursive } => {
                self.handle_list_directory(path, recursive).await
            }
            ParsedCommand::FindFiles { pattern, directory } => {
                self.handle_find_files(pattern, directory).await
            }
            ParsedCommand::SearchFileContent { query, directory, file_extensions } => {
                self.handle_search_file_content(query, directory, file_extensions).await
            }
            ParsedCommand::Unknown(msg) => {
                self.handle_unknown(msg).await
            }
        }
    }

    fn parse_message(&self, message: &str) -> ParsedCommand {
        let message = message.trim();
        
        // Check for specific command patterns
        for (command_type, pattern) in &self.command_patterns {
            if let Some(captures) = pattern.captures(message) {
                match command_type.as_str() {
                    "search" => {
                        if let Some(query) = captures.get(2) {
                            return ParsedCommand::Search {
                                query: query.as_str().to_string(),
                                search_type: SearchType::Hybrid,
                                include_suggestions: true,
                            };
                        }
                    }
                    "semantic_search" => {
                        if let Some(query) = captures.get(2) {
                            return ParsedCommand::Search {
                                query: query.as_str().to_string(),
                                search_type: SearchType::Semantic,
                                include_suggestions: true,
                            };
                        }
                    }
                    "learning_search" => {
                        if let Some(query) = captures.get(2) {
                            return ParsedCommand::Search {
                                query: query.as_str().to_string(),
                                search_type: SearchType::LearningFocused,
                                include_suggestions: true,
                            };
                        }
                    }
                    "create_tutorial" | "learn_topic" => {
                        if let Some(topic) = captures.get(3).or_else(|| captures.get(2)) {
                            let topic_str = topic.as_str();
                            let difficulty = self.infer_difficulty(message);
                            let format = self.infer_learning_format(message);
                            let time = self.extract_time_minutes(message);
                            
                            return ParsedCommand::CreateLearning {
                                topic: topic_str.to_string(),
                                difficulty,
                                format,
                                time_minutes: time,
                            };
                        }
                    }
                    "explain" => {
                        if let Some(topic) = captures.get(2) {
                            let difficulty = self.infer_difficulty(message);
                            return ParsedCommand::ExplainTopic {
                                topic: topic.as_str().to_string(),
                                level: difficulty,
                            };
                        }
                    }
                    "help" => return ParsedCommand::Help,
                    "list_docs" => {
                        let doc_type = self.infer_doc_type(message);
                        let limit = self.extract_limit(message);
                        return ParsedCommand::ListDocuments { doc_type, limit };
                    }
                    "progress" => return ParsedCommand::GetLearningProgress,
                    "recommendations" => return ParsedCommand::GetRecommendations,
                    // Web search commands
                    "web_search" => {
                        if let Some(query) = captures.get(2) {
                            return ParsedCommand::WebSearch {
                                query: query.as_str().to_string(),
                                search_type: WebSearchType::General,
                                max_results: None,
                            };
                        }
                    }
                    "programming_search" => {
                        if let Some(query) = captures.get(2) {
                            let language = self.extract_programming_language(message);
                            return ParsedCommand::WebSearchProgramming {
                                query: query.as_str().to_string(),
                                language,
                            };
                        }
                    }
                    "docs_search" => {
                        if let Some(query) = captures.get(2) {
                            let technology = self.extract_technology(message);
                            return ParsedCommand::WebSearchDocs {
                                query: query.as_str().to_string(),
                                technology,
                            };
                        }
                    }
                    "page_summary" => {
                        if let Some(url) = captures.get(2) {
                            return ParsedCommand::GetPageSummary {
                                url: url.as_str().to_string(),
                            };
                        }
                    }
                    // File operation commands
                    "read_file" => {
                        if let Some(path) = captures.get(2) {
                            return ParsedCommand::ReadFile {
                                path: path.as_str().to_string(),
                            };
                        }
                    }
                    "write_file" => {
                        if let Some(path) = captures.get(2) {
                            // Simple parsing for write command (could be improved)
                            return ParsedCommand::WriteFile {
                                path: path.as_str().to_string(),
                                content: "".to_string(), // Content would be provided in follow-up
                                append: message.to_lowercase().contains("append"),
                            };
                        }
                    }
                    "list_dir" => {
                        if let Some(path) = captures.get(2) {
                            return ParsedCommand::ListDirectory {
                                path: path.as_str().to_string(),
                                recursive: message.to_lowercase().contains("recursive") || message.to_lowercase().contains("-r"),
                            };
                        }
                    }
                    "find_files" => {
                        if let Some(pattern) = captures.get(2) {
                            let directory = self.extract_directory(message);
                            return ParsedCommand::FindFiles {
                                pattern: pattern.as_str().to_string(),
                                directory,
                            };
                        }
                    }
                    "search_files" => {
                        if let Some(query) = captures.get(2) {
                            let directory = self.extract_directory(message);
                            let extensions = self.extract_file_extensions(message);
                            return ParsedCommand::SearchFileContent {
                                query: query.as_str().to_string(),
                                directory,
                                file_extensions: extensions,
                            };
                        }
                    }
                    _ => {}
                }
            }
        }

        ParsedCommand::Unknown(message.to_string())
    }

    async fn handle_search(&self, session_id: String, query: String, search_type: SearchType, include_suggestions: bool) -> Result<ChatResponse> {
        let search_request = EnhancedSearchRequest {
            query: query.clone(),
            session_id: session_id.clone(),
            user_context: None,
            search_type: search_type.clone(),
            filters: crate::enhanced_search::SearchFilters {
                doc_types: vec![],
                difficulty_levels: vec![],
                content_types: vec![],
                date_range: None,
                exclude_completed: false,
            },
            include_suggestions,
            include_learning_paths: true,
        };

        let search_response = self.enhanced_search.enhanced_search(search_request).await?;
        
        let mut response_text = format!("Found {} results for \"{}\":\n\n", search_response.total_results, query);
        
        // Add top results
        for (i, result) in search_response.results.iter().take(5).enumerate() {
            response_text.push_str(&format!(
                "{}. **{}**\n   URL: {}\n   Relevance: {:.2}\n   Type: {:?}\n\n",
                i + 1,
                result.page.title,
                result.page.url,
                result.ranking_factors.text_relevance,
                "Documentation" // Simplified for now
            ));
        }

        let mut suggestions = Vec::new();
        if let Some(expansion) = &search_response.query_expansion {
            suggestions.push(format!("Try searching: {}", expansion));
        }
        
        for topic in &search_response.related_topics {
            suggestions.push(format!("Related topic: {}", topic));
        }

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some(format!("Performed {} search", match search_type {
                SearchType::Semantic => "semantic",
                SearchType::Keyword => "keyword", 
                SearchType::Hybrid => "hybrid",
                SearchType::LearningFocused => "learning-focused",
            })),
            results: Some(serde_json::to_value(&search_response)?),
            suggestions,
            session_id,
        })
    }

    async fn handle_create_learning(&self, session_id: String, topic: String, difficulty: DifficultyLevel, format: LearningFormat, time_minutes: Option<i32>) -> Result<ChatResponse> {
        let learning_request = LearningSessionRequest {
            session_id: session_id.clone(),
            topic: topic.clone(),
            target_difficulty: difficulty.clone(),
            learning_goals: Vec::new(),
            available_time_minutes: time_minutes,
            preferred_format: format.clone(),
        };

        let learning_response = self.enhanced_search.create_learning_session(learning_request).await?;
        
        let mut response_text = format!("Created {} learning session for \"{}\" at {:?} level:\n\n", 
            match format {
                LearningFormat::StructuredPath => "structured learning path",
                LearningFormat::InteractiveTutorial => "interactive tutorial",
                LearningFormat::ExploratorySearch => "exploratory learning",
                LearningFormat::PersonalizedRecommendations => "personalized recommendations",
            },
            topic,
            difficulty
        );

        if let Some(path) = &learning_response.recommended_path {
            response_text.push_str(&format!(
                "**Learning Path: {}**\n{}\nEstimated time: {} minutes\n\n",
                path.title,
                path.description,
                path.estimated_duration_minutes
            ));
        }

        if let Some(tutorial) = &learning_response.interactive_tutorial {
            response_text.push_str(&format!(
                "**Interactive Tutorial: {}**\n{}\n{} steps, estimated time: {} minutes\n\n",
                tutorial.title,
                tutorial.description,
                tutorial.steps.len(),
                tutorial.estimated_duration_minutes
            ));
        }

        let suggestions = learning_response.personalization_notes.clone();

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some("Created learning session".to_string()),
            results: Some(serde_json::to_value(&learning_response)?),
            suggestions,
            session_id,
        })
    }

    async fn handle_get_recommendations(&self, session_id: String) -> Result<ChatResponse> {
        let user_context = UserContext {
            session_id: session_id.clone(),
            skill_level: Some(DifficultyLevel::Intermediate),
            preferred_doc_types: Vec::new(),
            current_learning_paths: Vec::new(),
            recent_interactions: Vec::new(),
        };

        let recommendations = self.enhanced_search.generate_recommendations(&user_context).await?;
        
        let mut response_text = "Here are your personalized recommendations:\n\n".to_string();
        
        if !recommendations.suggested_content.is_empty() {
            response_text.push_str("**Suggested Content:**\n");
            for (i, suggestion) in recommendations.suggested_content.iter().take(5).enumerate() {
                response_text.push_str(&format!(
                    "{}. {} (confidence: {:.2})\n",
                    i + 1,
                    suggestion.reason.as_ref().unwrap_or(&"Recommended for you".to_string()),
                    suggestion.confidence_score
                ));
            }
            response_text.push('\n');
        }

        if !recommendations.skill_gaps.is_empty() {
            response_text.push_str("**Areas for Improvement:**\n");
            for gap in &recommendations.skill_gaps {
                response_text.push_str(&format!(
                    "- {}: Current level {:.1}/5.0, Target {:.1}/5.0\n",
                    gap.topic,
                    gap.current_level * 5.0,
                    gap.target_level * 5.0
                ));
            }
            response_text.push('\n');
        }

        if !recommendations.next_steps.is_empty() {
            response_text.push_str("**Next Steps:**\n");
            for step in &recommendations.next_steps {
                response_text.push_str(&format!("- {}\n", step));
            }
        }

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some("Generated personalized recommendations".to_string()),
            results: Some(serde_json::to_value(&recommendations)?),
            suggestions: Vec::new(),
            session_id,
        })
    }

    async fn handle_track_interaction(&self, session_id: String, page_id: String, interaction_type: InteractionType, duration: Option<i32>) -> Result<ChatResponse> {
        self.enhanced_search.track_interaction(
            &session_id,
            &page_id,
            interaction_type.clone(),
            duration,
            None,
        ).await?;

        let response_text = format!(
            "Tracked {:?} interaction with page {} for {} seconds",
            interaction_type,
            page_id,
            duration.unwrap_or(0)
        );

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some("Tracked interaction".to_string()),
            results: None,
            suggestions: vec!["Ask for recommendations to see updated suggestions".to_string()],
            session_id,
        })
    }

    async fn handle_help(&self) -> Result<ChatResponse> {
        let help_text = r#"
**Documentation Assistant Commands:**

**Search & Discovery:**
- "search for [topic]" - Find relevant documentation
- "semantic search [topic]" - Find conceptually similar content  
- "learn about [topic]" - Get learning-focused results
- "list documents" - Show available documentation
- "explain [concept]" - Get detailed explanations

**Learning & Tutorials:**
- "create tutorial for [topic]" - Generate interactive tutorial
- "teach me [topic]" - Create personalized learning path
- "help me learn [topic] at beginner/intermediate/advanced level"
- "create course for [topic] in 30 minutes"

**Progress & Recommendations:**
- "get recommendations" - Personalized content suggestions
- "show my progress" - Learning progress overview
- "what should I learn next?" - Adaptive next steps

**Content Interaction:**
- "bookmark [page_id]" - Save page for later
- "rate [page_id]" - Provide feedback
- "find related to [page_id]" - Discover connected content

**Examples:**
- "search for rust async programming"
- "create beginner tutorial for TypeScript"
- "semantic search functional programming concepts"
- "teach me Python web development"
- "get my learning recommendations"

The system learns from your interactions to provide increasingly personalized recommendations!
        "#;

        Ok(ChatResponse {
            response: help_text.to_string(),
            action_taken: Some("Displayed help information".to_string()),
            results: None,
            suggestions: vec![
                "Try: search for rust".to_string(),
                "Try: teach me python basics".to_string(),
                "Try: get recommendations".to_string(),
            ],
            session_id: "help".to_string(),
        })
    }

    async fn handle_list_documents(&self, doc_type: Option<DocType>, limit: Option<i32>) -> Result<ChatResponse> {
        // This would query the database for available documents
        let response_text = format!(
            "Available documentation types:\n\n\
            - **Rust**: The Rust Programming Language, Rust Standard Library, Cargo\n\
            - **TypeScript**: TypeScript Handbook, npm documentation\n\
            - **Python**: Python documentation, pip packages\n\
            - **React**: React documentation and guides\n\
            - **Tauri**: Tauri framework documentation\n\n\
            Total coverage: 8,563+ pages across all languages and frameworks.\n\n\
            Use 'search for [topic]' to find specific content."
        );

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some("Listed available documents".to_string()),
            results: None,
            suggestions: vec![
                "search for rust getting started".to_string(),
                "search for typescript basics".to_string(),
                "search for python tutorial".to_string(),
            ],
            session_id: "list".to_string(),
        })
    }

    async fn handle_explain_topic(&self, session_id: String, topic: String, level: DifficultyLevel) -> Result<ChatResponse> {
        // Perform a search to find explanatory content
        let search_result = self.handle_search(
            session_id.clone(),
            topic.clone(),
            SearchType::LearningFocused,
            true,
        ).await?;

        let mut response_text = format!("Here's what I found about \"{}\" at {:?} level:\n\n", topic, level);
        
        if let Some(results) = &search_result.results {
            if let Ok(search_response) = serde_json::from_value::<crate::enhanced_search::EnhancedSearchResponse>(results.clone()) {
                if let Some(first_result) = search_response.results.first() {
                    // Extract a snippet for explanation
                    let content_preview = if first_result.page.content.len() > 300 {
                        format!("{}...", &first_result.page.content[..300])
                    } else {
                        first_result.page.content.clone()
                    };
                    
                    response_text.push_str(&format!(
                        "**{}**\n{}\n\nSource: {}\n\n",
                        first_result.page.title,
                        content_preview,
                        first_result.page.url
                    ));
                }
            }
        }

        response_text.push_str("Would you like me to create a learning path or tutorial for this topic?");

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some(format!("Explained topic: {}", topic)),
            results: search_result.results,
            suggestions: vec![
                format!("create tutorial for {}", topic),
                format!("learn more about {}", topic),
                "get recommendations".to_string(),
            ],
            session_id,
        })
    }

    async fn handle_find_related(&self, page_id: String) -> Result<ChatResponse> {
        // This would use the document relationships in the database
        let response_text = format!(
            "Finding related content for page {}...\n\n\
            This feature uses our document relationship mapping to find:\n\
            - Prerequisites\n\
            - Follow-up topics\n\
            - Similar concepts\n\
            - Examples and applications\n\n\
            Use the enhanced search to explore related topics!",
            page_id
        );

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some("Found related content".to_string()),
            results: None,
            suggestions: vec![
                "search for related concepts".to_string(),
                "get recommendations".to_string(),
            ],
            session_id: "related".to_string(),
        })
    }

    async fn handle_get_progress(&self, session_id: String) -> Result<ChatResponse> {
        let response_text = format!(
            "Learning Progress for session {}:\n\n\
            📊 **Session Statistics:**\n\
            - Active learning paths: Check your current sessions\n\
            - Completed tutorials: Track your achievements\n\
            - Interaction history: Monitor your engagement\n\
            - Skill assessments: See your progress levels\n\n\
            Use 'get recommendations' to see personalized next steps based on your progress!",
            session_id
        );

        Ok(ChatResponse {
            response: response_text,
            action_taken: Some("Retrieved learning progress".to_string()),
            results: None,
            suggestions: vec![
                "get recommendations".to_string(),
                "what should I learn next?".to_string(),
            ],
            session_id,
        })
    }

    async fn handle_unknown(&self, message: String) -> Result<ChatResponse> {
        let response_text = format!(
            "I didn't understand: \"{}\"\n\n\
            I can help you with:\n\
            - Searching documentation (\"search for rust async\")\n\
            - Creating learning paths (\"teach me TypeScript\")\n\
            - Getting recommendations (\"get recommendations\")\n\
            - Explaining concepts (\"explain closures\")\n\n\
            Type 'help' for a complete list of commands!",
            message
        );

        Ok(ChatResponse {
            response: response_text,
            action_taken: None,
            results: None,
            suggestions: vec![
                "help".to_string(),
                "search for rust".to_string(),
                "get recommendations".to_string(),
            ],
            session_id: "unknown".to_string(),
        })
    }

    // Helper methods for parsing
    fn infer_difficulty(&self, message: &str) -> DifficultyLevel {
        let message_lower = message.to_lowercase();
        if message_lower.contains("beginner") || message_lower.contains("basic") || message_lower.contains("intro") {
            DifficultyLevel::Beginner
        } else if message_lower.contains("advanced") || message_lower.contains("expert") {
            DifficultyLevel::Advanced
        } else {
            DifficultyLevel::Intermediate
        }
    }

    fn infer_learning_format(&self, message: &str) -> LearningFormat {
        let message_lower = message.to_lowercase();
        if message_lower.contains("tutorial") || message_lower.contains("interactive") {
            LearningFormat::InteractiveTutorial
        } else if message_lower.contains("path") || message_lower.contains("course") || message_lower.contains("structured") {
            LearningFormat::StructuredPath
        } else if message_lower.contains("explore") || message_lower.contains("browse") {
            LearningFormat::ExploratorySearch
        } else {
            LearningFormat::PersonalizedRecommendations
        }
    }

    fn infer_doc_type(&self, message: &str) -> Option<DocType> {
        let message_lower = message.to_lowercase();
        if message_lower.contains("rust") {
            Some(DocType::Rust)
        } else if message_lower.contains("typescript") || message_lower.contains("ts") {
            Some(DocType::TypeScript)
        } else if message_lower.contains("python") || message_lower.contains("py") {
            Some(DocType::Python)
        } else if message_lower.contains("react") {
            Some(DocType::React)
        } else if message_lower.contains("tauri") {
            Some(DocType::Tauri)
        } else {
            None
        }
    }

    fn extract_time_minutes(&self, message: &str) -> Option<i32> {
        let time_regex = Regex::new(r"(\d+)\s*(minutes?|mins?|hours?)").unwrap();
        if let Some(captures) = time_regex.captures(message) {
            if let (Some(num), Some(unit)) = (captures.get(1), captures.get(2)) {
                if let Ok(value) = num.as_str().parse::<i32>() {
                    return Some(match unit.as_str() {
                        s if s.starts_with("hour") => value * 60,
                        _ => value,
                    });
                }
            }
        }
        None
    }

    fn extract_limit(&self, message: &str) -> Option<i32> {
        let limit_regex = Regex::new(r"(?:show|list|first|top)\s+(\d+)").unwrap();
        if let Some(captures) = limit_regex.captures(message) {
            if let Some(num) = captures.get(1) {
                return num.as_str().parse().ok();
            }
        }
        None
    }

    // Helper methods for new functionality
    fn extract_programming_language(&self, message: &str) -> Option<String> {
        let message_lower = message.to_lowercase();
        if message_lower.contains("rust") {
            Some("rust".to_string())
        } else if message_lower.contains("python") {
            Some("python".to_string())
        } else if message_lower.contains("javascript") || message_lower.contains("js") {
            Some("javascript".to_string())
        } else if message_lower.contains("typescript") || message_lower.contains("ts") {
            Some("typescript".to_string())
        } else if message_lower.contains("go") || message_lower.contains("golang") {
            Some("go".to_string())
        } else if message_lower.contains("java") {
            Some("java".to_string())
        } else {
            None
        }
    }

    fn extract_technology(&self, message: &str) -> Option<String> {
        let message_lower = message.to_lowercase();
        if message_lower.contains("react") {
            Some("react".to_string())
        } else if message_lower.contains("vue") {
            Some("vue".to_string())
        } else if message_lower.contains("angular") {
            Some("angular".to_string())
        } else if message_lower.contains("tauri") {
            Some("tauri".to_string())
        } else if message_lower.contains("node") {
            Some("nodejs".to_string())
        } else {
            None
        }
    }

    fn extract_directory(&self, message: &str) -> Option<String> {
        let dir_regex = Regex::new(r"in\s+(.+?)(?:\s|$)").unwrap();
        if let Some(captures) = dir_regex.captures(message) {
            if let Some(dir) = captures.get(1) {
                return Some(dir.as_str().to_string());
            }
        }
        None
    }

    fn extract_file_extensions(&self, message: &str) -> Option<Vec<String>> {
        let ext_regex = Regex::new(r"\.(\w+)").unwrap();
        let extensions: Vec<String> = ext_regex.captures_iter(message)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
            .collect();
        
        if extensions.is_empty() {
            None
        } else {
            Some(extensions)
        }
    }

    // New handler methods for web search
    async fn handle_web_search(&self, query: String, search_type: WebSearchType, max_results: Option<usize>) -> Result<ChatResponse> {
        let request = SearchRequest {
            query: query.clone(),
            max_results,
            search_type,
            filters: WebSearchFilters {
                site: None,
                file_type: None,
                date_range: None,
                language: None,
            },
        };

        match self.web_search.search(request).await {
            Ok(results) => {
                let response = format!(
                    "Found {} web results for '{}' (searched in {}ms):\n\n{}",
                    results.results.len(),
                    query,
                    results.search_time_ms,
                    results.results.iter()
                        .take(10)
                        .map(|r| format!("• **{}**\n  {}\n  {}", r.title, r.url, r.description))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("web_search".to_string()),
                    results: Some(serde_json::to_value(&results)?),
                    suggestions: results.suggestions,
                    session_id: "web_search".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I encountered an error while searching the web: {}", e),
                    action_taken: Some("web_search_error".to_string()),
                    results: None,
                    suggestions: vec!["Try a different search query".to_string()],
                    session_id: "web_search".to_string(),
                })
            }
        }
    }

    async fn handle_web_search_programming(&self, query: String, language: Option<String>) -> Result<ChatResponse> {
        match self.web_search.search_programming(&query, language.as_deref()).await {
            Ok(results) => {
                let lang_str = language.as_ref().map(|l| format!(" ({})", l)).unwrap_or_default();
                let response = format!(
                    "Found {} programming resources for '{}'{} (searched in {}ms):\n\n{}",
                    results.results.len(),
                    query,
                    lang_str,
                    results.search_time_ms,
                    results.results.iter()
                        .take(8)
                        .map(|r| format!("• **{}**\n  {}\n  {}", r.title, r.url, r.description))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("programming_search".to_string()),
                    results: Some(serde_json::to_value(&results)?),
                    suggestions: results.suggestions,
                    session_id: "programming_search".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I encountered an error while searching for programming resources: {}", e),
                    action_taken: Some("programming_search_error".to_string()),
                    results: None,
                    suggestions: vec!["Try a different programming topic".to_string()],
                    session_id: "programming_search".to_string(),
                })
            }
        }
    }

    async fn handle_web_search_docs(&self, query: String, technology: Option<String>) -> Result<ChatResponse> {
        match self.web_search.search_documentation(&query, technology.as_deref()).await {
            Ok(results) => {
                let tech_str = technology.as_ref().map(|t| format!(" ({})", t)).unwrap_or_default();
                let response = format!(
                    "Found {} documentation resources for '{}'{} (searched in {}ms):\n\n{}",
                    results.results.len(),
                    query,
                    tech_str,
                    results.search_time_ms,
                    results.results.iter()
                        .take(8)
                        .map(|r| format!("• **{}**\n  {}\n  {}", r.title, r.url, r.description))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("docs_search".to_string()),
                    results: Some(serde_json::to_value(&results)?),
                    suggestions: results.suggestions,
                    session_id: "docs_search".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I encountered an error while searching for documentation: {}", e),
                    action_taken: Some("docs_search_error".to_string()),
                    results: None,
                    suggestions: vec!["Try a different technology or topic".to_string()],
                    session_id: "docs_search".to_string(),
                })
            }
        }
    }

    async fn handle_page_summary(&self, url: String) -> Result<ChatResponse> {
        match self.web_search.get_page_summary(&url, 200).await {
            Ok(summary) => {
                let response = format!(
                    "Summary of {}:\n\n{}{}",
                    url,
                    summary,
                    if summary.len() > 800 { "\n\n[Summary truncated for readability]" } else { "" }
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("page_summary".to_string()),
                    results: None,
                    suggestions: vec![
                        "Get full page content".to_string(),
                        "Search for related topics".to_string(),
                    ],
                    session_id: "page_summary".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I couldn't retrieve the page summary: {}", e),
                    action_taken: Some("page_summary_error".to_string()),
                    results: None,
                    suggestions: vec!["Check if the URL is accessible".to_string()],
                    session_id: "page_summary".to_string(),
                })
            }
        }
    }

    // New handler methods for file operations
    async fn handle_read_file(&self, path: String) -> Result<ChatResponse> {
        match self.file_manager.read_file(&path).await {
            Ok(content) => {
                let preview = if content.len() > 2000 {
                    format!("{}...\n\n[File truncated for display. Total length: {} characters]", 
                           &content[..2000], content.len())
                } else {
                    content
                };

                Ok(ChatResponse {
                    response: format!("Contents of {}:\n\n```\n{}\n```", path, preview),
                    action_taken: Some("read_file".to_string()),
                    results: None,
                    suggestions: vec![
                        "Search file content".to_string(),
                        "Edit this file".to_string(),
                    ],
                    session_id: "file_read".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I couldn't read the file '{}': {}", path, e),
                    action_taken: Some("read_file_error".to_string()),
                    results: None,
                    suggestions: vec!["Check if the file exists and is readable".to_string()],
                    session_id: "file_read".to_string(),
                })
            }
        }
    }

    async fn handle_write_file(&self, path: String, content: String, append: bool) -> Result<ChatResponse> {
        let operation = if append { 
            self.file_manager.append_file(&path, &content).await 
        } else { 
            self.file_manager.write_file(&path, &content).await 
        };

        match operation {
            Ok(_) => {
                let action = if append { "appended to" } else { "written to" };
                Ok(ChatResponse {
                    response: format!("Successfully {} file '{}'", action, path),
                    action_taken: Some("write_file".to_string()),
                    results: None,
                    suggestions: vec![
                        "Read the file to verify".to_string(),
                        "Search for similar files".to_string(),
                    ],
                    session_id: "file_write".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I couldn't write to file '{}': {}", path, e),
                    action_taken: Some("write_file_error".to_string()),
                    results: None,
                    suggestions: vec!["Check file permissions and path".to_string()],
                    session_id: "file_write".to_string(),
                })
            }
        }
    }

    async fn handle_list_directory(&self, path: String, recursive: bool) -> Result<ChatResponse> {
        let operation = if recursive {
            self.file_manager.list_directory(&path).await
        } else {
            self.file_manager.list_directory(&path).await
        };

        match operation {
            Ok(entries) => {
                let all_items: Vec<_> = entries.files.iter()
                    .chain(entries.directories.iter())
                    .collect();
                
                let response = format!(
                    "Contents of {} ({} items):\n\n{}",
                    path,
                    all_items.len(),
                    all_items.iter()
                        .take(50)
                        .map(|entry| {
                            let size_str = if entry.is_file {
                                format!(" ({} bytes)", entry.size)
                            } else {
                                " [DIR]".to_string()
                            };
                            format!("• {}{}", entry.name, size_str)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("list_directory".to_string()),
                    results: Some(serde_json::to_value(&entries)?),
                    suggestions: vec![
                        "Read a specific file".to_string(),
                        "Search in this directory".to_string(),
                    ],
                    session_id: "dir_list".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I couldn't list directory '{}': {}", path, e),
                    action_taken: Some("list_directory_error".to_string()),
                    results: None,
                    suggestions: vec!["Check if the directory exists".to_string()],
                    session_id: "dir_list".to_string(),
                })
            }
        }
    }

    async fn handle_find_files(&self, pattern: String, directory: Option<String>) -> Result<ChatResponse> {
        let search_dir = directory.unwrap_or_else(|| ".".to_string());
        
        match self.file_manager.find_files(&pattern, Some(&search_dir)).await {
            Ok(files) => {
                let response = format!(
                    "Found {} files matching pattern '{}' in {}:\n\n{}",
                    files.len(),
                    pattern,
                    search_dir,
                    files.iter()
                        .take(30)
                        .map(|file| format!("• {}", file.name))
                        .collect::<Vec<_>>()
                        .join("\n")
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("find_files".to_string()),
                    results: Some(serde_json::to_value(&files)?),
                    suggestions: vec![
                        "Read one of these files".to_string(),
                        "Search content in these files".to_string(),
                    ],
                    session_id: "file_find".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I couldn't find files with pattern '{}': {}", pattern, e),
                    action_taken: Some("find_files_error".to_string()),
                    results: None,
                    suggestions: vec!["Try a different search pattern".to_string()],
                    session_id: "file_find".to_string(),
                })
            }
        }
    }

    async fn handle_search_file_content(&self, query: String, directory: Option<String>, file_extensions: Option<Vec<String>>) -> Result<ChatResponse> {
        let search_dir = directory.unwrap_or_else(|| ".".to_string());
        
        match self.file_manager.search_in_files(&query, Some(&search_dir), 3).await {
            Ok(matches) => {
                let response = format!(
                    "Found {} matches for '{}' in {}:\n\n{}",
                    matches.len(),
                    query,
                    search_dir,
                    matches.iter()
                        .take(20)
                        .map(|m| format!("• **{}** ({} matches)\n  {}", 
                            m.file_path.display(), 
                            m.total_matches, 
                            m.matches.first().map(|match_info| match_info.line_content.as_str()).unwrap_or("No match content")
                        ))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );

                Ok(ChatResponse {
                    response,
                    action_taken: Some("search_file_content".to_string()),
                    results: Some(serde_json::to_value(&matches)?),
                    suggestions: vec![
                        "Read the full file".to_string(),
                        "Refine the search query".to_string(),
                    ],
                    session_id: "content_search".to_string(),
                })
            }
            Err(e) => {
                Ok(ChatResponse {
                    response: format!("Sorry, I couldn't search file content: {}", e),
                    action_taken: Some("search_file_content_error".to_string()),
                    results: None,
                    suggestions: vec!["Try a different search term".to_string()],
                    session_id: "content_search".to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_search_commands() {
        // Test command parsing logic
    }

    #[test]
    fn test_infer_difficulty() {
        // Test difficulty inference
    }
}
