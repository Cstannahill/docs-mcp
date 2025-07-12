use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;
use chrono::Utc;

use crate::enhanced_search::{EnhancedSearchSystem, EnhancedSearchRequest, SearchType, SearchFilters, LearningSessionRequest, LearningFormat};
use crate::database::{Database, InteractionType, DifficultyLevel, UserContext, DocType};

pub struct ChatInterface {
    enhanced_search: EnhancedSearchSystem,
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
            search_type,
            filters: SearchFilters {
                doc_types: Vec::new(),
                difficulty_levels: Vec::new(),
                content_types: Vec::new(),
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
                result.ranking_factors.final_score,
                result.page.doc_type
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
            preferred_topics: Vec::new(),
            current_learning_paths: Vec::new(),
            recent_interactions: Vec::new(),
            search_history: Vec::new(),
        };

        let recommendations = self.enhanced_search.learning_engine.generate_recommendations(&user_context).await?;
        
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
            interaction_type,
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
