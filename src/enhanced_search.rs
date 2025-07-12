use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

use crate::database::{Database, DocumentPage, UserContext, DifficultyLevel, InteractionType, SearchAnalytics};
use crate::embeddings::EmbeddingService;
use crate::ranking::AdvancedRankingEngine;
use crate::learning::{LearningPathEngine, LearningRecommendation, InteractiveTutorial};

#[derive(Clone)]
pub struct EnhancedSearchSystem {
    db: Database,
    embedding_service: EmbeddingService,
    ranking_engine: AdvancedRankingEngine,
    learning_engine: LearningPathEngine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSearchRequest {
    pub query: String,
    pub session_id: String,
    pub user_context: Option<UserContext>,
    pub search_type: SearchType,
    pub filters: SearchFilters,
    pub include_suggestions: bool,
    pub include_learning_paths: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    Semantic,
    Keyword,
    Hybrid,
    LearningFocused,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    pub doc_types: Vec<crate::database::DocType>,
    pub difficulty_levels: Vec<DifficultyLevel>,
    pub content_types: Vec<String>,
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub exclude_completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSearchResponse {
    pub results: Vec<crate::database::EnhancedSearchResult>,
    pub total_results: usize,
    pub search_time_ms: u64,
    pub suggestions: Vec<crate::database::ContentSuggestion>,
    pub learning_recommendations: Option<LearningRecommendation>,
    pub related_topics: Vec<String>,
    pub query_expansion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSessionRequest {
    pub session_id: String,
    pub topic: String,
    pub target_difficulty: DifficultyLevel,
    pub learning_goals: Vec<String>,
    pub available_time_minutes: Option<i32>,
    pub preferred_format: LearningFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningFormat {
    StructuredPath,
    InteractiveTutorial,
    ExploratorySearch,
    PersonalizedRecommendations,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSessionResponse {
    pub session_id: String,
    pub recommended_path: Option<crate::database::LearningPath>,
    pub interactive_tutorial: Option<InteractiveTutorial>,
    pub immediate_suggestions: Vec<crate::database::ContentSuggestion>,
    pub estimated_completion_time: i32,
    pub personalization_notes: Vec<String>,
}

impl EnhancedSearchSystem {
    pub async fn new(db: Database, openai_api_key: Option<String>) -> Result<Self> {
        let embedding_service = EmbeddingService::new(openai_api_key).await?;
        let ranking_engine = AdvancedRankingEngine::new(db.clone()).await?;
        let learning_engine = LearningPathEngine::new(db.clone());

        Ok(Self {
            db,
            embedding_service,
            ranking_engine,
            learning_engine,
        })
    }

    /// Enhanced search with AI-powered ranking and semantic understanding
    pub async fn enhanced_search(&self, request: EnhancedSearchRequest) -> Result<EnhancedSearchResponse> {
        let start_time = std::time::Instant::now();
        
        // Record search analytics
        let analytics = SearchAnalytics {
            id: None,
            session_id: request.session_id.clone(),
            query: request.query.clone(),
            results_count: 0, // Will be updated
            clicked_page_id: None,
            click_position: None,
            search_time_ms: None, // Will be updated
            filters_used: serde_json::to_value(&request.filters).unwrap_or_default(),
            timestamp: Utc::now(),
        };

        // Perform search based on type
        let mut results = match request.search_type {
            SearchType::Semantic => self.semantic_search(&request).await?,
            SearchType::Keyword => self.keyword_search(&request).await?,
            SearchType::Hybrid => self.hybrid_search(&request).await?,
            SearchType::LearningFocused => self.learning_focused_search(&request).await?,
        };

        // Apply advanced ranking
        if let Some(user_context) = &request.user_context {
            results = self.ranking_engine.rank_search_results(
                results,
                &request.query,
                user_context,
            ).await?;
        }

        // Apply filters
        results = self.apply_search_filters(results, &request.filters).await?;

        // Generate suggestions if requested
        let suggestions = if request.include_suggestions {
            self.generate_content_suggestions(&request.session_id, &request.query, &results).await?
        } else {
            Vec::new()
        };

        // Generate learning recommendations if requested
        let learning_recommendations = if request.include_learning_paths {
            if let Some(user_context) = &request.user_context {
                Some(self.learning_engine.generate_recommendations(user_context).await?)
            } else {
                None
            }
        } else {
            None
        };

        // Extract related topics
        let related_topics = self.extract_related_topics(&results).await?;

        // Generate query expansion
        let query_expansion = self.generate_query_expansion(&request.query).await?;

        let search_time_ms = start_time.elapsed().as_millis() as u64;
        let total_results = results.len();

        // Update analytics
        let mut final_analytics = analytics;
        final_analytics.results_count = total_results as i32;
        final_analytics.search_time_ms = Some(search_time_ms as i32);
        self.db.record_search_analytics(&final_analytics).await?;

        Ok(EnhancedSearchResponse {
            results,
            total_results,
            search_time_ms,
            suggestions,
            learning_recommendations,
            related_topics,
            query_expansion,
        })
    }

    /// Create a personalized learning session
    pub async fn create_learning_session(&self, request: LearningSessionRequest) -> Result<LearningSessionResponse> {
        // Build user context from session history
        let user_context = self.build_user_context(&request.session_id).await?;

        let response = match request.preferred_format {
            LearningFormat::StructuredPath => {
                let path = self.learning_engine.generate_personalized_path(
                    &user_context,
                    &request.topic,
                    request.target_difficulty,
                ).await?;

                let estimated_time = path.estimated_duration_minutes;

                LearningSessionResponse {
                    session_id: request.session_id,
                    recommended_path: Some(path),
                    interactive_tutorial: None,
                    immediate_suggestions: Vec::new(),
                    estimated_completion_time: estimated_time,
                    personalization_notes: vec!["Path customized based on your learning history".to_string()],
                }
            }
            LearningFormat::InteractiveTutorial => {
                // Find relevant pages for the topic
                let pages = self.db.search_pages_by_topic(&request.topic).await?;
                let selected_pages = pages.into_iter().take(5).collect(); // Limit to 5 pages for tutorial

                let tutorial = self.learning_engine.create_interactive_tutorial(
                    &request.topic,
                    selected_pages,
                    request.target_difficulty,
                ).await?;

                let estimated_time = tutorial.estimated_duration_minutes;

                LearningSessionResponse {
                    session_id: request.session_id,
                    recommended_path: None,
                    interactive_tutorial: Some(tutorial),
                    immediate_suggestions: Vec::new(),
                    estimated_completion_time: estimated_time,
                    personalization_notes: vec!["Interactive tutorial with hands-on exercises".to_string()],
                }
            }
            LearningFormat::ExploratorySearch => {
                // Generate enhanced search suggestions
                let suggestions = self.generate_exploratory_suggestions(&request).await?;

                LearningSessionResponse {
                    session_id: request.session_id,
                    recommended_path: None,
                    interactive_tutorial: None,
                    immediate_suggestions: suggestions,
                    estimated_completion_time: 30, // Flexible exploration time
                    personalization_notes: vec!["Explore at your own pace with curated suggestions".to_string()],
                }
            }
            LearningFormat::PersonalizedRecommendations => {
                let recommendations = self.learning_engine.generate_recommendations(&user_context).await?;
                let suggestions = recommendations.suggested_content;

                LearningSessionResponse {
                    session_id: request.session_id,
                    recommended_path: None,
                    interactive_tutorial: None,
                    immediate_suggestions: suggestions,
                    estimated_completion_time: 45,
                    personalization_notes: vec![
                        "Recommendations based on your learning profile".to_string(),
                        format!("Targeting {} skill gaps", recommendations.skill_gaps.len()),
                    ],
                }
            }
        };

        Ok(response)
    }

    /// Track user interaction and update learning profile
    pub async fn track_interaction(&self, 
        session_id: &str, 
        page_id: &str, 
        interaction_type: InteractionType,
        duration_seconds: Option<i32>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        let interaction = crate::database::UserInteraction {
            id: None,
            session_id: session_id.to_string(),
            page_id: page_id.to_string(),
            interaction_type,
            duration_seconds,
            metadata: metadata.unwrap_or_default(),
            timestamp: Utc::now(),
        };

        self.db.record_user_interaction(&interaction).await?;

        // Update learning recommendations based on new interaction
        self.update_learning_recommendations(session_id).await?;

        Ok(())
    }

    /// Search implementation methods
    async fn semantic_search(&self, request: &EnhancedSearchRequest) -> Result<Vec<crate::database::EnhancedSearchResult>> {
        // Generate embedding for query
        let query_embedding = self.embedding_service.generate_embedding(&request.query).await?;
        
        // Find similar documents
        let similar_pages = self.db.search_similar_embeddings(&query_embedding, Some(50)).await?;
        
        let mut results = Vec::new();
        for (page_id, similarity_score) in similar_pages {
            if let Some(page) = self.db.get_page(&page_id).await? {
                let result = crate::database::EnhancedSearchResult {
                    page,
                    ranking_factors: crate::database::RankingFactors {
                        text_relevance_score: 0.5, // Default, will be calculated properly
                        semantic_similarity_score: similarity_score,
                        quality_score: 0.8, // Default
                        freshness_score: 0.7, // Default
                        user_engagement_score: 0.5, // Default
                        context_relevance_score: 0.6, // Default
                        final_score: similarity_score * 0.8, // Weighted by semantic similarity
                    },
                    highlighted_snippets: Vec::new(),
                    related_pages: Vec::new(),
                };
                results.push(result);
            }
        }
        
        Ok(results)
    }

    async fn keyword_search(&self, request: &EnhancedSearchRequest) -> Result<Vec<crate::database::EnhancedSearchResult>> {
        let pages = self.db.search_pages_by_topic(&request.query).await?;
        
        let mut results = Vec::new();
        for page in pages {
            let result = crate::database::EnhancedSearchResult {
                page,
                ranking_factors: crate::database::RankingFactors {
                    text_relevance_score: 0.8,
                    semantic_similarity_score: 0.3,
                    quality_score: 0.7,
                    freshness_score: 0.6,
                    user_engagement_score: 0.5,
                    context_relevance_score: 0.4,
                    final_score: 0.6,
                },
                highlighted_snippets: Vec::new(),
                related_pages: Vec::new(),
            };
            results.push(result);
        }
        
        Ok(results)
    }

    async fn hybrid_search(&self, request: &EnhancedSearchRequest) -> Result<Vec<crate::database::EnhancedSearchResult>> {
        // Combine semantic and keyword search results
        let semantic_results = self.semantic_search(request).await?;
        let keyword_results = self.keyword_search(request).await?;
        
        // Merge and deduplicate results
        let mut combined_results = Vec::new();
        let mut seen_pages = std::collections::HashSet::new();
        
        // Add semantic results first (higher weight)
        for mut result in semantic_results {
            if seen_pages.insert(result.page.id.clone()) {
                result.ranking_factors.final_score *= 1.2; // Boost semantic results
                combined_results.push(result);
            }
        }
        
        // Add keyword results that weren't already included
        for result in keyword_results {
            if seen_pages.insert(result.page.id.clone()) {
                combined_results.push(result);
            }
        }
        
        Ok(combined_results)
    }

    async fn learning_focused_search(&self, request: &EnhancedSearchRequest) -> Result<Vec<crate::database::EnhancedSearchResult>> {
        // Start with hybrid search
        let mut results = self.hybrid_search(request).await?;
        
        // Boost results that are good for learning
        for result in &mut results {
            let page = &result.page;
            
            // Boost based on educational content indicators
            let learning_boost = if page.title.to_lowercase().contains("tutorial") 
                || page.title.to_lowercase().contains("guide") 
                || page.title.to_lowercase().contains("introduction") {
                1.5
            } else if page.content.to_lowercase().contains("example") 
                || page.content.to_lowercase().contains("step") {
                1.2
            } else {
                1.0
            };
            
            result.ranking_factors.final_score *= learning_boost;
        }
        
        // Sort by final score
        results.sort_by(|a, b| b.ranking_factors.final_score.partial_cmp(&a.ranking_factors.final_score).unwrap());
        
        Ok(results)
    }

    async fn apply_search_filters(&self, results: Vec<crate::database::EnhancedSearchResult>, filters: &SearchFilters) -> Result<Vec<crate::database::EnhancedSearchResult>> {
        let filtered_results = results.into_iter()
            .filter(|result| {
                // Filter by doc type
                if !filters.doc_types.is_empty() && !filters.doc_types.contains(&result.page.doc_type) {
                    return false;
                }
                
                // Add other filters as needed
                true
            })
            .collect();
            
        Ok(filtered_results)
    }

    async fn generate_content_suggestions(&self, session_id: &str, query: &str, results: &[crate::database::EnhancedSearchResult]) -> Result<Vec<crate::database::ContentSuggestion>> {
        // Simple implementation - suggest related pages
        let mut suggestions = Vec::new();
        
        for result in results.iter().take(3) {
            let suggestion = crate::database::ContentSuggestion {
                id: None,
                session_id: session_id.to_string(),
                suggested_page_id: result.page.id.clone(),
                suggestion_type: crate::database::SuggestionType::Related,
                confidence_score: result.ranking_factors.final_score,
                reason: Some(format!("Related to your search for '{}'", query)),
                shown: false,
                clicked: false,
                created_at: Utc::now(),
            };
            suggestions.push(suggestion);
        }
        
        Ok(suggestions)
    }

    async fn extract_related_topics(&self, results: &[crate::database::EnhancedSearchResult]) -> Result<Vec<String>> {
        // Extract common keywords from top results
        let mut topic_counts = HashMap::new();
        
        for result in results.iter().take(10) {
            let words: Vec<&str> = result.page.title.split_whitespace()
                .chain(result.page.content.split_whitespace())
                .filter(|word| word.len() > 3)
                .collect();
                
            for word in words {
                let word_lower = word.to_lowercase();
                *topic_counts.entry(word_lower).or_insert(0) += 1;
            }
        }
        
        let mut topics: Vec<(String, i32)> = topic_counts.into_iter().collect();
        topics.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        
        Ok(topics.into_iter().take(5).map(|(topic, _)| topic).collect())
    }

    async fn generate_query_expansion(&self, query: &str) -> Result<Option<String>> {
        // Simple query expansion using synonyms
        let expanded_terms = if query.to_lowercase().contains("rust") {
            vec!["rust", "cargo", "rustc", "programming"]
        } else if query.to_lowercase().contains("python") {
            vec!["python", "pip", "django", "flask"]
        } else if query.to_lowercase().contains("typescript") {
            vec!["typescript", "javascript", "npm", "node"]
        } else {
            return Ok(None);
        };
        
        Ok(Some(expanded_terms.join(" OR ")))
    }

    async fn build_user_context(&self, session_id: &str) -> Result<UserContext> {
        let recent_interactions = self.db.get_user_interactions(session_id, Some(50)).await?;
        
        Ok(UserContext {
            session_id: session_id.to_string(),
            skill_level: Some(DifficultyLevel::Intermediate), // Default
            preferred_topics: Vec::new(),
            current_learning_paths: Vec::new(),
            recent_interactions,
            search_history: Vec::new(),
        })
    }

    async fn generate_exploratory_suggestions(&self, request: &LearningSessionRequest) -> Result<Vec<crate::database::ContentSuggestion>> {
        let pages = self.db.search_pages_by_topic(&request.topic).await?;
        
        let mut suggestions = Vec::new();
        for (index, page) in pages.into_iter().take(10).enumerate() {
            let suggestion = crate::database::ContentSuggestion {
                id: None,
                session_id: request.session_id.clone(),
                suggested_page_id: page.id,
                suggestion_type: crate::database::SuggestionType::Related,
                confidence_score: 1.0 - (index as f32 * 0.1),
                reason: Some(format!("Explore {} concepts", request.topic)),
                shown: false,
                clicked: false,
                created_at: Utc::now(),
            };
            suggestions.push(suggestion);
        }
        
        Ok(suggestions)
    }

    async fn update_learning_recommendations(&self, session_id: &str) -> Result<()> {
        // This would update the user's learning recommendations based on new interactions
        // Implementation would analyze the interaction patterns and update suggestions
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_search_system() {
        // Tests would go here
    }
}
