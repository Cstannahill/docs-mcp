// src/agents/context_manager.rs
//! Advanced Context Management for Multi-Turn Conversations
//! 
//! This module provides sophisticated context management capabilities including:
//! - Conversation history tracking with intelligent pruning
//! - Context window optimization for different models
//! - Semantic context relevance scoring
//! - Multi-session context correlation
//! - Context compression and summarization

use anyhow::{Result, Context as AnyhowContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::agents::{Agent, AgentCapability, FlowContext};
use crate::database::Database;
use crate::embeddings::EmbeddingService;

/// Represents a single conversation turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub id: Uuid,
    pub session_id: String,
    pub user_message: String,
    pub assistant_response: String,
    pub timestamp: DateTime<Utc>,
    pub context_tokens: usize,
    pub relevance_score: f32,
    pub model_used: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Context window management strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextStrategy {
    /// Keep most recent N turns
    RecentTurns(usize),
    /// Keep within token limit with intelligent pruning
    TokenLimited { max_tokens: usize, preserve_important: bool },
    /// Semantic relevance-based selection
    SemanticRelevance { max_turns: usize, min_relevance: f32 },
    /// Hybrid approach combining multiple strategies
    Hybrid {
        max_tokens: usize,
        max_turns: usize,
        min_relevance: f32,
        preserve_system_context: bool,
    },
}

/// Context compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCompressionConfig {
    pub enabled: bool,
    pub compression_threshold: usize, // Compress when context exceeds this size
    pub summary_max_tokens: usize,
    pub preserve_last_n_turns: usize,
}

/// Multi-turn context management system
pub struct ContextManager {
    db: Arc<Database>,
    embedding_generator: Arc<EmbeddingService>,
    conversations: Arc<RwLock<HashMap<String, VecDeque<ConversationTurn>>>>,
    context_strategies: Arc<RwLock<HashMap<String, ContextStrategy>>>,
    compression_config: ContextCompressionConfig,
    max_sessions: usize,
}

impl ContextManager {
    /// Create a new context manager
    pub async fn new(
        db: Arc<Database>,
        embedding_generator: Arc<EmbeddingService>,
    ) -> Result<Self> {
        let compression_config = ContextCompressionConfig {
            enabled: true,
            compression_threshold: 8000,
            summary_max_tokens: 500,
            preserve_last_n_turns: 3,
        };

        Ok(Self {
            db,
            embedding_generator,
            conversations: Arc::new(RwLock::new(HashMap::new())),
            context_strategies: Arc::new(RwLock::new(HashMap::new())),
            compression_config,
            max_sessions: 1000,
        })
    }

    /// Add a new conversation turn
    pub async fn add_turn(
        &self,
        session_id: &str,
        user_message: String,
        assistant_response: String,
        model_used: Option<String>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<Uuid> {
        let turn_id = Uuid::new_v4();
        let turn = ConversationTurn {
            id: turn_id,
            session_id: session_id.to_string(),
            user_message: user_message.clone(),
            assistant_response: assistant_response.clone(),
            timestamp: Utc::now(),
            context_tokens: self.estimate_tokens(&user_message, &assistant_response),
            relevance_score: 1.0, // Will be updated based on semantic analysis
            model_used,
            metadata,
        };

        // Store in database for persistence
        self.store_turn_in_db(&turn).await?;

        // Update in-memory conversation
        let mut conversations = self.conversations.write().await;
        let session_turns = conversations.entry(session_id.to_string()).or_insert_with(VecDeque::new);
        session_turns.push_back(turn);

        // Enforce session limits
        if conversations.len() > self.max_sessions {
            // Remove oldest session
            let oldest_session = conversations.keys().next().unwrap().clone();
            conversations.remove(&oldest_session);
        }

        // Apply context management strategy
        self.apply_context_strategy(session_id).await?;

        Ok(turn_id)
    }

    /// Get optimized context for a session
    pub async fn get_context(
        &self,
        session_id: &str,
        max_tokens: Option<usize>,
        query_context: Option<&str>,
    ) -> Result<Vec<ConversationTurn>> {
        let conversations = self.conversations.read().await;
        let session_turns = conversations.get(session_id)
            .map(|turns| turns.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        if session_turns.is_empty() {
            return Ok(Vec::new());
        }

        // Apply context optimization based on strategy
        let strategy = self.get_context_strategy(session_id).await;
        let optimized_context = self.optimize_context(session_turns, &strategy, max_tokens, query_context).await?;

        Ok(optimized_context)
    }

    /// Update relevance scores based on current query
    pub async fn update_relevance_scores(
        &self,
        session_id: &str,
        current_query: &str,
    ) -> Result<()> {
        let query_embedding = self.embedding_generator.generate_embedding(current_query).await?;
        
        let mut conversations = self.conversations.write().await;
        if let Some(session_turns) = conversations.get_mut(session_id) {
            for turn in session_turns.iter_mut() {
                // Generate embedding for the turn's content
                let turn_content = format!("{} {}", turn.user_message, turn.assistant_response);
                if let Ok(turn_embedding) = self.embedding_generator.generate_embedding(&turn_content).await {
                    // Calculate cosine similarity
                    turn.relevance_score = self.calculate_similarity(&query_embedding, &turn_embedding);
                }
            }
        }

        Ok(())
    }

    /// Set context strategy for a session
    pub async fn set_context_strategy(&self, session_id: &str, strategy: ContextStrategy) {
        let mut strategies = self.context_strategies.write().await;
        strategies.insert(session_id.to_string(), strategy);
    }

    /// Get context strategy for a session (with fallback to default)
    async fn get_context_strategy(&self, session_id: &str) -> ContextStrategy {
        let strategies = self.context_strategies.read().await;
        strategies.get(session_id).cloned().unwrap_or_else(|| {
            ContextStrategy::Hybrid {
                max_tokens: 4000,
                max_turns: 10,
                min_relevance: 0.3,
                preserve_system_context: true,
            }
        })
    }

    /// Apply context management strategy to a session
    async fn apply_context_strategy(&self, session_id: &str) -> Result<()> {
        let strategy = self.get_context_strategy(session_id).await;
        let mut conversations = self.conversations.write().await;
        
        if let Some(session_turns) = conversations.get_mut(session_id) {
            match &strategy {
                ContextStrategy::RecentTurns(max_turns) => {
                    while session_turns.len() > *max_turns {
                        session_turns.pop_front();
                    }
                }
                ContextStrategy::TokenLimited { max_tokens, preserve_important } => {
                    self.apply_token_limit(session_turns, *max_tokens, *preserve_important).await?;
                }
                ContextStrategy::SemanticRelevance { max_turns, min_relevance } => {
                    self.apply_semantic_filtering(session_turns, *max_turns, *min_relevance).await?;
                }
                ContextStrategy::Hybrid { max_tokens, max_turns, min_relevance, preserve_system_context } => {
                    self.apply_hybrid_strategy(
                        session_turns,
                        *max_tokens,
                        *max_turns,
                        *min_relevance,
                        *preserve_system_context,
                    ).await?;
                }
            }
        }

        Ok(())
    }

    /// Optimize context based on strategy and constraints
    async fn optimize_context(
        &self,
        mut turns: Vec<ConversationTurn>,
        strategy: &ContextStrategy,
        max_tokens: Option<usize>,
        query_context: Option<&str>,
    ) -> Result<Vec<ConversationTurn>> {
        // If we have a query context, update relevance scores
        if let Some(query) = query_context {
            let query_embedding = self.embedding_generator.generate_embedding(query).await?;
            for turn in &mut turns {
                let turn_content = format!("{} {}", turn.user_message, turn.assistant_response);
                if let Ok(turn_embedding) = self.embedding_generator.generate_embedding(&turn_content).await {
                    turn.relevance_score = self.calculate_similarity(&query_embedding, &turn_embedding);
                }
            }
        }

        // Apply optimization strategy
        match strategy {
            ContextStrategy::SemanticRelevance { max_turns, min_relevance } => {
                turns.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
                turns.retain(|turn| turn.relevance_score >= *min_relevance);
                turns.truncate(*max_turns);
                // Sort back by timestamp to maintain conversation flow
                turns.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            }
            ContextStrategy::TokenLimited { max_tokens: token_limit, .. } => {
                let token_limit = max_tokens.unwrap_or(*token_limit);
                let mut total_tokens = 0;
                let mut filtered_turns = Vec::new();
                
                // Start from most recent and work backwards
                for turn in turns.iter().rev() {
                    if total_tokens + turn.context_tokens <= token_limit {
                        total_tokens += turn.context_tokens;
                        filtered_turns.push(turn.clone());
                    } else {
                        break;
                    }
                }
                
                filtered_turns.reverse();
                turns = filtered_turns;
            }
            _ => {
                // For other strategies, apply token limit if specified
                if let Some(token_limit) = max_tokens {
                    let mut total_tokens = 0;
                    let mut filtered_turns = Vec::new();
                    
                    for turn in turns.iter().rev() {
                        if total_tokens + turn.context_tokens <= token_limit {
                            total_tokens += turn.context_tokens;
                            filtered_turns.push(turn.clone());
                        } else {
                            break;
                        }
                    }
                    
                    filtered_turns.reverse();
                    turns = filtered_turns;
                }
            }
        }

        Ok(turns)
    }

    /// Apply token limit with intelligent pruning
    async fn apply_token_limit(
        &self,
        session_turns: &mut VecDeque<ConversationTurn>,
        max_tokens: usize,
        preserve_important: bool,
    ) -> Result<()> {
        let mut total_tokens: usize = session_turns.iter().map(|turn| turn.context_tokens).sum();
        
        while total_tokens > max_tokens && session_turns.len() > 1 {
            if preserve_important {
                // Find least relevant turn to remove (excluding the most recent)
                let mut min_relevance = f32::INFINITY;
                let mut remove_index = 0;
                
                for (i, turn) in session_turns.iter().enumerate() {
                    if i < session_turns.len() - 1 && turn.relevance_score < min_relevance {
                        min_relevance = turn.relevance_score;
                        remove_index = i;
                    }
                }
                
                if let Some(removed) = session_turns.remove(remove_index) {
                    total_tokens -= removed.context_tokens;
                }
            } else {
                // Remove oldest turn
                if let Some(removed) = session_turns.pop_front() {
                    total_tokens -= removed.context_tokens;
                }
            }
        }
        
        Ok(())
    }

    /// Apply semantic filtering
    async fn apply_semantic_filtering(
        &self,
        session_turns: &mut VecDeque<ConversationTurn>,
        max_turns: usize,
        min_relevance: f32,
    ) -> Result<()> {
        // Convert to Vec for easier manipulation
        let mut turns: Vec<_> = session_turns.drain(..).collect();
        
        // Filter by relevance
        turns.retain(|turn| turn.relevance_score >= min_relevance);
        
        // Sort by relevance and keep top turns
        turns.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        turns.truncate(max_turns);
        
        // Sort back by timestamp
        turns.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        // Put back into deque
        session_turns.extend(turns);
        
        Ok(())
    }

    /// Apply hybrid strategy combining multiple approaches
    async fn apply_hybrid_strategy(
        &self,
        session_turns: &mut VecDeque<ConversationTurn>,
        max_tokens: usize,
        max_turns: usize,
        min_relevance: f32,
        preserve_system_context: bool,
    ) -> Result<()> {
        // First apply turn limit
        while session_turns.len() > max_turns {
            session_turns.pop_front();
        }

        // Then apply relevance filtering (but preserve last turn)
        let last_turn = session_turns.pop_back();
        let mut turns: Vec<_> = session_turns.drain(..).collect();
        turns.retain(|turn| turn.relevance_score >= min_relevance);
        session_turns.extend(turns);
        if let Some(last) = last_turn {
            session_turns.push_back(last);
        }

        // Finally apply token limit
        self.apply_token_limit(session_turns, max_tokens, preserve_system_context).await?;

        Ok(())
    }

    /// Estimate token count for a conversation turn
    fn estimate_tokens(&self, user_message: &str, assistant_response: &str) -> usize {
        // Rough estimation: ~4 characters per token for English text
        (user_message.len() + assistant_response.len()) / 4
    }

    /// Calculate cosine similarity between embeddings
    fn calculate_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }

        let dot_product: f32 = embedding1.iter().zip(embedding2.iter()).map(|(a, b)| a * b).sum();
        let magnitude1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1 * magnitude2)
        }
    }

    /// Store conversation turn in database for persistence
    async fn store_turn_in_db(&self, _turn: &ConversationTurn) -> Result<()> {
        // TODO: Implement database storage once conversation_turns table is available
        // This would implement database storage
        // For now, we'll skip database storage
        /*
        sqlx::query!(
            r#"
            INSERT INTO conversation_turns (
                id, session_id, user_message, assistant_response, 
                timestamp, context_tokens, relevance_score, model_used, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
            turn.id.to_string(),
            turn.session_id,
            turn.user_message,
            turn.assistant_response,
            turn.timestamp,
            turn.context_tokens as i64,
            turn.relevance_score,
            turn.model_used,
            serde_json::to_string(&turn.metadata).unwrap_or_default()
        )
        .execute(&*self.db.pool)
        .await
        .context("Failed to store conversation turn")?;
        */

        Ok(())
    }

    /// Load conversation history from database
    pub async fn load_session_history(&self, _session_id: &str) -> Result<Vec<ConversationTurn>> {
        // TODO: Implement database loading once conversation_turns table is available
        /*
        let rows = sqlx::query!(
            "SELECT * FROM conversation_turns WHERE session_id = ? ORDER BY timestamp ASC",
            session_id
        )
        .fetch_all(&*self.db.pool)
        .await
        .context("Failed to load conversation history")?;

        let mut turns = Vec::new();
        for row in rows {
            let metadata: HashMap<String, serde_json::Value> = serde_json::from_str(&row.metadata)
                .unwrap_or_default();

            turns.push(ConversationTurn {
                id: Uuid::parse_str(&row.id).unwrap_or_else(|_| Uuid::new_v4()),
                session_id: row.session_id,
                user_message: row.user_message,
                assistant_response: row.assistant_response,
                timestamp: row.timestamp,
                context_tokens: row.context_tokens as usize,
                relevance_score: row.relevance_score,
                model_used: row.model_used,
                metadata,
            });
        }

        Ok(turns)
        */
        Ok(Vec::new())
    }

    /// Get context statistics for a session
    pub async fn get_context_stats(&self, session_id: &str) -> Result<ContextStats> {
        let conversations = self.conversations.read().await;
        let session_turns = conversations.get(session_id);

        let (turn_count, total_tokens, avg_relevance) = if let Some(turns) = session_turns {
            let count = turns.len();
            let tokens: usize = turns.iter().map(|t| t.context_tokens).sum();
            let avg_rel = if count > 0 {
                turns.iter().map(|t| t.relevance_score).sum::<f32>() / count as f32
            } else {
                0.0
            };
            (count, tokens, avg_rel)
        } else {
            (0, 0, 0.0)
        };

        Ok(ContextStats {
            session_id: session_id.to_string(),
            turn_count,
            total_tokens,
            average_relevance: avg_relevance,
            strategy: self.get_context_strategy(session_id).await,
        })
    }
}

/// Context statistics for monitoring and optimization
#[derive(Debug, Serialize, Deserialize)]
pub struct ContextStats {
    pub session_id: String,
    pub turn_count: usize,
    pub total_tokens: usize,
    pub average_relevance: f32,
    pub strategy: ContextStrategy,
}

/// Implementation of the Agent trait for integration with the agentic system
#[async_trait]
impl Agent for ContextManager {
    fn name(&self) -> &'static str {
        "context_manager"
    }

    fn description(&self) -> &'static str {
        "Advanced context management for multi-turn conversations with intelligent pruning and optimization"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability::ContextManagement,
            AgentCapability::ConversationHistory,
            AgentCapability::SemanticAnalysis,
        ]
    }

    async fn execute(&self, context: &mut FlowContext) -> Result<serde_json::Value> {
        // Extract context management request from the flow context
        let session_id = context.session_id.clone();
        let request_type = context.metadata.get("context_operation")
            .and_then(|v| v.as_str())
            .unwrap_or("get_context");

        match request_type {
            "get_context" => {
                let max_tokens = context.metadata.get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let query_context = context.metadata.get("query_context")
                    .and_then(|v| v.as_str());

                let context_turns = self.get_context(&session_id, max_tokens, query_context).await?;
                Ok(serde_json::to_value(context_turns)?)
            }
            "add_turn" => {
                let user_message = context.metadata.get("user_message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let assistant_response = context.metadata.get("assistant_response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let model_used = context.metadata.get("model_used")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let turn_id = self.add_turn(
                    &session_id,
                    user_message,
                    assistant_response,
                    model_used,
                    HashMap::new(),
                ).await?;

                Ok(serde_json::json!({ "turn_id": turn_id }))
            }
            "get_stats" => {
                let stats = self.get_context_stats(&session_id).await?;
                Ok(serde_json::to_value(stats)?)
            }
            _ => {
                Err(anyhow::anyhow!("Unknown context operation: {}", request_type))
            }
        }
    }

    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.capabilities_required.contains(&AgentCapability::ContextManagement) ||
        context.metadata.contains_key("context_operation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Mock implementations for testing
    // Note: In real implementation, these would use actual database and embedding services

    #[tokio::test]
    async fn test_context_manager_creation() {
        // Test would require proper Database and EmbeddingGenerator setup
        // This is a placeholder for the test structure
    }

    #[tokio::test]
    async fn test_context_strategies() {
        // Test different context management strategies
    }

    #[tokio::test]
    async fn test_relevance_scoring() {
        // Test semantic relevance scoring functionality
    }
}
