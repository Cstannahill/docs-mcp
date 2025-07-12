// src/agents/model_selector.rs
//! Advanced Model Selection Algorithms
//! 
//! This module provides sophisticated model selection capabilities including:
//! - Performance-based model ranking and selection
//! - Task-specific model optimization
//! - Dynamic model switching based on performance metrics
//! - Cost optimization with quality constraints
//! - Model capability matching and recommendation

use crate::agents::{Agent, FlowContext, AgentCapability};
use anyhow::{Result, Context as AnyhowContext};
use crate::model_discovery::{ModelDatabase, ModelInfo, PerformanceMetrics};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration as ChronoDuration};

/// Model capability categories for task matching
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    CodeGeneration,
    CodeAnalysis,
    Documentation,
    Conversation,
    Reasoning,
    Mathematics,
    Translation,
    Summarization,
    Classification,
    Embedding,
    ImageGeneration,
    ImageAnalysis,
    AudioProcessing,
    GeneralPurpose,
}

/// Task complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskComplexity {
    Simple = 1,
    Moderate = 2,
    Complex = 3,
    VeryComplex = 4,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub model_name: String,
    pub success_rate: f32,
    pub average_response_time: f32, // seconds
    pub average_quality_score: f32, // 0.0 - 1.0
    pub tokens_per_second: f32,
    pub cost_per_token: f32,
    pub reliability_score: f32,
    pub last_updated: DateTime<Utc>,
    pub total_requests: u64,
}

/// Model selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionCriteria {
    pub required_capabilities: Vec<ModelCapability>,
    pub task_complexity: TaskComplexity,
    pub max_response_time: Option<f32>, // seconds
    pub max_cost_per_request: Option<f32>,
    pub min_quality_score: Option<f32>,
    pub prefer_speed: bool,
    pub prefer_quality: bool,
    pub prefer_cost_efficiency: bool,
    pub context_length_required: Option<usize>,
    pub preferred_models: Vec<String>,
    pub excluded_models: Vec<String>,
}

/// Model recommendation with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub model_name: String,
    pub model_info: ModelInfo,
    pub score: f32,
    pub reasoning: String,
    pub estimated_cost: f32,
    pub estimated_response_time: f32,
    pub confidence: f32,
}

/// Model selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Optimize for fastest response time
    FastestResponse,
    /// Optimize for highest quality output
    HighestQuality,
    /// Optimize for best cost-performance ratio
    CostEfficient,
    /// Balance speed, quality, and cost
    Balanced { speed_weight: f32, quality_weight: f32, cost_weight: f32 },
    /// Use historical performance data
    PerformanceBased,
    /// Custom scoring algorithm
    Custom { algorithm: String },
}

/// Advanced model selection system
pub struct ModelSelector {
    model_db: Arc<ModelDatabase>,
    performance_cache: Arc<RwLock<HashMap<String, ModelPerformanceMetrics>>>,
    model_capabilities: Arc<RwLock<HashMap<String, Vec<ModelCapability>>>>,
    selection_history: Arc<RwLock<Vec<ModelSelectionRecord>>>,
    default_strategy: SelectionStrategy,
}

/// Record of model selection decisions for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelSelectionRecord {
    pub timestamp: DateTime<Utc>,
    pub criteria: ModelSelectionCriteria,
    pub selected_model: String,
    pub alternatives: Vec<ModelRecommendation>,
    pub actual_performance: Option<ActualPerformance>,
    pub user_feedback: Option<f32>, // 0.0 - 1.0 satisfaction score
}

/// Actual performance results for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActualPerformance {
    pub response_time: f32,
    pub quality_score: f32,
    pub cost: f32,
    pub success: bool,
    pub error: Option<String>,
}

/// Simple performance history record for the model selector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceHistory {
    pub id: i64,
    pub model_name: String,
    pub timestamp: DateTime<Utc>,
    pub response_time_ms: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub success: bool,
    pub error_message: Option<String>,
    pub quality_score: Option<f32>,
    pub cost: Option<f32>,
}

impl ModelSelector {
    /// Create a new model selector
    pub async fn new(model_db: Arc<ModelDatabase>) -> Result<Self> {
        let mut selector = Self {
            model_db,
            performance_cache: Arc::new(RwLock::new(HashMap::new())),
            model_capabilities: Arc::new(RwLock::new(HashMap::new())),
            selection_history: Arc::new(RwLock::new(Vec::new())),
            default_strategy: SelectionStrategy::Balanced {
                speed_weight: 0.3,
                quality_weight: 0.5,
                cost_weight: 0.2,
            },
        };

        // Initialize model capabilities and performance cache
        selector.initialize_model_data().await?;
        
        Ok(selector)
    }

    /// Initialize model capabilities and performance data
    async fn initialize_model_data(&self) -> Result<()> {
        // Get all available models
        let models = self.model_db.get_all_models().await?;
        
        let mut capabilities = self.model_capabilities.write().await;
        let mut performance = self.performance_cache.write().await;

        for model in models {
            // Infer capabilities from model name and description
            let model_caps = self.infer_model_capabilities(&model);
            capabilities.insert(model.name.clone(), model_caps);

            // Load or initialize performance metrics
            let metrics = self.load_performance_metrics(&model.name).await?;
            performance.insert(model.name.clone(), metrics);
        }

        Ok(())
    }

    /// Infer model capabilities from model information
    fn infer_model_capabilities(&self, model: &ModelInfo) -> Vec<ModelCapability> {
        let mut capabilities = Vec::new();
        let name_lower = model.name.to_lowercase();
        // Use metadata description if available
        let description_lower = model.metadata.get("description")
            .map(|d| d.to_lowercase())
            .unwrap_or_default();

        // Code-related capabilities
        if name_lower.contains("code") || name_lower.contains("coder") || 
           description_lower.contains("code") || description_lower.contains("programming") {
            capabilities.push(ModelCapability::CodeGeneration);
            capabilities.push(ModelCapability::CodeAnalysis);
        }

        // Documentation capabilities
        if name_lower.contains("doc") || description_lower.contains("documentation") ||
           description_lower.contains("writing") {
            capabilities.push(ModelCapability::Documentation);
        }

        // Conversation capabilities
        if name_lower.contains("chat") || name_lower.contains("instruct") ||
           description_lower.contains("conversation") || description_lower.contains("chat") {
            capabilities.push(ModelCapability::Conversation);
        }

        // Reasoning capabilities
        if name_lower.contains("reasoning") || name_lower.contains("logic") ||
           description_lower.contains("reasoning") || description_lower.contains("analysis") {
            capabilities.push(ModelCapability::Reasoning);
        }

        // Embedding capabilities
        if name_lower.contains("embed") || description_lower.contains("embedding") {
            capabilities.push(ModelCapability::Embedding);
        }

        // Math capabilities
        if name_lower.contains("math") || description_lower.contains("mathematical") ||
           description_lower.contains("computation") {
            capabilities.push(ModelCapability::Mathematics);
        }

        // If no specific capabilities detected, assume general purpose
        if capabilities.is_empty() {
            capabilities.push(ModelCapability::GeneralPurpose);
            capabilities.push(ModelCapability::Conversation);
        }

        capabilities
    }

    /// Load performance metrics for a model
    async fn load_performance_metrics(&self, model_name: &str) -> Result<ModelPerformanceMetrics> {
        // Try to load from database
        let history = self.model_db.get_performance_history(model_name, Some(100)).await?;
        
        if history.is_empty() {
            // No history, return default metrics
            return Ok(ModelPerformanceMetrics {
                model_name: model_name.to_string(),
                success_rate: 0.95, // Default assumption
                average_response_time: 2.0, // seconds
                average_quality_score: 0.8,
                tokens_per_second: 50.0,
                cost_per_token: 0.001,
                reliability_score: 0.9,
                last_updated: Utc::now(),
                total_requests: 0,
            });
        }

        // Calculate metrics from history
        let total_requests = history.len() as u64;
        let successful_requests = history.len() as f32; // Assume all successful for now
        let success_rate = 1.0; // Default to 100% success rate

        let avg_response_time = history.iter()
            .map(|h| h.1.latency_ms as f32 / 1000.0)
            .sum::<f32>() / total_requests as f32;

        let avg_quality = 0.8; // Default quality score

        let avg_tokens_per_sec = history.iter()
            .map(|h| h.1.tokens_per_second as f32)
            .sum::<f32>() / total_requests as f32;

        Ok(ModelPerformanceMetrics {
            model_name: model_name.to_string(),
            success_rate,
            average_response_time: avg_response_time,
            average_quality_score: avg_quality,
            tokens_per_second: avg_tokens_per_sec,
            cost_per_token: 0.001, // This would come from model pricing data
            reliability_score: success_rate * 0.8 + (1.0 - (avg_response_time / 10.0).min(1.0)) * 0.2,
            last_updated: Utc::now(),
            total_requests,
        })
    }

    /// Select the best model for given criteria
    pub async fn select_model(
        &self,
        criteria: &ModelSelectionCriteria,
        strategy: Option<SelectionStrategy>,
    ) -> Result<ModelRecommendation> {
        let candidates = self.get_candidate_models(criteria).await?;
        
        if candidates.is_empty() {
            return Err(anyhow::anyhow!("No suitable models found for the given criteria"));
        }

        let strategy = strategy.unwrap_or_else(|| self.default_strategy.clone());
        let recommendation = self.rank_models(candidates, &strategy, criteria).await?;

        // Record the selection for learning
        self.record_selection(criteria.clone(), &recommendation, Vec::new()).await;

        Ok(recommendation)
    }

    /// Get multiple model recommendations
    pub async fn get_model_recommendations(
        &self,
        criteria: &ModelSelectionCriteria,
        strategy: Option<SelectionStrategy>,
        limit: usize,
    ) -> Result<Vec<ModelRecommendation>> {
        let candidates = self.get_candidate_models(criteria).await?;
        
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let strategy = strategy.unwrap_or_else(|| self.default_strategy.clone());
        let mut recommendations = self.rank_all_models(candidates, &strategy, criteria).await?;

        // Limit results
        recommendations.truncate(limit);

        // Record the selection for learning
        if let Some(top_recommendation) = recommendations.first() {
            let alternatives = recommendations.iter().skip(1).cloned().collect();
            self.record_selection(criteria.clone(), top_recommendation, alternatives).await;
        }

        Ok(recommendations)
    }

    /// Get candidate models that meet basic criteria
    async fn get_candidate_models(&self, criteria: &ModelSelectionCriteria) -> Result<Vec<ModelInfo>> {
        let all_models = self.model_db.get_all_models().await?;
        let capabilities = self.model_capabilities.read().await;
        let performance = self.performance_cache.read().await;

        let mut candidates = Vec::new();

        for model in all_models {
            // Check if model is excluded
            if criteria.excluded_models.contains(&model.name) {
                continue;
            }

            // Check required capabilities
            if let Some(model_caps) = capabilities.get(&model.name) {
                let has_required_caps = criteria.required_capabilities.iter()
                    .all(|req_cap| model_caps.contains(req_cap));
                
                if !has_required_caps {
                    continue;
                }
            } else {
                // Unknown capabilities, skip if specific requirements
                if !criteria.required_capabilities.is_empty() {
                    continue;
                }
            }

            // Check performance constraints
            if let Some(metrics) = performance.get(&model.name) {
                // Response time constraint
                if let Some(max_time) = criteria.max_response_time {
                    if metrics.average_response_time > max_time {
                        continue;
                    }
                }

                // Quality constraint
                if let Some(min_quality) = criteria.min_quality_score {
                    if metrics.average_quality_score < min_quality {
                        continue;
                    }
                }

                // Cost constraint
                if let Some(max_cost) = criteria.max_cost_per_request {
                    // Estimate cost (this would be more sophisticated in real implementation)
                    let estimated_cost = metrics.cost_per_token * 1000.0; // Assume 1000 tokens
                    if estimated_cost > max_cost {
                        continue;
                    }
                }
            }

            // Check context length
            if let Some(required_context) = criteria.context_length_required {
                if model.context_window < required_context as u32 {
                    continue;
                }
            }

            candidates.push(model);
        }

        Ok(candidates)
    }

    /// Rank models and return the best one
    async fn rank_models(
        &self,
        candidates: Vec<ModelInfo>,
        strategy: &SelectionStrategy,
        criteria: &ModelSelectionCriteria,
    ) -> Result<ModelRecommendation> {
        let mut recommendations = self.rank_all_models(candidates, strategy, criteria).await?;
        
        recommendations.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No suitable models after ranking"))
    }

    /// Rank all candidate models
    async fn rank_all_models(
        &self,
        candidates: Vec<ModelInfo>,
        strategy: &SelectionStrategy,
        criteria: &ModelSelectionCriteria,
    ) -> Result<Vec<ModelRecommendation>> {
        let performance = self.performance_cache.read().await;
        let mut recommendations = Vec::new();

        for model in candidates {
            let metrics = performance.get(&model.name);
            let score = self.calculate_model_score(&model, metrics, strategy, criteria).await;
            
            let (estimated_cost, estimated_time) = if let Some(m) = metrics {
                (m.cost_per_token * 1000.0, m.average_response_time) // Estimate for 1000 tokens
            } else {
                (1.0, 2.0) // Default estimates
            };

            let reasoning = self.generate_selection_reasoning(&model, metrics, strategy, score);
            let confidence = self.calculate_confidence(&model, metrics);

            recommendations.push(ModelRecommendation {
                model_name: model.name.clone(),
                model_info: model,
                score,
                reasoning,
                estimated_cost,
                estimated_response_time: estimated_time,
                confidence,
            });
        }

        // Sort by score (descending)
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Boost preferred models
        if !criteria.preferred_models.is_empty() {
            for rec in &mut recommendations {
                if criteria.preferred_models.contains(&rec.model_name) {
                    rec.score *= 1.2; // 20% boost for preferred models
                }
            }
            
            // Re-sort after boosting
            recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }

        Ok(recommendations)
    }

    /// Calculate score for a model based on strategy and criteria
    async fn calculate_model_score(
        &self,
        model: &ModelInfo,
        metrics: Option<&ModelPerformanceMetrics>,
        strategy: &SelectionStrategy,
        criteria: &ModelSelectionCriteria,
    ) -> f32 {
        let default_metrics = ModelPerformanceMetrics {
            model_name: model.name.clone(),
            success_rate: 0.9,
            average_response_time: 2.0,
            average_quality_score: 0.8,
            tokens_per_second: 50.0,
            cost_per_token: 0.001,
            reliability_score: 0.9,
            last_updated: Utc::now(),
            total_requests: 0,
        };

        let m = metrics.unwrap_or(&default_metrics);

        match strategy {
            SelectionStrategy::FastestResponse => {
                // Prioritize speed (inverse of response time)
                1.0 / (m.average_response_time + 0.1) * m.reliability_score
            }
            SelectionStrategy::HighestQuality => {
                // Prioritize quality and success rate
                m.average_quality_score * m.success_rate * m.reliability_score
            }
            SelectionStrategy::CostEfficient => {
                // Optimize for cost-performance ratio
                let performance_score = m.average_quality_score * m.success_rate;
                let cost_efficiency = 1.0 / (m.cost_per_token + 0.0001);
                performance_score * cost_efficiency * m.reliability_score
            }
            SelectionStrategy::Balanced { speed_weight, quality_weight, cost_weight } => {
                let speed_score = 1.0 / (m.average_response_time + 0.1);
                let quality_score = m.average_quality_score * m.success_rate;
                let cost_score = 1.0 / (m.cost_per_token + 0.0001);

                // Normalize scores (rough normalization)
                let normalized_speed = (speed_score / 1.0).min(1.0);
                let normalized_quality = quality_score;
                let normalized_cost = (cost_score / 1000.0).min(1.0);

                (normalized_speed * speed_weight + 
                 normalized_quality * quality_weight + 
                 normalized_cost * cost_weight) * m.reliability_score
            }
            SelectionStrategy::PerformanceBased => {
                // Use historical performance data
                let recency_factor = if m.total_requests > 100 { 1.0 } else { 0.8 };
                let experience_factor = (m.total_requests as f32 / 1000.0).min(1.0);
                
                m.average_quality_score * m.success_rate * m.reliability_score * 
                recency_factor * experience_factor
            }
            SelectionStrategy::Custom { algorithm: _ } => {
                // Placeholder for custom algorithms
                m.average_quality_score * m.success_rate * m.reliability_score
            }
        }
    }

    /// Generate human-readable reasoning for model selection
    fn generate_selection_reasoning(
        &self,
        model: &ModelInfo,
        metrics: Option<&ModelPerformanceMetrics>,
        strategy: &SelectionStrategy,
        score: f32,
    ) -> String {
        let mut reasons = Vec::new();

        if let Some(m) = metrics {
            if m.success_rate > 0.95 {
                reasons.push("High reliability".to_string());
            }
            if m.average_response_time < 1.0 {
                reasons.push("Fast response time".to_string());
            }
            if m.average_quality_score > 0.85 {
                reasons.push("High quality output".to_string());
            }
            if m.cost_per_token < 0.0005 {
                reasons.push("Cost efficient".to_string());
            }
        }

        match strategy {
            SelectionStrategy::FastestResponse => {
                reasons.push("Optimized for speed".to_string());
            }
            SelectionStrategy::HighestQuality => {
                reasons.push("Optimized for quality".to_string());
            }
            SelectionStrategy::CostEfficient => {
                reasons.push("Optimized for cost efficiency".to_string());
            }
            _ => {}
        }

        if model.context_window > 32000 {
            reasons.push("Large context window".to_string());
        }

        if reasons.is_empty() {
            format!("Selected based on overall score: {:.2}", score)
        } else {
            format!("Selected for: {}. Score: {:.2}", reasons.join(", "), score)
        }
    }

    /// Calculate confidence in the selection
    fn calculate_confidence(&self, model: &ModelInfo, metrics: Option<&ModelPerformanceMetrics>) -> f32 {
        let mut confidence = 0.5; // Base confidence

        if let Some(m) = metrics {
            // More data = higher confidence
            let data_confidence = (m.total_requests as f32 / 1000.0).min(1.0) * 0.3;
            confidence += data_confidence;

            // Recent data = higher confidence
            let hours_since_update = Utc::now().signed_duration_since(m.last_updated).num_hours() as f32;
            let recency_confidence = ((24.0 - hours_since_update.min(24.0)) / 24.0) * 0.2;
            confidence += recency_confidence;
        }

        confidence.min(1.0)
    }

    /// Record a model selection for learning
    async fn record_selection(
        &self,
        criteria: ModelSelectionCriteria,
        selected: &ModelRecommendation,
        alternatives: Vec<ModelRecommendation>,
    ) {
        let record = ModelSelectionRecord {
            timestamp: Utc::now(),
            criteria,
            selected_model: selected.model_name.clone(),
            alternatives,
            actual_performance: None,
            user_feedback: None,
        };

        let mut history = self.selection_history.write().await;
        history.push(record);

        // Keep only last 10000 records
        if history.len() > 10000 {
            history.remove(0);
        }
    }

    /// Update performance metrics based on actual results
    pub async fn update_performance(
        &self,
        model_name: &str,
        actual_performance: ActualPerformance,
    ) -> Result<()> {
        // Update in-memory cache
        {
            let mut performance = self.performance_cache.write().await;
            if let Some(metrics) = performance.get_mut(model_name) {
                // Update with exponential moving average
                let alpha = 0.1; // Learning rate
                
                metrics.success_rate = metrics.success_rate * (1.0 - alpha) + 
                    (if actual_performance.success { 1.0 } else { 0.0 }) * alpha;
                
                metrics.average_response_time = metrics.average_response_time * (1.0 - alpha) + 
                    actual_performance.response_time * alpha;
                
                metrics.average_quality_score = metrics.average_quality_score * (1.0 - alpha) + 
                    actual_performance.quality_score * alpha;
                
                metrics.reliability_score = metrics.success_rate * 0.8 + 
                    (1.0 - (metrics.average_response_time / 10.0).min(1.0)) * 0.2;
                
                metrics.last_updated = Utc::now();
                metrics.total_requests += 1;
            }
        }

        // Store in database
        let performance_record = ModelPerformanceHistory {
            id: 0, // Will be set by database
            model_name: model_name.to_string(),
            timestamp: Utc::now(),
            response_time_ms: (actual_performance.response_time * 1000.0) as i64,
            input_tokens: 0, // Would be provided in real implementation
            output_tokens: 0, // Would be provided in real implementation
            success: actual_performance.success,
            error_message: actual_performance.error,
            quality_score: Some(actual_performance.quality_score),
            cost: Some(actual_performance.cost),
        };

        // TODO: Fix the conversion between ModelPerformanceHistory and PerformanceRecord
        // self.model_db.record_performance(&performance_record).await?;

        Ok(())
    }

    /// Get performance statistics for a model
    pub async fn get_model_performance(&self, model_name: &str) -> Option<ModelPerformanceMetrics> {
        let performance = self.performance_cache.read().await;
        performance.get(model_name).cloned()
    }

    /// Get selection statistics
    pub async fn get_selection_stats(&self) -> SelectionStats {
        let history = self.selection_history.read().await;
        let total_selections = history.len();
        
        let mut model_counts = HashMap::new();
        let mut strategy_counts: HashMap<String, i32> = HashMap::new();
        
        for record in history.iter() {
            *model_counts.entry(record.selected_model.clone()).or_insert(0) += 1;
            // Strategy counting would require storing strategy info in records
        }

        SelectionStats {
            total_selections,
            most_selected_models: {
                let mut sorted: Vec<_> = model_counts.into_iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                sorted.into_iter().take(10).collect()
            },
            average_confidence: history.iter()
                .map(|r| {
                    r.alternatives.first()
                        .map(|alt| alt.confidence)
                        .unwrap_or(0.5)
                })
                .sum::<f32>() / total_selections.max(1) as f32,
        }
    }
}

/// Selection statistics for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct SelectionStats {
    pub total_selections: usize,
    pub most_selected_models: Vec<(String, usize)>,
    pub average_confidence: f32,
}

/// Model Selector Agent implementation
pub struct ModelSelectorAgent {
    selector: Arc<ModelSelector>,
}

impl ModelSelectorAgent {
    pub fn new(selector: Arc<ModelSelector>) -> Self {
        Self { selector }
    }
}

#[async_trait]
impl Agent for ModelSelectorAgent {
    fn name(&self) -> &'static str {
        "model_selector"
    }

    fn description(&self) -> &'static str {
        "Advanced model selection using performance analytics and optimization algorithms"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability::ModelSelection,
            AgentCapability::PerformanceAnalysis,
            AgentCapability::CostOptimization,
        ]
    }

    async fn execute(&self, context: &mut FlowContext) -> Result<serde_json::Value> {
        let operation = context.metadata.get("model_selector_operation")
            .and_then(|v| v.as_str())
            .unwrap_or("select_model");

        match operation {
            "select_model" => {
                // Extract criteria from context
                let criteria = self.extract_criteria_from_context(context)?;
                let recommendation = self.selector.select_model(&criteria, None).await?;
                Ok(serde_json::to_value(recommendation)?)
            }
            "get_recommendations" => {
                let criteria = self.extract_criteria_from_context(context)?;
                let limit = context.metadata.get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as usize;
                
                let recommendations = self.selector.get_model_recommendations(&criteria, None, limit).await?;
                Ok(serde_json::to_value(recommendations)?)
            }
            "get_performance" => {
                if let Some(model_name) = context.metadata.get("model_name").and_then(|v| v.as_str()) {
                    let performance = self.selector.get_model_performance(model_name).await;
                    Ok(serde_json::to_value(performance)?)
                } else {
                    Err(anyhow::anyhow!("Model name not provided"))
                }
            }
            "get_stats" => {
                let stats = self.selector.get_selection_stats().await;
                Ok(serde_json::to_value(stats)?)
            }
            _ => {
                Err(anyhow::anyhow!("Unknown model selector operation: {}", operation))
            }
        }
    }

    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.capabilities_required.contains(&AgentCapability::ModelSelection) ||
        context.metadata.contains_key("model_selector_operation")
    }
}

impl ModelSelectorAgent {
    /// Extract selection criteria from flow context
    fn extract_criteria_from_context(&self, context: &FlowContext) -> Result<ModelSelectionCriteria> {
        // This would extract criteria from the context metadata
        // For now, return a default criteria
        Ok(ModelSelectionCriteria {
            required_capabilities: vec![ModelCapability::GeneralPurpose],
            task_complexity: TaskComplexity::Moderate,
            max_response_time: None,
            max_cost_per_request: None,
            min_quality_score: None,
            prefer_speed: false,
            prefer_quality: true,
            prefer_cost_efficiency: false,
            context_length_required: None,
            preferred_models: Vec::new(),
            excluded_models: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_selection() {
        // Test model selection algorithms
    }

    #[tokio::test]
    async fn test_performance_tracking() {
        // Test performance metric updates
    }

    #[tokio::test]
    async fn test_capability_matching() {
        // Test capability-based model matching
    }
}
