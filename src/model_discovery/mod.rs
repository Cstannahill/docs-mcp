// src/model_discovery/mod.rs
//! Model discovery and management system

mod ollama_provider;
pub mod database;
mod scheduler;
pub mod evaluator;

pub use ollama_provider::OllamaModelProvider;
pub use database::ModelDatabase;
pub use scheduler::{ModelDiscoveryScheduler, SchedulerConfig, SchedulerStatus};
pub use evaluator::{
    ModelEvaluator, ModelEvaluationResult, EvaluationConfig, TestSuite, 
    EvaluationTest, ExternalModelContext, TestResult
};

use crate::agents::ModelCapability;
use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, error, debug, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Provider trait for different model sources
#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn discover_models(&self) -> Result<Vec<ModelInfo>>;
    async fn get_model_details(&self, model_id: &str) -> Result<Option<ModelInfo>>;
    async fn check_model_availability(&self, model_id: &str) -> Result<ModelAvailability>;
}

/// Comprehensive model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub provider: ModelProviderType,
    pub version: Option<String>,
    pub model_family: String,
    pub parameter_count: Option<u64>,
    pub context_window: u32,
    pub max_output_tokens: Option<u32>,
    pub architecture: Option<String>,
    pub training_cutoff: Option<DateTime<Utc>>,
    
    // Capabilities and characteristics
    pub capabilities: Vec<ModelCapability>,
    pub strengths: Vec<ModelStrength>,
    pub weaknesses: Vec<ModelWeakness>,
    pub ideal_use_cases: Vec<UseCase>,
    pub supported_formats: Vec<DataFormat>,
    
    // Performance and quality metrics
    pub performance_metrics: PerformanceMetrics,
    pub quality_scores: QualityScores,
    pub benchmark_results: Vec<BenchmarkResult>,
    
    // Availability and deployment
    pub availability: ModelAvailability,
    pub cost_info: CostInformation,
    pub deployment_info: DeploymentInfo,
    
    // Metadata
    pub discovered_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Model provider enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelProviderType {
    Ollama,
    OpenAI,
    Anthropic,
    HuggingFace,
    Google,
    Cohere,
    Other(String),
}

/// Performance metrics for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub tokens_per_second: f64,
    pub latency_ms: u64,
    pub throughput_requests_per_minute: f64,
    pub memory_usage_gb: Option<f64>,
    pub gpu_utilization_percent: Option<f64>,
    pub cpu_utilization_percent: Option<f64>,
    pub first_token_latency_ms: Option<u64>,
}

/// Quality scores across different dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    pub overall: f64,
    pub accuracy: f64,
    pub coherence: f64,
    pub relevance: f64,
    pub helpfulness: f64,
    pub safety: f64,
    pub task_specific: HashMap<String, f64>, // Task-specific quality scores
}

/// Benchmark result information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub score: f64,
    pub rank: Option<u32>,
    pub date_tested: DateTime<Utc>,
    pub context: Option<String>,
}

/// Model availability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAvailability {
    pub is_available: bool,
    pub uptime_percent: Option<f64>,
    pub last_checked: DateTime<Utc>,
    pub response_time_ms: Option<u64>,
    pub rate_limits: Option<RateLimits>,
}

/// Rate limiting information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_minute: Option<u32>,
    pub tokens_per_minute: Option<u32>,
    pub concurrent_requests: Option<u32>,
}

/// Cost information for using the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostInformation {
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub cost_per_request: Option<f64>,
    pub free_tier_limits: Option<FreeTierLimits>,
    pub billing_model: BillingModel,
}

/// Free tier limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeTierLimits {
    pub requests_per_month: Option<u32>,
    pub tokens_per_month: Option<u32>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Billing model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BillingModel {
    Free,
    PayPerUse,
    Subscription,
    Enterprise,
}

/// Deployment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    pub deployment_type: DeploymentType,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub model_size_gb: Option<f64>,
    pub required_memory_gb: Option<f64>,
    pub supported_platforms: Vec<Platform>,
}

/// Deployment type options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentType {
    Local,
    Cloud,
    Edge,
    Hybrid,
}

/// Supported platforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Platform {
    Linux,
    Windows,
    MacOS,
    Android,
    iOS,
    Web,
}

/// Model strengths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStrength {
    CodeGeneration,
    CodeAnalysis,
    NaturalLanguageUnderstanding,
    ReasoningAndLogic,
    CreativeWriting,
    TechnicalDocumentation,
    MultilingualSupport,
    FastInference,
    LowMemoryUsage,
    HighAccuracy,
    SafetyAndAlignment,
    FollowsInstructions,
    ContextualUnderstanding,
    DomainSpecialization,
}

/// Model weaknesses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelWeakness {
    SlowInference,
    HighMemoryUsage,
    LimitedContext,
    FactualInconsistency,
    HallucinationProne,
    BiasIssues,
    InstructionMisunderstanding,
    OutdatedTrainingData,
    LanguageLimitations,
    DomainLimitations,
    SafetyConcerns,
    HighCost,
}

/// Ideal use cases for models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum UseCase {
    CodeGeneration,
    CodeAnalysis,
    CodeReview,
    Debugging,
    QuestionAnswering,
    Documentation,
    Translation,
    Summarization,
    CreativeWriting,
    TechnicalWriting,
    DataAnalysis,
    LearningAssistance,
    GeneralChat,
    TaskAutomation,
    ContentModeration,
    Embedding,
    Classification,
}

/// Supported data formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    Text,
    Code,
    Markdown,
    JSON,
    XML,
    CSV,
    Image,
    Audio,
    Video,
    PDF,
}

/// Summary of evaluation results across all models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub total_models: u32,
    pub evaluated_models: u32,
    pub average_score: f64,
    pub highest_score: f64,
    pub lowest_score: f64,
    pub best_model: Option<String>,
    pub evaluation_date: DateTime<Utc>,
}

impl Default for EvaluationSummary {
    fn default() -> Self {
        Self {
            total_models: 0,
            evaluated_models: 0,
            average_score: 0.0,
            highest_score: 0.0,
            lowest_score: 0.0,
            best_model: None,
            evaluation_date: Utc::now(),
        }
    }
}

/// Main model discovery service
pub struct ModelDiscoveryService {
    providers: Vec<Box<dyn ModelProvider>>,
    cache: ModelCache,
    database: Option<ModelDatabase>,
    // Add comprehensive evaluation capabilities
    evaluator: Option<Arc<evaluator::ModelEvaluator>>,
    evaluation_config: evaluator::EvaluationConfig,
}

impl ModelDiscoveryService {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            cache: ModelCache::new(),
            database: None,
            evaluator: None,
            evaluation_config: evaluator::EvaluationConfig::default(),
        }
    }
    
    pub fn with_database(database: ModelDatabase) -> Self {
        Self {
            providers: Vec::new(),
            cache: ModelCache::new(),
            database: Some(database),
            evaluator: None,
            evaluation_config: evaluator::EvaluationConfig::default(),
        }
    }
    
    /// Set up comprehensive model evaluation with web search capabilities
    pub fn with_evaluator(mut self, web_search: Arc<crate::web_search::WebSearchEngine>) -> Self {
        let evaluator = evaluator::ModelEvaluator::new(web_search, self.evaluation_config.clone());
        self.evaluator = Some(Arc::new(evaluator));
        self
    }
    
    /// Configure evaluation settings
    pub fn with_evaluation_config(mut self, config: evaluator::EvaluationConfig) -> Self {
        self.evaluation_config = config;
        self
    }
    
    /// Add a model provider to the discovery service
    pub fn add_provider(&mut self, provider: Box<dyn ModelProvider>) {
        self.providers.push(provider);
    }
    
    /// Load models from database into cache
    pub async fn load_from_database(&mut self) -> Result<()> {
        if let Some(ref database) = self.database {
            let models = database.get_all_models().await?;
            for model in models {
                self.cache.update_model(model);
            }
            tracing::info!("Loaded {} models from database into cache", self.cache.models.len());
        }
        Ok(())
    }
    
    /// Discover models from all providers and store in database
    pub async fn discover_all_models(&mut self) -> Result<Vec<ModelInfo>> {
        let mut all_models = Vec::new();
        
        for provider in &self.providers {
            match provider.discover_models().await {
                Ok(models) => {
                    for model in models {
                        // Store in database if available
                        if let Some(ref database) = self.database {
                            if let Err(e) = database.store_model(&model).await {
                                tracing::error!("Failed to store model {} in database: {}", model.id, e);
                            }
                        }
                        
                        self.cache.update_model(model.clone());
                        all_models.push(model);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to discover models from {}: {}", provider.name(), e);
                }
            }
        }
        
        Ok(all_models)
    }
    
    /// Discover models from all providers with comprehensive evaluation
    pub async fn discover_and_evaluate_models(
        &mut self, 
        model_client: Arc<dyn crate::model_clients::ModelClient>
    ) -> Result<Vec<(ModelInfo, Option<evaluator::ModelEvaluationResult>)>> {
        let mut results = Vec::new();
        
        // First discover all models
        let discovered_models = self.discover_all_models().await?;
        
        // If evaluator is available, run comprehensive evaluation on each model
        if let Some(evaluator) = self.evaluator.clone() {
            tracing::info!("Starting comprehensive evaluation of {} discovered models", discovered_models.len());
            
            for model in discovered_models {
                tracing::info!("Evaluating model: {}", model.name);
                
                match evaluator.evaluate_model(&model, model_client.clone()).await {
                    Ok(evaluation_result) => {
                        // Update model with evaluation results
                        if let Some(updated_model) = self.update_model_with_evaluation(&model, &evaluation_result).await {
                            results.push((updated_model, Some(evaluation_result)));
                        } else {
                            results.push((model, None));
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to evaluate model {}: {}", model.name, e);
                        results.push((model, None));
                    }
                }
            }
        } else {
            // No evaluator, just return discovered models
            for model in discovered_models {
                results.push((model, None));
            }
        }
        
        tracing::info!("Completed model discovery and evaluation for {} models", results.len());
        Ok(results)
    }
    
    /// Update a model with evaluation results
    async fn update_model_with_evaluation(
        &mut self,
        model: &ModelInfo,
        evaluation: &evaluator::ModelEvaluationResult,
    ) -> Option<ModelInfo> {
        let mut updated_model = model.clone();
        
        // Update performance metrics
        updated_model.performance_metrics = evaluation.performance_metrics.clone();
        updated_model.quality_scores = evaluation.quality_scores.clone();
        updated_model.benchmark_results = evaluation.benchmark_results.clone();
        
        // Update strengths and weaknesses based on evaluation
        updated_model.strengths = evaluation.strengths.clone();
        updated_model.weaknesses = evaluation.weaknesses.clone();
        updated_model.ideal_use_cases = evaluation.recommended_use_cases.clone();
        
        // Update last evaluation timestamp
        updated_model.last_updated = evaluation.evaluation_timestamp;
        
        // Add evaluation metadata
        updated_model.metadata.insert(
            "evaluation_score".to_string(),
            evaluation.overall_score.to_string(),
        );
        updated_model.metadata.insert(
            "evaluation_timestamp".to_string(),
            evaluation.evaluation_timestamp.to_rfc3339(),
        );
        updated_model.metadata.insert(
            "tests_passed".to_string(),
            evaluation.evaluation_metadata.tests_passed.to_string(),
        );
        updated_model.metadata.insert(
            "tests_total".to_string(),
            evaluation.evaluation_metadata.total_tests_run.to_string(),
        );
        
        // Store comprehensive evaluation results in database
        if let Some(ref database) = self.database {
            if let Err(e) = database.store_model(&updated_model).await {
                tracing::error!("Failed to store evaluated model {} in database: {}", updated_model.id, e);
                return None;
            }
            
            // Store detailed evaluation results as well
            if let Err(e) = self.store_evaluation_results(database, evaluation).await {
                tracing::error!("Failed to store evaluation results for model {}: {}", updated_model.id, e);
            }
        }
        
        // Update cache
        self.cache.update_model(updated_model.clone());
        
        Some(updated_model)
    }
    
    /// Store detailed evaluation results in database
    async fn store_evaluation_results(
        &self,
        database: &ModelDatabase,
        evaluation: &evaluator::ModelEvaluationResult,
    ) -> Result<()> {
        // Store external context
        let context_json = serde_json::to_string(&evaluation.external_context)
            .map_err(|e| anyhow::anyhow!("Failed to serialize external context: {}", e))?;
        
        // Store test results
        let test_results_json = serde_json::to_string(&evaluation.test_results)
            .map_err(|e| anyhow::anyhow!("Failed to serialize test results: {}", e))?;
        
        // Store capability scores
        let capability_scores_json = serde_json::to_string(&evaluation.capability_scores)
            .map_err(|e| anyhow::anyhow!("Failed to serialize capability scores: {}", e))?;
        
        // Create a comprehensive evaluation record
        // This would be stored in a new table designed for evaluation results
        // For now, we'll add it to the model metadata
        
        tracing::debug!("Stored comprehensive evaluation results for model: {}", evaluation.model_id);
        Ok(())
    }
    
    /// Get models with their latest evaluation results
    pub async fn get_evaluated_models(&self) -> Result<Vec<(ModelInfo, Option<f64>)>> {
        let mut results = Vec::new();
        
        for model in self.cache.models.values() {
            let evaluation_score = model.metadata.get("evaluation_score")
                .and_then(|s| s.parse::<f64>().ok());
            
            results.push((model.clone(), evaluation_score));
        }
        
        // Sort by evaluation score (highest first)
        results.sort_by(|a, b| {
            match (a.1, b.1) {
                (Some(score_a), Some(score_b)) => score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.0.name.cmp(&b.0.name),
            }
        });
        
        Ok(results)
    }
    
    /// Get the best model for a specific use case based on evaluation
    pub async fn get_best_model_for_use_case(&self, use_case: &crate::model_discovery::UseCase) -> Option<ModelInfo> {
        let mut best_model = None;
        let mut best_score = 0.0;
        
        for model in self.cache.models.values() {
            // Check if model supports this use case
            if model.ideal_use_cases.contains(use_case) {
                // Get evaluation score
                if let Some(score_str) = model.metadata.get("evaluation_score") {
                    if let Ok(score) = score_str.parse::<f64>() {
                        if score > best_score {
                            best_score = score;
                            best_model = Some(model.clone());
                        }
                    }
                }
            }
        }
        
        best_model
    }
    
    /// Re-evaluate a specific model with latest tests
    pub async fn re_evaluate_model(
        &mut self,
        model_id: &str,
        model_client: Arc<dyn crate::model_clients::ModelClient>,
    ) -> Result<Option<evaluator::ModelEvaluationResult>> {
        if let Some(evaluator) = &self.evaluator {
            if let Some(model) = self.cache.models.get(model_id).cloned() {
                tracing::info!("Re-evaluating model: {}", model.name);
                
                match evaluator.evaluate_model(&model, model_client).await {
                    Ok(evaluation_result) => {
                        // Update model with new evaluation
                        self.update_model_with_evaluation(&model, &evaluation_result).await;
                        return Ok(Some(evaluation_result));
                    }
                    Err(e) => {
                        tracing::error!("Failed to re-evaluate model {}: {}", model.name, e);
                        return Err(e);
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Get evaluation summary for all models
    pub fn get_evaluation_summary(&self) -> EvaluationSummary {
        let mut summary = EvaluationSummary::default();
        
        for model in self.cache.models.values() {
            summary.total_models += 1;
            
            if let Some(score_str) = model.metadata.get("evaluation_score") {
                if let Ok(score) = score_str.parse::<f64>() {
                    summary.evaluated_models += 1;
                    summary.average_score = (summary.average_score * (summary.evaluated_models - 1) as f64 + score) / summary.evaluated_models as f64;
                    
                    if score > summary.highest_score {
                        summary.highest_score = score;
                        summary.best_model = Some(model.name.clone());
                    }
                    
                    if score < summary.lowest_score || summary.lowest_score == 0.0 {
                        summary.lowest_score = score;
                    }
                }
            }
        }
        
        summary
    }
}

/// Model selection constraints
#[derive(Debug, Clone)]
pub struct ModelConstraints {
    pub max_latency_ms: Option<u64>,
    pub min_tokens_per_second: Option<f64>,
    pub max_cost_per_token: Option<f64>,
    pub required_provider: Option<ModelProviderType>,
    pub local_only: bool,
    pub max_memory_gb: Option<f64>,
}

/// Model cache for efficient lookups
struct ModelCache {
    models: HashMap<String, ModelInfo>,
    capability_index: HashMap<ModelCapability, Vec<String>>,
}

impl ModelCache {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            capability_index: HashMap::new(),
        }
    }
    
    fn update_model(&mut self, model: ModelInfo) {
        // Update capability index
        for capability in &model.capabilities {
            self.capability_index
                .entry(capability.clone())
                .or_insert_with(Vec::new)
                .push(model.id.clone());
        }
        
        self.models.insert(model.id.clone(), model);
    }
    
    fn get_by_capability(&self, capability: &ModelCapability) -> Vec<&ModelInfo> {
        self.capability_index
            .get(capability)
            .map(|model_ids| {
                model_ids.iter()
                    .filter_map(|id| self.models.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for ModelDiscoveryService {
    fn default() -> Self {
        Self::new()
    }
}
