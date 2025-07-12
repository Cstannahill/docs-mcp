// src/model_discovery/mod.rs
//! Model discovery and management system

mod ollama_provider;
pub mod database;
mod scheduler;

pub use ollama_provider::OllamaModelProvider;
pub use database::ModelDatabase;
pub use scheduler::{ModelDiscoveryScheduler, SchedulerConfig, SchedulerStatus};

use crate::agents::ModelCapability;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Main model discovery service
pub struct ModelDiscoveryService {
    providers: Vec<Box<dyn ModelProvider>>,
    cache: ModelCache,
    database: Option<ModelDatabase>,
}

impl ModelDiscoveryService {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            cache: ModelCache::new(),
            database: None,
        }
    }
    
    pub fn with_database(database: ModelDatabase) -> Self {
        Self {
            providers: Vec::new(),
            cache: ModelCache::new(),
            database: Some(database),
        }
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
    
    /// Update performance metrics for a model
    pub async fn update_model_performance(&mut self, model_id: &str, metrics: PerformanceMetrics) -> Result<()> {
        // Update in cache
        if let Some(model) = self.cache.models.get_mut(model_id) {
            model.performance_metrics = metrics.clone();
            model.last_updated = chrono::Utc::now();
            
            // Store in database
            if let Some(ref database) = self.database {
                database.store_performance_history(model_id, &metrics, chrono::Utc::now()).await?;
                database.store_model(model).await?;
            }
        }
        
        Ok(())
    }
    
    /// Update availability for a model
    pub async fn update_model_availability(&mut self, model_id: &str, availability: ModelAvailability) -> Result<()> {
        // Update in cache
        if let Some(model) = self.cache.models.get_mut(model_id) {
            model.availability = availability.clone();
            model.last_updated = chrono::Utc::now();
            
            // Store in database
            if let Some(ref database) = self.database {
                database.store_availability_history(model_id, &availability).await?;
                database.store_model(model).await?;
            }
        }
        
        Ok(())
    }
    
    /// Get models by capability
    pub fn get_models_by_capability(&self, capability: &ModelCapability) -> Vec<&ModelInfo> {
        self.cache.get_by_capability(capability)
    }
    
    /// Get models by provider
    pub fn get_models_by_provider(&self, provider: &ModelProviderType) -> Vec<&ModelInfo> {
        self.cache.models.values()
            .filter(|model| &model.provider == provider)
            .collect()
    }
    
    /// Get all cached models
    pub fn get_all_models(&self) -> Vec<&ModelInfo> {
        self.cache.models.values().collect()
    }
    
    /// Get model by ID
    pub fn get_model(&self, model_id: &str) -> Option<&ModelInfo> {
        self.cache.models.get(model_id)
    }
    
    /// Recommend best model for a specific task
    pub fn recommend_model(&self, 
        use_case: &UseCase, 
        capabilities: &[ModelCapability],
        constraints: Option<&ModelConstraints>
    ) -> Option<&ModelInfo> {
        let candidates: Vec<_> = self.cache.models.values()
            .filter(|model| {
                // Check if model supports all required capabilities
                capabilities.iter().all(|cap| model.capabilities.contains(cap)) &&
                // Check if model is suitable for the use case
                model.ideal_use_cases.contains(use_case) &&
                // Check availability
                model.availability.is_available &&
                // Check constraints
                constraints.map_or(true, |c| self.meets_constraints(model, c))
            })
            .collect();
        
        if candidates.is_empty() {
            return None;
        }
        
        // Score and rank candidates
        let mut scored_models: Vec<_> = candidates.into_iter()
            .map(|model| {
                let score = self.calculate_model_score(model, use_case, capabilities);
                (model, score)
            })
            .collect();
        
        scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        scored_models.first().map(|(model, _)| *model)
    }
    
    /// Get top N recommended models for a use case
    pub fn get_top_recommendations(&self, 
        use_case: &UseCase, 
        capabilities: &[ModelCapability],
        n: usize,
        constraints: Option<&ModelConstraints>
    ) -> Vec<(&ModelInfo, f64)> {
        let mut scored_models: Vec<_> = self.cache.models.values()
            .filter(|model| {
                capabilities.iter().all(|cap| model.capabilities.contains(cap)) &&
                model.availability.is_available &&
                constraints.map_or(true, |c| self.meets_constraints(model, c))
            })
            .map(|model| {
                let score = self.calculate_model_score(model, use_case, capabilities);
                (model, score)
            })
            .collect();
        
        scored_models.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_models.truncate(n);
        
        scored_models
    }
    
    fn meets_constraints(&self, model: &ModelInfo, constraints: &ModelConstraints) -> bool {
        if let Some(max_latency) = constraints.max_latency_ms {
            if model.performance_metrics.latency_ms > max_latency {
                return false;
            }
        }
        
        if let Some(min_tokens_per_second) = constraints.min_tokens_per_second {
            if model.performance_metrics.tokens_per_second < min_tokens_per_second {
                return false;
            }
        }
        
        if let Some(max_cost) = constraints.max_cost_per_token {
            if let Some(cost) = model.cost_info.output_cost_per_token {
                if cost > max_cost {
                    return false;
                }
            }
        }
        
        if let Some(required_provider) = &constraints.required_provider {
            if &model.provider != required_provider {
                return false;
            }
        }
        
        true
    }
    
    fn calculate_model_score(&self, 
        model: &ModelInfo, 
        use_case: &UseCase, 
        capabilities: &[ModelCapability]
    ) -> f64 {
        let mut score = model.quality_scores.overall;
        
        // Boost score for ideal use case match
        if model.ideal_use_cases.contains(use_case) {
            score += 0.2;
        }
        
        // Boost score for capability matches
        let capability_match_ratio = capabilities.iter()
            .filter(|cap| model.capabilities.contains(cap))
            .count() as f64 / capabilities.len() as f64;
        score += capability_match_ratio * 0.3;
        
        // Consider performance metrics
        if model.performance_metrics.tokens_per_second > 50.0 {
            score += 0.1;
        }
        
        // Penalize for known weaknesses that might affect the use case
        let weakness_penalty = model.weaknesses.len() as f64 * 0.02;
        score -= weakness_penalty;
        
        score.max(0.0).min(1.0)
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
