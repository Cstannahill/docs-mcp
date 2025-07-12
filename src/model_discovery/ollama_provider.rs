// src/model_discovery/ollama_provider.rs
//! Ollama-specific model discovery and information provider

use super::{ModelProvider, ModelInfo, PerformanceMetrics, QualityScores, ModelAvailability, CostInformation, DeploymentInfo};
use crate::agents::ModelCapability;
use crate::model_clients::{ModelClient, ollama_client::{OllamaClient, OllamaModelInfo}};
use crate::model_discovery::{
    ModelProviderType, ModelStrength, ModelWeakness, UseCase, DataFormat,
    BillingModel, DeploymentType, Platform, RateLimits
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use reqwest::Client;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

/// Ollama model provider for local model discovery
pub struct OllamaModelProvider {
    client: OllamaClient,
    http_client: Client,
    base_url: String,
    model_database: ModelInfoDatabase,
}

impl OllamaModelProvider {
    pub fn new(base_url: String) -> Result<Self> {
        let client = OllamaClient::new(&base_url, "dummy")?;
        let http_client = Client::new();
        let model_database = ModelInfoDatabase::new();
        
        Ok(Self {
            client,
            http_client,
            base_url,
            model_database,
        })
    }
    
    /// Get detailed information about a specific model from various sources
    async fn gather_comprehensive_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        info!("Gathering comprehensive information for model: {}", model_name);
        
        // 1. Get basic info from Ollama
        let _ollama_info = self.get_ollama_model_info(model_name).await?;
        
        // 2. Get detailed specs from model database
        let detailed_specs = self.model_database.get_model_specs(model_name);
        
        // 3. Test performance metrics
        let performance_metrics = self.measure_model_performance(model_name).await
            .unwrap_or_else(|e| {
                warn!("Failed to measure performance for {}: {}", model_name, e);
                self.get_estimated_performance(model_name)
            });
        
        // 4. Check availability
        let availability = self.check_model_availability(model_name).await?;
        
        // 5. Determine capabilities and strengths
        let (capabilities, strengths, weaknesses, use_cases) = self.analyze_model_capabilities(model_name);
        
        Ok(ModelInfo {
            id: format!("ollama:{}", model_name),
            name: model_name.to_string(),
            provider: ModelProviderType::Ollama,
            version: detailed_specs.version.clone(),
            model_family: detailed_specs.family.clone(),
            parameter_count: detailed_specs.parameter_count,
            context_window: detailed_specs.context_window,
            max_output_tokens: detailed_specs.max_output_tokens,
            architecture: detailed_specs.architecture.clone(),
            training_cutoff: detailed_specs.training_cutoff,
            capabilities,
            strengths,
            weaknesses,
            ideal_use_cases: use_cases,
            supported_formats: vec![DataFormat::Text, DataFormat::Code, DataFormat::Markdown],
            performance_metrics,
            quality_scores: self.get_quality_scores(model_name),
            benchmark_results: detailed_specs.benchmark_results.clone(),
            availability,
            cost_info: CostInformation {
                input_cost_per_token: None,
                output_cost_per_token: None,
                cost_per_request: None,
                free_tier_limits: None,
                billing_model: BillingModel::Free,
            },
            deployment_info: DeploymentInfo {
                deployment_type: DeploymentType::Local,
                region: None,
                endpoint: Some(self.base_url.clone()),
                model_size_gb: detailed_specs.model_size_gb,
                required_memory_gb: detailed_specs.required_memory_gb,
                supported_platforms: vec![Platform::Linux, Platform::MacOS, Platform::Windows],
            },
            discovered_at: Utc::now(),
            last_updated: Utc::now(),
            metadata: HashMap::new(),
        })
    }
    
    async fn get_ollama_model_info(&self, model_name: &str) -> Result<OllamaModelInfo> {
        // This would use the Ollama API to get model info
        debug!("Getting Ollama model info for: {}", model_name);
        
        let url = format!("{}/api/show", self.base_url);
        let response = self.http_client
            .post(&url)
            .json(&serde_json::json!({ "name": model_name }))
            .send()
            .await?;
        
        if response.status().is_success() {
            let info: OllamaModelInfo = response.json().await?;
            Ok(info)
        } else {
            Err(anyhow::anyhow!("Failed to get model info: {}", response.status()))
        }
    }
    
    async fn measure_model_performance(&self, model_name: &str) -> Result<PerformanceMetrics> {
        info!("Measuring performance for model: {}", model_name);
        
        // Create a client for this specific model
        let test_client = OllamaClient::new(&self.base_url, model_name)?;
        
        // Test prompt for performance measurement
        let test_prompt = "Write a simple 'Hello, World!' program in Python.";
        
        let start_time = std::time::Instant::now();
        let response = test_client.generate(test_prompt).await?;
        let total_time = start_time.elapsed();
        
        // Estimate tokens (rough approximation)
        let estimated_output_tokens = response.split_whitespace().count() as f64;
        let tokens_per_second = estimated_output_tokens / total_time.as_secs_f64();
        
        Ok(PerformanceMetrics {
            tokens_per_second,
            latency_ms: total_time.as_millis() as u64,
            throughput_requests_per_minute: 60.0 / total_time.as_secs_f64(),
            memory_usage_gb: None, // Would need system monitoring
            gpu_utilization_percent: None,
            cpu_utilization_percent: None,
            first_token_latency_ms: Some(total_time.as_millis() as u64 / 2), // Estimate
        })
    }
    
    fn get_estimated_performance(&self, model_name: &str) -> PerformanceMetrics {
        // Provide estimated performance based on model name patterns
        let (tokens_per_second, latency_ms) = if model_name.contains("7b") {
            (45.0, 1000)
        } else if model_name.contains("13b") {
            (25.0, 2000)
        } else if model_name.contains("3b") {
            (80.0, 500)
        } else if model_name.contains("1b") {
            (120.0, 300)
        } else {
            (35.0, 1500)
        };
        
        PerformanceMetrics {
            tokens_per_second,
            latency_ms,
            throughput_requests_per_minute: 60.0 / (latency_ms as f64 / 1000.0),
            memory_usage_gb: None,
            gpu_utilization_percent: None,
            cpu_utilization_percent: None,
            first_token_latency_ms: Some(latency_ms / 3),
        }
    }
    
    fn analyze_model_capabilities(&self, model_name: &str) -> (Vec<ModelCapability>, Vec<ModelStrength>, Vec<ModelWeakness>, Vec<UseCase>) {
        let name_lower = model_name.to_lowercase();
        
        let (capabilities, strengths, use_cases) = if name_lower.contains("coder") || name_lower.contains("code") {
            (
                vec![
                    ModelCapability::CodeGeneration,
                    ModelCapability::CodeUnderstanding,
                    ModelCapability::Debugging,
                    ModelCapability::PatternRecognition,
                    ModelCapability::Documentation,
                    ModelCapability::Explanation,
                ],
                vec![
                    ModelStrength::CodeGeneration,
                    ModelStrength::CodeAnalysis,
                    ModelStrength::TechnicalDocumentation,
                ],
                vec![
                    UseCase::CodeGeneration,
                    UseCase::CodeAnalysis,
                    UseCase::CodeReview,
                    UseCase::Debugging,
                    UseCase::TechnicalWriting,
                ]
            )
        } else if name_lower.contains("gemma") {
            (
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::Reasoning,
                    ModelCapability::CodeUnderstanding,
                    ModelCapability::PatternRecognition,
                    ModelCapability::Documentation,
                    ModelCapability::Explanation,
                ],
                vec![
                    ModelStrength::NaturalLanguageUnderstanding,
                    ModelStrength::ReasoningAndLogic,
                    ModelStrength::FollowsInstructions,
                ],
                vec![
                    UseCase::QuestionAnswering,
                    UseCase::Documentation,
                    UseCase::LearningAssistance,
                    UseCase::GeneralChat,
                ]
            )
        } else if name_lower.contains("llama") {
            (
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::Reasoning,
                    ModelCapability::CodeUnderstanding,
                    ModelCapability::PatternRecognition,
                    ModelCapability::Documentation,
                    ModelCapability::Explanation,
                    ModelCapability::Translation,
                ],
                vec![
                    ModelStrength::NaturalLanguageUnderstanding,
                    ModelStrength::ReasoningAndLogic,
                    ModelStrength::MultilingualSupport,
                    ModelStrength::FollowsInstructions,
                ],
                vec![
                    UseCase::QuestionAnswering,
                    UseCase::Documentation,
                    UseCase::Translation,
                    UseCase::LearningAssistance,
                    UseCase::GeneralChat,
                    UseCase::Summarization,
                ]
            )
        } else if name_lower.contains("zephyr") {
            (
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::Reasoning,
                    ModelCapability::CodeUnderstanding,
                    ModelCapability::Documentation,
                    ModelCapability::Explanation,
                ],
                vec![
                    ModelStrength::FollowsInstructions,
                    ModelStrength::SafetyAndAlignment,
                    ModelStrength::NaturalLanguageUnderstanding,
                ],
                vec![
                    UseCase::GeneralChat,
                    UseCase::QuestionAnswering,
                    UseCase::LearningAssistance,
                ]
            )
        } else {
            // Default capabilities for unknown models
            (
                vec![
                    ModelCapability::TextGeneration,
                    ModelCapability::Reasoning,
                    ModelCapability::Documentation,
                    ModelCapability::Explanation,
                ],
                vec![
                    ModelStrength::NaturalLanguageUnderstanding,
                ],
                vec![
                    UseCase::GeneralChat,
                    UseCase::QuestionAnswering,
                ]
            )
        };
        
        // Determine weaknesses based on model size and type
        let weaknesses = if name_lower.contains("3b") || name_lower.contains("1b") {
            vec![
                ModelWeakness::LimitedContext,
                ModelWeakness::FactualInconsistency,
            ]
        } else if name_lower.contains("embed") {
            vec![
                ModelWeakness::LimitedContext,
                ModelWeakness::InstructionMisunderstanding,
            ]
        } else {
            vec![
                ModelWeakness::SlowInference,
                ModelWeakness::HighMemoryUsage,
            ]
        };
        
        (capabilities, strengths, weaknesses, use_cases)
    }
    
    fn get_quality_scores(&self, model_name: &str) -> QualityScores {
        // Base quality scores with adjustments based on model characteristics
        let base_score = if model_name.contains("coder") || model_name.contains("code") {
            0.88
        } else if model_name.contains("gemma") {
            0.85
        } else if model_name.contains("llama") {
            0.82
        } else {
            0.75
        };
        
        let mut task_specific = HashMap::new();
        
        if model_name.contains("coder") || model_name.contains("code") {
            task_specific.insert("code_generation".to_string(), base_score + 0.05);
            task_specific.insert("code_analysis".to_string(), base_score + 0.03);
        }
        
        QualityScores {
            overall: base_score,
            accuracy: base_score + 0.02,
            coherence: base_score,
            relevance: base_score + 0.01,
            helpfulness: base_score,
            safety: base_score + 0.05,
            task_specific,
        }
    }
}

#[async_trait]
impl ModelProvider for OllamaModelProvider {
    fn name(&self) -> &str {
        "Ollama"
    }
    
    async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        info!("Discovering models from Ollama");
        
        let model_names = self.client.list_models().await
            .context("Failed to list models from Ollama")?;
        
        let mut models = Vec::new();
        
        for model_name in model_names {
            match self.gather_comprehensive_model_info(&model_name).await {
                Ok(model_info) => {
                    debug!("Successfully gathered info for model: {}", model_name);
                    models.push(model_info);
                }
                Err(e) => {
                    error!("Failed to gather info for model {}: {}", model_name, e);
                    // Continue with other models
                }
            }
        }
        
        info!("Successfully discovered {} models from Ollama", models.len());
        Ok(models)
    }
    
    async fn get_model_details(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        // Extract model name from ID (remove "ollama:" prefix if present)
        let model_name = model_id.strip_prefix("ollama:").unwrap_or(model_id);
        
        match self.gather_comprehensive_model_info(model_name).await {
            Ok(model_info) => Ok(Some(model_info)),
            Err(e) => {
                debug!("Model {} not found or error: {}", model_name, e);
                Ok(None)
            }
        }
    }
    
    async fn check_model_availability(&self, model_id: &str) -> Result<ModelAvailability> {
        let model_name = model_id.strip_prefix("ollama:").unwrap_or(model_id);
        
        let start_time = std::time::Instant::now();
        let is_available = self.client.is_model_available(model_name).await.unwrap_or(false);
        let response_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ModelAvailability {
            is_available,
            uptime_percent: if is_available { Some(99.0) } else { Some(0.0) },
            last_checked: Utc::now(),
            response_time_ms: Some(response_time),
            rate_limits: Some(RateLimits {
                requests_per_minute: Some(60), // Local models typically unlimited
                tokens_per_minute: None,
                concurrent_requests: Some(1), // Ollama typically single-threaded
            }),
        })
    }
}

/// Detailed model specifications database
struct ModelInfoDatabase {
    specs: HashMap<String, DetailedModelSpecs>,
}

#[derive(Debug, Clone)]
struct DetailedModelSpecs {
    version: Option<String>,
    family: String,
    parameter_count: Option<u64>,
    context_window: u32,
    max_output_tokens: Option<u32>,
    architecture: Option<String>,
    training_cutoff: Option<chrono::DateTime<Utc>>,
    model_size_gb: Option<f64>,
    required_memory_gb: Option<f64>,
    benchmark_results: Vec<crate::model_discovery::BenchmarkResult>,
}

impl ModelInfoDatabase {
    fn new() -> Self {
        let mut specs = HashMap::new();
        
        // Add known model specifications
        specs.insert("deepseek-coder:6.7b".to_string(), DetailedModelSpecs {
            version: Some("6.7B".to_string()),
            family: "DeepSeek".to_string(),
            parameter_count: Some(6_700_000_000),
            context_window: 16384,
            max_output_tokens: Some(4096),
            architecture: Some("Transformer".to_string()),
            training_cutoff: Some(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()),
            model_size_gb: Some(3.8),
            required_memory_gb: Some(8.0),
            benchmark_results: Vec::new(),
        });
        
        specs.insert("codellama:7b".to_string(), DetailedModelSpecs {
            version: Some("7B".to_string()),
            family: "Code Llama".to_string(),
            parameter_count: Some(7_000_000_000),
            context_window: 4096,
            max_output_tokens: Some(2048),
            architecture: Some("Llama".to_string()),
            training_cutoff: Some(Utc.with_ymd_and_hms(2023, 9, 1, 0, 0, 0).unwrap()),
            model_size_gb: Some(3.8),
            required_memory_gb: Some(8.0),
            benchmark_results: Vec::new(),
        });
        
        specs.insert("gemma2:9b".to_string(), DetailedModelSpecs {
            version: Some("9B".to_string()),
            family: "Gemma".to_string(),
            parameter_count: Some(9_000_000_000),
            context_window: 8192,
            max_output_tokens: Some(4096),
            architecture: Some("Gemma".to_string()),
            training_cutoff: Some(Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap()),
            model_size_gb: Some(5.4),
            required_memory_gb: Some(12.0),
            benchmark_results: Vec::new(),
        });
        
        specs.insert("llama3.2:3b".to_string(), DetailedModelSpecs {
            version: Some("3B".to_string()),
            family: "Llama".to_string(),
            parameter_count: Some(3_000_000_000),
            context_window: 131072,
            max_output_tokens: Some(4096),
            architecture: Some("Llama".to_string()),
            training_cutoff: Some(Utc.with_ymd_and_hms(2024, 9, 1, 0, 0, 0).unwrap()),
            model_size_gb: Some(2.0),
            required_memory_gb: Some(4.0),
            benchmark_results: Vec::new(),
        });
        
        Self { specs }
    }
    
    fn get_model_specs(&self, model_name: &str) -> DetailedModelSpecs {
        self.specs.get(model_name).cloned().unwrap_or_else(|| {
            // Provide default specs based on model name patterns
            let parameter_count = if model_name.contains("13b") {
                Some(13_000_000_000)
            } else if model_name.contains("7b") {
                Some(7_000_000_000)
            } else if model_name.contains("3b") {
                Some(3_000_000_000)
            } else if model_name.contains("1b") {
                Some(1_000_000_000)
            } else {
                None
            };
            
            let (model_size_gb, required_memory_gb) = match parameter_count {
                Some(p) if p >= 10_000_000_000 => (Some(7.0), Some(16.0)),
                Some(p) if p >= 5_000_000_000 => (Some(4.0), Some(8.0)),
                Some(p) if p >= 1_000_000_000 => (Some(2.0), Some(4.0)),
                _ => (Some(1.0), Some(2.0)),
            };
            
            DetailedModelSpecs {
                version: None,
                family: "Unknown".to_string(),
                parameter_count,
                context_window: 4096,
                max_output_tokens: Some(2048),
                architecture: Some("Transformer".to_string()),
                training_cutoff: None,
                model_size_gb,
                required_memory_gb,
                benchmark_results: Vec::new(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ollama_provider_creation() {
        let provider = OllamaModelProvider::new("http://localhost:11434".to_string());
        assert!(provider.is_ok());
    }
    
    #[test]
    fn test_model_info_database() {
        let db = ModelInfoDatabase::new();
        let specs = db.get_model_specs("deepseek-coder:6.7b");
        
        assert_eq!(specs.family, "DeepSeek");
        assert_eq!(specs.parameter_count, Some(6_700_000_000));
        assert_eq!(specs.context_window, 16384);
    }
}
