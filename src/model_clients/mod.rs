// src/model_clients/mod.rs
//! Model client implementations for different AI services

pub mod ollama_client;

pub use ollama_client::OllamaClient;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Trait for AI model clients that can generate text responses
#[async_trait]
pub trait ModelClient: Send + Sync {
    /// Generate a text response from a prompt
    async fn generate(&self, prompt: &str) -> Result<String>;
    
    /// Generate a response with additional context and parameters
    async fn generate_with_context(&self, request: &ModelRequest) -> Result<ModelResponse>;
    
    /// Get the name/identifier of the model
    fn model_name(&self) -> &str;
    
    /// Check if the model is available/healthy
    async fn health_check(&self) -> Result<bool>;
}

/// Request structure for model generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub context: Option<serde_json::Value>,
}

impl ModelRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            context: None,
        }
    }
    
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }
    
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
    
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }
    
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = Some(context);
        self
    }
}

/// Response from model generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: String,
    pub model_used: String,
    pub tokens_used: Option<u32>,
    pub finish_reason: Option<String>,
    pub usage_stats: Option<ModelUsageStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub cost_estimate: Option<f64>,
}

impl ModelResponse {
    pub fn new(content: impl Into<String>, model_used: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            model_used: model_used.into(),
            tokens_used: None,
            finish_reason: None,
            usage_stats: None,
        }
    }
    
    pub fn with_tokens(mut self, tokens: u32) -> Self {
        self.tokens_used = Some(tokens);
        self
    }
    
    pub fn with_usage_stats(mut self, stats: ModelUsageStats) -> Self {
        self.usage_stats = Some(stats);
        self
    }
}
