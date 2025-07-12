// src/model_clients/ollama_client.rs
//! Ollama client implementation for local LLM inference

use super::{ModelClient, ModelRequest, ModelResponse, ModelUsageStats};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Deserialize, Serialize, Debug)]
pub struct OllamaModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OllamaListResponse {
    pub models: Vec<OllamaModelInfo>,
}

/// Client for interacting with Ollama API
pub struct OllamaClient {
    base_url: String,
    model_name: String,
    client: reqwest::Client,
    timeout: Duration,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(base_url: impl Into<String>, model_name: impl Into<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120)) // 2 minute timeout for LLM calls
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            base_url: base_url.into(),
            model_name: model_name.into(),
            client,
            timeout: Duration::from_secs(120),
        })
    }
    
    /// Create with default localhost settings
    pub fn localhost(model_name: impl Into<String>) -> Result<Self> {
        Self::new("http://localhost:11434", model_name)
    }
    
    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// List all available models from Ollama
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        
        let response = self.client
            .get(&url)
            .timeout(self.timeout)
            .send()
            .await
            .context("Failed to send request to Ollama")?;

        if !response.status().is_success() {
            anyhow::bail!("Ollama API error: {}", response.status());
        }

        let list_response: OllamaListResponse = response
            .json()
            .await
            .context("Failed to parse Ollama list response")?;

        Ok(list_response.models.into_iter().map(|m| m.name).collect())
    }

    /// Check if a specific model is available
    pub async fn is_model_available(&self, model_name: &str) -> Result<bool> {
        let available_models = self.list_models().await?;
        Ok(available_models.iter().any(|m| m == model_name))
    }

    /// Pull a model if it's not available
    pub async fn ensure_model_available(&self) -> Result<()> {
        if self.is_model_available(&self.model_name).await? {
            log::debug!("Model {} is already available", self.model_name);
            return Ok(());
        }
        
        log::info!("Pulling model {} from Ollama...", self.model_name);
        
        let response = self.client
            .post(&format!("{}/api/pull", self.base_url))
            .json(&OllamaPullRequest {
                name: self.model_name.clone(),
                stream: false,
            })
            .timeout(Duration::from_secs(300)) // 5 minutes for model pull
            .send()
            .await
            .context("Failed to pull model from Ollama")?;
            
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to pull model {}: {}", self.model_name, error_text));
        }
        
        log::info!("Successfully pulled model {}", self.model_name);
        Ok(())
    }
}

#[async_trait]
impl ModelClient for OllamaClient {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let request = ModelRequest::new(prompt);
        let response = self.generate_with_context(&request).await?;
        Ok(response.content)
    }
    
    async fn generate_with_context(&self, request: &ModelRequest) -> Result<ModelResponse> {
        log::debug!("Generating with Ollama model: {}", self.model_name);
        
        // Build the prompt with system context if provided
        let full_prompt = match &request.system_prompt {
            Some(system) => format!("{}\n\nUser: {}", system, request.prompt),
            None => request.prompt.clone(),
        };
        
        let ollama_request = OllamaGenerateRequest {
            model: self.model_name.clone(),
            prompt: full_prompt,
            stream: false,
            options: OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens.map(|t| t as i32),
            },
            context: None, // TODO: Implement conversation context
        };
        
        let start_time = std::time::Instant::now();
        
        let response = self.client
            .post(&format!("{}/api/generate", self.base_url))
            .json(&ollama_request)
            .timeout(self.timeout)
            .send()
            .await
            .context("Failed to send request to Ollama")?;
            
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Ollama API error ({}): {}", 
                status, 
                error_text
            ));
        }
        
        let ollama_response: OllamaGenerateResponse = response
            .json()
            .await
            .context("Failed to parse Ollama response")?;
            
        let duration = start_time.elapsed();
        
        log::debug!("Ollama generation completed in {:?}", duration);
        
        // Estimate token usage (rough approximation)
        let estimated_prompt_tokens = (ollama_request.prompt.len() / 4) as u32;
        let estimated_completion_tokens = (ollama_response.response.len() / 4) as u32;
        let total_tokens = estimated_prompt_tokens + estimated_completion_tokens;
        
        Ok(ModelResponse {
            content: ollama_response.response,
            model_used: self.model_name.clone(),
            tokens_used: Some(total_tokens),
            finish_reason: Some(if ollama_response.done { "stop" } else { "length" }.to_string()),
            usage_stats: Some(ModelUsageStats {
                prompt_tokens: estimated_prompt_tokens,
                completion_tokens: estimated_completion_tokens,
                total_tokens,
                cost_estimate: Some(0.0), // Ollama is free locally
            }),
        })
    }
    
    fn model_name(&self) -> &str {
        &self.model_name
    }
    
    async fn health_check(&self) -> Result<bool> {
        let response = self.client
            .get(&format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .await;
            
        match response {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

// Ollama API types
#[derive(Debug, Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
    context: Option<Vec<i64>>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    done: bool,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    eval_duration: Option<u64>,
}

#[derive(Debug, Serialize)]
struct OllamaPullRequest {
    name: String,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    digest: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ollama_client_creation() {
        let client = OllamaClient::new("http://localhost:11434", "llama2");
        assert!(client.is_ok());
        
        let client = client.unwrap();
        assert_eq!(client.model_name(), "llama2");
        assert_eq!(client.base_url, "http://localhost:11434");
    }
    
    #[test]
    fn test_localhost_constructor() {
        let client = OllamaClient::localhost("codellama");
        assert!(client.is_ok());
        
        let client = client.unwrap();
        assert_eq!(client.model_name(), "codellama");
        assert_eq!(client.base_url, "http://localhost:11434");
    }
    
    #[tokio::test]
    async fn test_model_request_builder() {
        let request = ModelRequest::new("Test prompt")
            .with_system_prompt("You are a helpful assistant")
            .with_max_tokens(100)
            .with_temperature(0.7);
            
        assert_eq!(request.prompt, "Test prompt");
        assert_eq!(request.system_prompt, Some("You are a helpful assistant".to_string()));
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
    }
}
