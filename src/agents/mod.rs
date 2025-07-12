// src/agents/mod.rs
//! Agentic Flow System for MCP Server
//! 
//! This module implements a sophisticated agent-based workflow system where:
//! 1. Each agent specializes in specific tasks (code analysis, documentation, etc.)
//! 2. An orchestrator decomposes complex requests into discrete tasks
//! 3. A model router selects the optimal AI model for each task
//! 4. Tasks are executed in optimized pipelines (parallel where possible)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;

pub mod code_analyzer;
// TODO: Implement additional agents
// pub mod doc_generator;
// pub mod test_generator; 
// pub mod security_auditor;
// pub mod performance_analyzer;

// Re-export model client types for convenience
pub use crate::model_clients::{ModelClient, ModelRequest, ModelResponse};

/// Core trait that all agents must implement
#[async_trait]
pub trait Agent: Send + Sync {
    /// Unique identifier for this agent
    fn name(&self) -> &'static str;
    
    /// What capabilities this agent provides
    fn capabilities(&self) -> Vec<AgentCapability>;
    
    /// JSON schema for expected input
    fn input_schema(&self) -> serde_json::Value;
    
    /// JSON schema for output format
    fn output_schema(&self) -> serde_json::Value;
    
    /// Execute the agent's core functionality
    async fn execute(
        &self,
        input: AgentInput,
        context: &FlowContext,
        model_client: Arc<dyn ModelClient>,
    ) -> Result<AgentOutput>;
    
    /// What model capabilities are required for this agent
    fn required_model_capabilities(&self) -> Vec<ModelCapability>;
    
    /// Estimate token usage for cost calculation
    fn estimated_tokens(&self, input: &AgentInput) -> usize;
    
    /// Validate input before execution
    fn validate_input(&self, input: &AgentInput) -> Result<()> {
        // Default implementation - agents can override for custom validation
        Ok(())
    }
    
    /// Agent-specific configuration
    fn default_config(&self) -> AgentConfig {
        AgentConfig::default()
    }
}

/// Categories of capabilities an agent can provide
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentCapability {
    CodeAnalysis,
    DocumentGeneration,
    TestCreation,
    SecurityAudit,
    PerformanceAnalysis,
    Debugging,
    Optimization,
    Translation,
    Explanation,
    Planning,
    Validation,
    ExampleGeneration,
    LearningPathCreation,
    ErrorDiagnosis,
    RefactoringAssistance,
    DataAnalysis,
}

/// Model capabilities that agents can require
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModelCapability {
    CodeUnderstanding,
    CodeGeneration,
    PatternRecognition,
    Documentation,
    ExampleGeneration,
    SecurityAnalysis,
    PerformanceAnalysis,
    Debugging,
    Translation,
    Explanation,
    Planning,
    Reasoning,       // For complex logic and inference
    LongContext,     // For large codebases
    FastInference,   // For real-time scenarios
    HighAccuracy,    // For critical tasks
    TextGeneration,  // For text generation tasks
    DataProcessing,  // For data analysis tasks
}

/// Input data structure for agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    fields: HashMap<String, serde_json::Value>,
    metadata: InputMetadata,
}

impl AgentInput {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: InputMetadata::default(),
        }
    }
    
    pub fn with_field<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.fields.insert(key.to_string(), serde_json::to_value(value).unwrap());
        self
    }
    
    pub fn get_field<T: for<'a> Deserialize<'a>>(&self, key: &str) -> Result<T> {
        let value = self.fields.get(key)
            .ok_or_else(|| anyhow::anyhow!("Field '{}' not found", key))?;
        Ok(serde_json::from_value(value.clone())?)
    }
    
    pub fn has_field(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }
    
    pub fn with_metadata(mut self, metadata: InputMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    
    pub fn metadata(&self) -> &InputMetadata {
        &self.metadata
    }
}

/// Output data structure from agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    fields: HashMap<String, serde_json::Value>,
    metadata: OutputMetadata,
    metrics: ExecutionMetrics,
}

impl AgentOutput {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            metadata: OutputMetadata::default(),
            metrics: ExecutionMetrics::default(),
        }
    }
    
    pub fn with_field<T: Serialize>(mut self, key: &str, value: T) -> Self {
        self.fields.insert(key.to_string(), serde_json::to_value(value).unwrap());
        self
    }
    
    pub fn get_field<T: for<'a> Deserialize<'a>>(&self, key: &str) -> Result<T> {
        let value = self.fields.get(key)
            .ok_or_else(|| anyhow::anyhow!("Field '{}' not found", key))?;
        Ok(serde_json::from_value(value.clone())?)
    }
    
    pub fn with_metrics(mut self, metrics: ExecutionMetrics) -> Self {
        self.metrics = metrics;
        self
    }
    
    pub fn metrics(&self) -> &ExecutionMetrics {
        &self.metrics
    }
}

/// Metadata about the input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMetadata {
    pub source: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub priority: TaskPriority,
    pub user_context: Option<UserContext>,
    pub project_context: Option<ProjectContext>,
}

impl Default for InputMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            timestamp: chrono::Utc::now(),
            priority: TaskPriority::Normal,
            user_context: None,
            project_context: None,
        }
    }
}

/// Metadata about the output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub agent_name: String,
    pub model_used: String,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub token_usage: TokenUsage,
}

impl Default for OutputMetadata {
    fn default() -> Self {
        Self {
            agent_name: "unknown".to_string(),
            model_used: "unknown".to_string(),
            confidence_score: 0.0,
            processing_time_ms: 0,
            token_usage: TokenUsage::default(),
        }
    }
}

/// Execution metrics for monitoring and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub tokens_used: TokenUsage,
    pub processing_time_ms: u64,
    pub model_calls: u32,
    pub cache_hits: u32,
    pub quality_score: f64,
    pub cost_estimate: f64,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            tokens_used: TokenUsage::default(),
            processing_time_ms: 0,
            model_calls: 0,
            cache_hits: 0,
            quality_score: 0.0,
            cost_estimate: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl Default for TokenUsage {
    fn default() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
        }
    }
}

/// Priority levels for task scheduling
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// User context for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub skill_level: SkillLevel,
    pub preferences: UserPreferences,
    pub learning_goals: Vec<String>,
    pub recent_topics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub explanation_style: ExplanationStyle,
    pub code_style: CodeStyle,
    pub verbosity: VerbosityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationStyle {
    Concise,
    Detailed,
    StepByStep,
    ExampleFocused,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CodeStyle {
    Functional,
    ObjectOriented,
    Procedural,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerbosityLevel {
    Minimal,
    Normal,
    Verbose,
    Comprehensive,
}

/// Project context for better understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    pub language: String,
    pub framework: Option<String>,
    pub project_type: ProjectType,
    pub dependencies: Vec<String>,
    pub patterns: Vec<String>,
    pub constraints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectType {
    WebApplication,
    Library,
    CommandLineTool,
    Game,
    SystemService,
    MobileApp,
    Desktop,
    Embedded,
    Other(String),
}

/// Configuration for agent behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_retries: u32,
    pub timeout_seconds: u64,
    pub quality_threshold: f64,
    pub enable_caching: bool,
    pub parallel_execution: bool,
    pub custom_settings: HashMap<String, serde_json::Value>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            timeout_seconds: 30,
            quality_threshold: 0.7,
            enable_caching: true,
            parallel_execution: false,
            custom_settings: HashMap::new(),
        }
    }
}

/// Context shared across all agents in a flow
#[derive(Debug, Clone)]
pub struct FlowContext {
    pub flow_id: String,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub intermediate_results: HashMap<String, AgentOutput>,
    pub global_context: HashMap<String, serde_json::Value>,
    pub user_context: Option<UserContext>,
    pub project_context: Option<ProjectContext>,
}

impl FlowContext {
    pub fn new(flow_id: String) -> Self {
        Self {
            flow_id,
            start_time: chrono::Utc::now(),
            intermediate_results: HashMap::new(),
            global_context: HashMap::new(),
            user_context: None,
            project_context: None,
        }
    }
    
    pub fn add_result(&mut self, task_id: &str, result: &AgentOutput) {
        self.intermediate_results.insert(task_id.to_string(), result.clone());
    }
    
    pub fn get_result(&self, task_id: &str) -> Option<&AgentOutput> {
        self.intermediate_results.get(task_id)
    }
    
    pub fn add_global_context<T: Serialize>(&mut self, key: &str, value: T) {
        self.global_context.insert(key.to_string(), serde_json::to_value(value).unwrap());
    }
    
    pub fn get_global_context<T: for<'a> Deserialize<'a>>(&self, key: &str) -> Option<T> {
        self.global_context.get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
    
    pub fn flow_id(&self) -> &str {
        &self.flow_id
    }
    
    pub fn elapsed_time(&self) -> chrono::Duration {
        chrono::Utc::now() - self.start_time
    }
}

/// Agent registry for managing available agents
pub struct AgentRegistry {
    agents: HashMap<String, Box<dyn Agent>>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }
    
    pub fn register<A: Agent + 'static>(&mut self, agent: A) {
        self.agents.insert(agent.name().to_string(), Box::new(agent));
    }
    
    pub fn get(&self, name: &str) -> Option<&dyn Agent> {
        self.agents.get(name).map(|a| a.as_ref())
    }
    
    pub fn get_by_capability(&self, capability: &AgentCapability) -> Vec<&dyn Agent> {
        self.agents
            .values()
            .filter(|agent| agent.capabilities().contains(capability))
            .map(|agent| agent.as_ref())
            .collect()
    }
    
    pub fn list_agents(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_input_creation() {
        let input = AgentInput::new()
            .with_field("code", "fn main() {}")
            .with_field("language", "rust");
        
        assert!(input.has_field("code"));
        assert!(input.has_field("language"));
        assert_eq!(input.get_field::<String>("language").unwrap(), "rust");
    }
    
    #[test]
    fn test_agent_output_creation() {
        let output = AgentOutput::new()
            .with_field("analysis", "Code looks good")
            .with_field("score", 0.95);
        
        assert_eq!(output.get_field::<String>("analysis").unwrap(), "Code looks good");
        assert_eq!(output.get_field::<f64>("score").unwrap(), 0.95);
    }
    
    #[test]
    fn test_flow_context() {
        let mut context = FlowContext::new("test-flow".to_string());
        let output = AgentOutput::new().with_field("result", "test");
        
        context.add_result("task1", &output);
        assert!(context.get_result("task1").is_some());
        
        context.add_global_context("project_type", "web_app");
        assert_eq!(
            context.get_global_context::<String>("project_type").unwrap(),
            "web_app"
        );
    }
}
