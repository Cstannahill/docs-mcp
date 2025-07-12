// src/orchestrator/model_router.rs
//! Intelligent Model Router for Agentic Workflows
//! 
//! This module implements sophisticated model selection and routing logic that:
//! 1. Analyzes task requirements and agent capabilities
//! 2. Scores available models based on multiple factors
//! 3. Optimizes for performance, cost, and quality
//! 4. Provides fallback strategies and load balancing
//! 5. Learns from execution history to improve routing decisions

use crate::agents::{ModelCapability, TaskPriority};
use crate::orchestrator::{AgentAssignment, ModelAssignment, ModelSelection};
use crate::model_clients::ollama_client::OllamaClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

/// Advanced model router with intelligent selection capabilities
pub struct ModelRouter {
    model_registry: ModelRegistry,
    performance_tracker: ModelPerformanceTracker,
    cost_optimizer: CostOptimizer,
    load_balancer: LoadBalancer,
    routing_strategy: RoutingStrategy,
}

impl ModelRouter {
    pub fn new() -> Self {
        Self {
            model_registry: ModelRegistry::new(),
            performance_tracker: ModelPerformanceTracker::new(),
            cost_optimizer: CostOptimizer::new(),
            load_balancer: LoadBalancer::new(),
            routing_strategy: RoutingStrategy::default(),
        }
    }
    
    /// Initialize with dynamic discovery of available models
    pub async fn new_with_discovery() -> Result<Self> {
        let mut router = Self::new();
        router.discover_and_register_models().await?;
        Ok(router)
    }
    
    /// Discover available models from Ollama and register them
    pub async fn discover_and_register_models(&mut self) -> Result<()> {
        let client = OllamaClient::new("http://127.0.0.1:11434", "dummy")?;
        let available_models = client.list_models().await.map_err(|e| {
            error!("Failed to discover models from Ollama: {}", e);
            e
        })?;
        
        info!("Discovered {} models from Ollama: {:?}", available_models.len(), available_models);
        
        for model_name in available_models {
            let model_info = self.create_model_info_for(&model_name);
            self.model_registry.add_model(model_info);
            debug!("Registered model: {}", model_name);
        }
        
        Ok(())
    }
    
    /// Create ModelInfo based on model name heuristics
    fn create_model_info_for(&self, model_name: &str) -> ModelInfo {
        let name = model_name.to_string();
        
        // Determine capabilities based on model name patterns
        let capabilities = if model_name.contains("coder") || model_name.contains("code") {
            vec![
                ModelCapability::CodeGeneration,
                ModelCapability::CodeUnderstanding,
                ModelCapability::Debugging,
                ModelCapability::PerformanceAnalysis,
                ModelCapability::Documentation,
                ModelCapability::Explanation,
            ]
        } else if model_name.contains("gemma") {
            vec![
                ModelCapability::TextGeneration,
                ModelCapability::Reasoning,
                ModelCapability::PerformanceAnalysis,
                ModelCapability::Documentation,
                ModelCapability::Explanation,
            ]
        } else {
            vec![
                ModelCapability::TextGeneration,
                ModelCapability::Reasoning,
                ModelCapability::Documentation,
                ModelCapability::Explanation,
            ]
        };
        
        // Estimate specs based on model name
        let (tokens_per_second, max_context, quality_score) = if model_name.contains(":7b") || model_name.contains("7b") {
            (45.0, 4096, 0.85)
        } else if model_name.contains(":13b") || model_name.contains("13b") {
            (25.0, 8192, 0.90)
        } else if model_name.contains("6.7b") {
            (50.0, 8192, 0.88)
        } else {
            (35.0, 4096, 0.80)
        };
        
        let mut quality_metrics = HashMap::new();
        quality_metrics.insert("overall".to_string(), quality_score);
        if capabilities.contains(&ModelCapability::CodeGeneration) {
            quality_metrics.insert("code_generation".to_string(), quality_score + 0.05);
        }
        
        ModelInfo {
            name,
            capabilities,
            average_latency_ms: (1000.0 / tokens_per_second * 10.0) as u64,
            tokens_per_second,
            cost_per_token: 0.0, // Local models are free
            quality_metrics,
            max_context_length: max_context,
            available: true,
            current_load: 0.1,
        }
    }
    
    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.routing_strategy = strategy;
        self
    }
    
    /// Register a model with the internal registry
    pub fn register_model(&mut self, model_info: ModelInfo) {
        self.model_registry.add_model(model_info);
    }
    
    /// Route multiple tasks to optimal models with global optimization
    pub async fn route_tasks(&self, agent_assignments: &[AgentAssignment]) -> Result<Vec<ModelAssignment>> {
        let mut assignments = Vec::new();
        
        // First pass: Individual optimal selection for each task
        for assignment in agent_assignments {
            // Extract required capabilities from task
            let required_capabilities = self.extract_required_capabilities(assignment)?;
            let model_selection = self.select_optimal_model(&required_capabilities, &assignment.task).await?;
            
            assignments.push(ModelAssignment {
                task_id: assignment.task_id.clone(),
                agent_name: assignment.agent_name.clone(),
                model_selection,
                estimated_cost: 0.0, // Will be calculated
                estimated_time_ms: 0, // Will be calculated
                dependencies: assignment.dependencies.clone(),
            });
        }
        
        // Second pass: Global optimization across all assignments
        self.optimize_global_assignments(&mut assignments).await?;
        
        // Third pass: Calculate final estimates and validate
        self.finalize_assignments(&mut assignments).await?;
        
        Ok(assignments)
    }
    
    /// Select the optimal model for a specific agent-task combination
    pub async fn select_optimal_model(&self, 
                                      required_capabilities: &[crate::agents::ModelCapability],
                                      task: &crate::orchestrator::Task) -> Result<ModelSelection> {
        // Find models that can handle these capabilities
        let candidate_models = self.model_registry.get_capable_models(&required_capabilities);
        
        if candidate_models.is_empty() {
            return Err(anyhow::anyhow!(
                "No models available with required capabilities: {:?}",
                required_capabilities
            ));
        }
        
        // Score each candidate model
        let mut scored_models = Vec::new();
        for model in candidate_models {
            let score = self.calculate_model_score(&model, task).await?;
            scored_models.push((model, score));
        }
        
        // Sort by score (highest first) and apply routing strategy
        scored_models.sort_by(|a, b| b.1.total_score.partial_cmp(&a.1.total_score).unwrap());
        
        let selected = self.apply_routing_strategy(&scored_models)?;
        
        Ok(ModelSelection {
            model_name: selected.0.name.clone(),
            confidence_score: selected.1.total_score,
            selection_reasoning: self.generate_selection_reasoning(&selected.0, &selected.1),
            fallback_models: scored_models.iter()
                .skip(1)
                .take(3)
                .map(|(model, _)| model.name.clone())
                .collect(),
        })
    }
    
    /// Extract required model capabilities from an agent assignment
    fn extract_required_capabilities(&self, assignment: &AgentAssignment) -> Result<Vec<crate::agents::ModelCapability>> {
        let mut required_model_caps = Vec::new();
        
        // Map agent capabilities to model capabilities
        for agent_cap in &assignment.task.required_capabilities {
            match agent_cap {
                crate::agents::AgentCapability::CodeAnalysis => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::CodeUnderstanding,
                        crate::agents::ModelCapability::PatternRecognition,
                    ]);
                }
                crate::agents::AgentCapability::DocumentGeneration => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::Documentation,
                        crate::agents::ModelCapability::Explanation,
                    ]);
                }
                crate::agents::AgentCapability::Debugging => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::Debugging,
                        crate::agents::ModelCapability::CodeUnderstanding,
                    ]);
                }
                crate::agents::AgentCapability::TestCreation => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::CodeGeneration,
                        crate::agents::ModelCapability::CodeUnderstanding,
                    ]);
                }
                crate::agents::AgentCapability::SecurityAudit => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::SecurityAnalysis,
                        crate::agents::ModelCapability::PatternRecognition,
                    ]);
                }
                crate::agents::AgentCapability::PerformanceAnalysis => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::PerformanceAnalysis,
                        crate::agents::ModelCapability::CodeUnderstanding,
                    ]);
                }
                crate::agents::AgentCapability::Explanation => {
                    required_model_caps.push(crate::agents::ModelCapability::Explanation);
                }
                crate::agents::AgentCapability::Validation => {
                    required_model_caps.extend(vec![
                        crate::agents::ModelCapability::HighAccuracy,
                        crate::agents::ModelCapability::Reasoning,
                    ]);
                }
                // Add more mappings as needed
                _ => {
                    // Default fallback
                    required_model_caps.push(crate::agents::ModelCapability::Reasoning);
                }
            }
        }
        
        // Remove duplicates
        required_model_caps.sort();
        required_model_caps.dedup();
        
        Ok(required_model_caps)
    }
    
    /// Calculate comprehensive score for a model-task combination
    async fn calculate_model_score(&self, model: &ModelInfo, task: &crate::orchestrator::Task) -> Result<ModelScore> {
        let performance_score = self.performance_tracker
            .get_performance_score(&model.name, "default_agent") // TODO: Get actual agent name
            .await?;
        
        let capability_score = self.calculate_capability_match_score(model, task);
        let cost_score = self.cost_optimizer.calculate_cost_efficiency_score(model, task);
        let availability_score = self.load_balancer.get_availability_score(&model.name).await?;
        let latency_score = self.calculate_latency_score(model, task);
        let quality_score = self.calculate_expected_quality_score(model, task).await?;
        
        // Apply weights based on routing strategy
        let weights = &self.routing_strategy.scoring_weights;
        
        let total_score = 
            performance_score * weights.performance +
            capability_score * weights.capability_match +
            cost_score * weights.cost_efficiency +
            availability_score * weights.availability +
            latency_score * weights.latency +
            quality_score * weights.quality;
        
        Ok(ModelScore {
            total_score,
            performance_score,
            capability_score,
            cost_score,
            availability_score,
            latency_score,
            quality_score,
        })
    }
    
    fn calculate_capability_match_score(&self, model: &ModelInfo, task: &crate::orchestrator::Task) -> f64 {
        let required_caps = &task.required_capabilities;
        
        if required_caps.is_empty() {
            return 1.0; // No specific requirements
        }

        // Convert agent capabilities to model capabilities
        let model_caps: Vec<crate::agents::ModelCapability> = required_caps.iter()
            .filter_map(|agent_cap| {
                match agent_cap {
                    crate::agents::AgentCapability::CodeAnalysis => Some(crate::agents::ModelCapability::CodeUnderstanding),
                    crate::agents::AgentCapability::DocumentGeneration => Some(crate::agents::ModelCapability::TextGeneration),
                    crate::agents::AgentCapability::Debugging => Some(crate::agents::ModelCapability::CodeUnderstanding),
                    crate::agents::AgentCapability::TestCreation => Some(crate::agents::ModelCapability::CodeGeneration),
                    crate::agents::AgentCapability::DataAnalysis => Some(crate::agents::ModelCapability::DataProcessing),
                    crate::agents::AgentCapability::Optimization => Some(crate::agents::ModelCapability::Reasoning),
                    _ => None,
                }
            })
            .collect();
        
        if model_caps.is_empty() {
            return 0.8; // Default score for generic capabilities
        }
        
        let matches = model_caps.iter()
            .filter(|cap| model.capabilities.contains(cap))
            .count();
        
        let exact_match_score = matches as f64 / model_caps.len() as f64;
        
        // Bonus for having additional relevant capabilities
        let bonus_capabilities = model.capabilities.iter()
            .filter(|cap| !model_caps.contains(cap))
            .count();
        
        let bonus_score = (bonus_capabilities as f64 * 0.1).min(0.2);
        
        (exact_match_score + bonus_score).min(1.0)
    }
    
    fn calculate_latency_score(&self, model: &ModelInfo, task: &crate::orchestrator::Task) -> f64 {
        let priority_weight = match task.priority {
            TaskPriority::Critical => 1.0,
            TaskPriority::High => 0.8,
            TaskPriority::Normal => 0.5,
            TaskPriority::Low => 0.2,
        };
        
        // Lower latency = higher score
        let base_score = 1.0 / (model.average_latency_ms as f64 / 1000.0 + 1.0);
        
        base_score * priority_weight
    }
    
    async fn calculate_expected_quality_score(&self, model: &ModelInfo, task: &crate::orchestrator::Task) -> Result<f64> {
        // Base quality score from model specifications
        let base_quality = model.quality_metrics.get("overall")
            .unwrap_or(&0.7) // Default quality score
            .clone();
        
        // Adjust based on task complexity
        let complexity_adjustment = match task.estimated_complexity {
            crate::orchestrator::TaskComplexity::Simple => 0.1,
            crate::orchestrator::TaskComplexity::Moderate => 0.0,
            crate::orchestrator::TaskComplexity::Complex => -0.1,
            crate::orchestrator::TaskComplexity::Intensive => -0.2,
        };
        
        // Historical performance adjustment
        let historical_adjustment = self.performance_tracker
            .get_quality_trend(&model.name, "default_agent") // TODO: derive agent name from task
            .await
            .unwrap_or(0.0);
        
        Ok((base_quality + complexity_adjustment + historical_adjustment).clamp(0.0, 1.0))
    }
    
    fn apply_routing_strategy(&self, scored_models: &[(ModelInfo, ModelScore)]) -> Result<(ModelInfo, ModelScore)> {
        match &self.routing_strategy.selection_method {
            SelectionMethod::HighestScore => {
                scored_models.first()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("No models available"))
            }
            SelectionMethod::WeightedRandom => {
                self.weighted_random_selection(scored_models)
            }
            SelectionMethod::RoundRobin => {
                self.round_robin_selection(scored_models)
            }
            SelectionMethod::LoadBalanced => {
                self.load_balanced_selection(scored_models)
            }
        }
    }
    
    fn weighted_random_selection(&self, scored_models: &[(ModelInfo, ModelScore)]) -> Result<(ModelInfo, ModelScore)> {
        use rand::Rng;
        
        let total_weight: f64 = scored_models.iter()
            .map(|(_, score)| score.total_score)
            .sum();
        
        if total_weight <= 0.0 {
            return scored_models.first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No models available"));
        }
        
        let mut rng = rand::thread_rng();
        let random_value = rng.gen::<f64>() * total_weight;
        
        let mut cumulative_weight = 0.0;
        for (model, score) in scored_models {
            cumulative_weight += score.total_score;
            if random_value <= cumulative_weight {
                return Ok((model.clone(), score.clone()));
            }
        }
        
        // Fallback to first model
        scored_models.first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No models available"))
    }
    
    fn round_robin_selection(&self, scored_models: &[(ModelInfo, ModelScore)]) -> Result<(ModelInfo, ModelScore)> {
        // Simple rotation through available models
        let index = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize % scored_models.len();
        Ok(scored_models[index].clone())
    }
    
    fn load_balanced_selection(&self, scored_models: &[(ModelInfo, ModelScore)]) -> Result<(ModelInfo, ModelScore)> {
        // Select the model with the best combination of score and low current load
        scored_models.iter()
            .min_by(|(_, score_a), (_, score_b)| {
                let combined_score_a = score_a.total_score * score_a.availability_score;
                let combined_score_b = score_b.total_score * score_b.availability_score;
                combined_score_b.partial_cmp(&combined_score_a).unwrap()
            })
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No models available"))
    }
    
    fn generate_selection_reasoning(&self, model: &ModelInfo, score: &ModelScore) -> String {
        let mut reasons = Vec::new();
        
        if score.capability_score > 0.9 {
            reasons.push("Perfect capability match".to_string());
        } else if score.capability_score > 0.7 {
            reasons.push("Good capability match".to_string());
        }
        
        if score.performance_score > 0.8 {
            reasons.push("Excellent historical performance".to_string());
        }
        
        if score.cost_score > 0.8 {
            reasons.push("Cost-efficient choice".to_string());
        }
        
        if score.latency_score > 0.8 {
            reasons.push("Fast response time".to_string());
        }
        
        if score.quality_score > 0.9 {
            reasons.push("High quality output expected".to_string());
        }
        
        if reasons.is_empty() {
            format!("Selected based on overall score: {:.2}", score.total_score)
        } else {
            format!("Selected for: {}. Overall score: {:.2}", 
                reasons.join(", "), score.total_score)
        }
    }
    
    /// Global optimization across all task assignments
    async fn optimize_global_assignments(&self, assignments: &mut [ModelAssignment]) -> Result<()> {
        match &self.routing_strategy.optimization_strategy {
            OptimizationStrategy::None => Ok(()),
            OptimizationStrategy::CostMinimization => {
                self.cost_optimizer.minimize_total_cost(assignments).await
            }
            OptimizationStrategy::LatencyMinimization => {
                self.minimize_total_latency(assignments).await
            }
            OptimizationStrategy::LoadBalancing => {
                self.load_balancer.balance_assignments(assignments).await
            }
            OptimizationStrategy::Balanced => {
                self.apply_balanced_optimization(assignments).await
            }
        }
    }
    
    async fn minimize_total_latency(&self, assignments: &mut [ModelAssignment]) -> Result<()> {
        // Look for opportunities to parallelize or use faster models
        // This is a simplified implementation
        
        for assignment in assignments.iter_mut() {
            // If this is a high-priority task, consider upgrading to a faster model
            if matches!(assignment.task_id.as_str(), "critical" | "high_priority") {
                if let Some(faster_model) = self.find_faster_alternative(&assignment.model_selection.model_name) {
                    assignment.model_selection.model_name = faster_model;
                    assignment.model_selection.selection_reasoning = 
                        "Upgraded to faster model for latency optimization".to_string();
                }
            }
        }
        
        Ok(())
    }
    
    fn find_faster_alternative(&self, current_model: &str) -> Option<String> {
        // In practice, this would consult the model registry
        match current_model {
            "llama3.2:8b" => Some("llama3.2:3b".to_string()), // Smaller, faster version
            "codellama:13b" => Some("codellama:7b".to_string()),
            _ => None,
        }
    }
    
    async fn apply_balanced_optimization(&self, assignments: &mut [ModelAssignment]) -> Result<()> {
        // Apply a combination of optimization strategies with weights
        
        // 40% cost optimization
        self.cost_optimizer.minimize_total_cost(assignments).await?;
        
        // 30% latency optimization  
        self.minimize_total_latency(assignments).await?;
        
        // 30% load balancing
        self.load_balancer.balance_assignments(assignments).await?;
        
        Ok(())
    }
    
    /// Finalize assignments with cost and time estimates
    async fn finalize_assignments(&self, assignments: &mut [ModelAssignment]) -> Result<()> {
        for assignment in assignments.iter_mut() {
            let model_info = self.model_registry.get_model(&assignment.model_selection.model_name)?;
            
            // Calculate estimated cost
            assignment.estimated_cost = self.cost_optimizer
                .estimate_task_cost(&assignment.task_id, &model_info);
            
            // Calculate estimated time
            assignment.estimated_time_ms = self.estimate_execution_time(&assignment.task_id, &model_info);
        }
        
        Ok(())
    }
    
    fn estimate_execution_time(&self, task_id: &str, model: &ModelInfo) -> u64 {
        // Base time + model latency + complexity factor
        let base_time = 1000; // 1 second base
        let model_latency = model.average_latency_ms;
        let complexity_factor = match task_id.contains("complex") {
            true => 3,
            false => 1,
        };
        
        (base_time + model_latency) * complexity_factor
    }
    
}

/// Model registry maintaining information about available models
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }
    
    pub fn add_model(&mut self, model_info: ModelInfo) {
        self.models.insert(model_info.name.clone(), model_info);
    }
    
    pub fn get_capable_models(&self, required_capabilities: &[ModelCapability]) -> Vec<ModelInfo> {
        self.models
            .values()
            .filter(|model| {
                model.available && 
                required_capabilities.iter()
                    .all(|cap| model.capabilities.contains(cap))
            })
            .cloned()
            .collect()
    }
    
    pub fn get_model(&self, name: &str) -> Result<ModelInfo> {
        self.models.get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", name))
    }
    
    pub fn update_model_load(&mut self, model_name: &str, load: f64) -> Result<()> {
        if let Some(model) = self.models.get_mut(model_name) {
            model.current_load = load.clamp(0.0, 1.0);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Model not found: {}", model_name))
        }
    }
}

/// Information about a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub capabilities: Vec<ModelCapability>,
    pub average_latency_ms: u64,
    pub tokens_per_second: f64,
    pub cost_per_token: f64,
    pub quality_metrics: HashMap<String, f64>,
    pub max_context_length: usize,
    pub available: bool,
    pub current_load: f64, // 0.0 to 1.0
}

/// Comprehensive score for a model-task combination
#[derive(Debug, Clone)]
pub struct ModelScore {
    pub total_score: f64,
    pub performance_score: f64,
    pub capability_score: f64,
    pub cost_score: f64,
    pub availability_score: f64,
    pub latency_score: f64,
    pub quality_score: f64,
}

/// Strategy configuration for model routing
#[derive(Debug, Clone)]
pub struct RoutingStrategy {
    pub selection_method: SelectionMethod,
    pub optimization_strategy: OptimizationStrategy,
    pub scoring_weights: ScoringWeights,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self {
            selection_method: SelectionMethod::HighestScore,
            optimization_strategy: OptimizationStrategy::Balanced,
            scoring_weights: ScoringWeights::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SelectionMethod {
    HighestScore,
    WeightedRandom,
    RoundRobin,
    LoadBalanced,
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    None,
    CostMinimization,
    LatencyMinimization,
    LoadBalancing,
    Balanced,
}

#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub performance: f64,
    pub capability_match: f64,
    pub cost_efficiency: f64,
    pub availability: f64,
    pub latency: f64,
    pub quality: f64,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            performance: 0.25,
            capability_match: 0.25,
            cost_efficiency: 0.15,
            availability: 0.10,
            latency: 0.15,
            quality: 0.10,
        }
    }
}

/// Tracks model performance over time
pub struct ModelPerformanceTracker {
    performance_history: HashMap<String, ModelPerformanceHistory>,
}

impl ModelPerformanceTracker {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
        }
    }
    
    pub async fn get_performance_score(&self, model_name: &str, agent_name: &str) -> Result<f64> {
        let key = format!("{}:{}", model_name, agent_name);
        
        if let Some(history) = self.performance_history.get(&key) {
            Ok(history.average_score())
        } else {
            Ok(0.7) // Default score for new combinations
        }
    }
    
    pub async fn get_quality_trend(&self, model_name: &str, agent_name: &str) -> Option<f64> {
        let key = format!("{}:{}", model_name, agent_name);
        
        self.performance_history.get(&key)
            .map(|history| history.quality_trend())
    }
    
    pub fn record_performance(&mut self, model_name: &str, agent_name: &str, metrics: PerformanceMetrics) {
        let key = format!("{}:{}", model_name, agent_name);
        
        self.performance_history
            .entry(key)
            .or_insert_with(ModelPerformanceHistory::new)
            .add_record(metrics);
    }
}

#[derive(Debug, Clone)]
pub struct ModelPerformanceHistory {
    records: Vec<PerformanceRecord>,
    max_records: usize,
}

impl ModelPerformanceHistory {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            max_records: 100,
        }
    }
    
    pub fn add_record(&mut self, metrics: PerformanceMetrics) {
        let record = PerformanceRecord {
            timestamp: Instant::now(),
            metrics,
        };
        
        self.records.push(record);
        
        // Keep only the most recent records
        if self.records.len() > self.max_records {
            self.records.remove(0);
        }
    }
    
    pub fn average_score(&self) -> f64 {
        if self.records.is_empty() {
            return 0.7; // Default
        }
        
        let sum: f64 = self.records.iter()
            .map(|record| record.metrics.overall_score)
            .sum();
        
        sum / self.records.len() as f64
    }
    
    pub fn quality_trend(&self) -> f64 {
        if self.records.len() < 2 {
            return 0.0;
        }
        
        let recent_count = (self.records.len() / 2).max(5);
        let recent_avg: f64 = self.records.iter()
            .rev()
            .take(recent_count)
            .map(|r| r.metrics.quality_score)
            .sum::<f64>() / recent_count as f64;
        
        let historical_avg: f64 = self.records.iter()
            .take(self.records.len() - recent_count)
            .map(|r| r.metrics.quality_score)
            .sum::<f64>() / (self.records.len() - recent_count) as f64;
        
        recent_avg - historical_avg
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub timestamp: Instant,
    pub metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time_ms: u64,
    pub tokens_used: u32,
    pub quality_score: f64,
    pub overall_score: f64,
    pub error_occurred: bool,
}

#[derive(Debug, Clone)]
pub struct CostModel {
    pub cost_per_token: f64,
    pub base_cost: f64,
    pub latency_penalty: f64,
}

/// Cost optimizer for model selection
pub struct CostOptimizer {
    cost_models: HashMap<String, CostModel>,
}

impl CostOptimizer {
    pub fn new() -> Self {
        Self {
            cost_models: HashMap::new(),
        }
    }
    
    pub fn calculate_cost_efficiency_score(&self, model: &ModelInfo, task: &crate::orchestrator::Task) -> f64 {
        let estimated_tokens = CostOptimizer::estimate_token_usage_for_task(task);
        let cost = (estimated_tokens as f64) * model.cost_per_token;
        
        // Higher score for lower cost (inverse relationship)
        let max_cost = 0.10; // $0.10 maximum expected cost
        (max_cost - cost).max(0.0) / max_cost
    }
    
    pub async fn minimize_total_cost(&self, assignments: &mut [ModelAssignment]) -> Result<()> {
        // Look for cost optimization opportunities
        for assignment in assignments.iter_mut() {
            if let Some(cheaper_alternative) = self.find_cheaper_alternative(&assignment.model_selection.model_name) {
                assignment.model_selection.model_name = cheaper_alternative;
                assignment.model_selection.selection_reasoning = 
                    "Selected cheaper model for cost optimization".to_string();
            }
        }
        
        Ok(())
    }
    
    pub fn estimate_task_cost(&self, task_id: &str, model: &ModelInfo) -> f64 {
        // Simplified cost estimation
        let base_tokens = 1000;
        let complexity_multiplier = if task_id.contains("complex") { 3.0 } else { 1.0 };
        
        (base_tokens as f64) * complexity_multiplier * model.cost_per_token
    }
    
    fn estimate_token_usage(&self, assignment: &AgentAssignment) -> u32 {
        // Simplified token estimation based on task complexity
        match assignment.task.estimated_complexity {
            crate::orchestrator::TaskComplexity::Simple => 500,
            crate::orchestrator::TaskComplexity::Moderate => 1500,
            crate::orchestrator::TaskComplexity::Complex => 3000,
            crate::orchestrator::TaskComplexity::Intensive => 5000,
        }
    }
    
    fn find_cheaper_alternative(&self, current_model: &str) -> Option<String> {
        match current_model {
            "codellama:13b" => Some("codellama:7b".to_string()),
            "llama3.2:8b" => Some("llama3.2:3b".to_string()),
            _ => None,
        }
    }
    
    pub fn estimate_token_usage_for_task(task: &crate::orchestrator::Task) -> u32 {
        // Rough estimation based on task type and complexity
        let base_tokens = match task.task_type {
            crate::orchestrator::TaskType::Analysis => 1000,
            crate::orchestrator::TaskType::Generation => 2000,
            crate::orchestrator::TaskType::Query => 500,
            crate::orchestrator::TaskType::Synthesis => 1500,
        };

        let complexity_multiplier = match task.estimated_complexity {
            crate::orchestrator::TaskComplexity::Simple => 0.5,
            crate::orchestrator::TaskComplexity::Moderate => 1.0,
            crate::orchestrator::TaskComplexity::Complex => 2.0,
            crate::orchestrator::TaskComplexity::Intensive => 3.0,
        };

        (base_tokens as f64 * complexity_multiplier) as u32
    }
}

/// Load balancer for distributing work across models
pub struct LoadBalancer {
    current_loads: HashMap<String, f64>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            current_loads: HashMap::new(),
        }
    }
    
    pub async fn get_availability_score(&self, model_name: &str) -> Result<f64> {
        let load = self.current_loads.get(model_name).unwrap_or(&0.0);
        Ok(1.0 - load) // Higher score for lower load
    }
    
    pub async fn balance_assignments(&self, assignments: &mut [ModelAssignment]) -> Result<()> {
        // Simple load balancing: prefer models with lower current load
        for assignment in assignments.iter_mut() {
            if let Some(less_loaded_model) = self.find_less_loaded_alternative(&assignment.model_selection.model_name) {
                assignment.model_selection.model_name = less_loaded_model;
                assignment.model_selection.selection_reasoning = 
                    "Selected less loaded model for load balancing".to_string();
            }
        }
        
        Ok(())
    }
    
    fn find_less_loaded_alternative(&self, current_model: &str) -> Option<String> {
        // In practice, this would check actual load metrics
        None // Placeholder
    }
    
    pub fn update_load(&mut self, model_name: &str, load: f64) {
        self.current_loads.insert(model_name.to_string(), load.clamp(0.0, 1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::AgentCapability;
    use crate::orchestrator::{Task, TaskComplexity};
    
    #[test]
    fn test_model_registry() {
        let registry = ModelRegistry::new();
        
        let code_models = registry.get_capable_models(&vec![
            ModelCapability::CodeGeneration,
            ModelCapability::CodeUnderstanding,
        ]);
        
        assert!(!code_models.is_empty());
        assert!(code_models.iter().any(|m| m.name == "codellama:13b"));
    }
    
    #[test]
    fn test_capability_matching() {
        let router = ModelRouter::new();
        let model = ModelInfo {
            name: "test-model".to_string(),
            capabilities: vec![
                ModelCapability::CodeUnderstanding,
                ModelCapability::Documentation,
            ],
            average_latency_ms: 1000,
            tokens_per_second: 50.0,
            cost_per_token: 0.0001,
            quality_metrics: HashMap::new(),
            max_context_length: 8192,
            available: true,
            current_load: 0.3,
        };
        
        let assignment = AgentAssignment {
            task_id: "test".to_string(),
            task: Task {
                id: "test".to_string(),
                name: "Test Task".to_string(),
                description: "Test".to_string(),
                required_capabilities: vec![AgentCapability::CodeAnalysis],
                input_requirements: vec![],
                output_schema: serde_json::json!({}),
                dependencies: vec![],
                priority: crate::agents::TaskPriority::Normal,
                estimated_complexity: TaskComplexity::Simple,
                timeout_seconds: Some(60),
            },
            agent_name: "test_agent".to_string(),
            priority: crate::agents::TaskPriority::Normal,
            dependencies: vec![],
        };
        
        let score = router.calculate_capability_match_score(&model, &assignment);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }
    
    #[test]
    fn test_performance_history() {
        let mut history = ModelPerformanceHistory::new();
        
        let metrics = PerformanceMetrics {
            execution_time_ms: 1000,
            tokens_used: 500,
            quality_score: 0.8,
            overall_score: 0.85,
            error_occurred: false,
        };
        
        history.add_record(metrics);
        assert_eq!(history.average_score(), 0.85);
    }
}
