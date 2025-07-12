// src/agents/coordinator.rs
//! Enhanced AI Agent Coordination and Routing System
//! 
//! This module provides sophisticated agent coordination capabilities including:
//! - Intelligent task routing based on agent capabilities and current load
//! - Dynamic agent orchestration with parallel execution support
//! - Agent health monitoring and failover handling
//! - Request priority and queue management
//! - Inter-agent communication and data sharing

use anyhow::{Result, Context as AnyhowContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore, mpsc, Mutex};
use tokio::time::{Duration, Instant, sleep};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::agents::{Agent, AgentCapability, FlowContext};

/// Priority levels for agent requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Agent execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Running { task_id: Uuid, started_at: DateTime<Utc> },
    Failed { error: String, failed_at: DateTime<Utc> },
    Maintenance,
}

/// Agent health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    pub agent_name: String,
    pub status: AgentStatus,
    pub success_rate: f32,
    pub average_response_time: Duration,
    pub current_load: usize,
    pub max_concurrent: usize,
    pub last_health_check: DateTime<Utc>,
}

/// Task execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub id: Uuid,
    pub session_id: String,
    pub priority: Priority,
    pub required_capabilities: Vec<AgentCapability>,
    pub preferred_agents: Vec<String>,
    pub context: FlowContext,
    pub timeout: Option<Duration>,
    pub retry_count: usize,
    pub created_at: DateTime<Utc>,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: Uuid,
    pub agent_name: String,
    pub status: TaskStatus,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub execution_time: Duration,
    pub completed_at: DateTime<Utc>,
}

/// Task execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
}

/// Agent coordination and routing system
pub struct AgentCoordinator {
    agents: Arc<RwLock<HashMap<String, Arc<dyn Agent>>>>,
    agent_health: Arc<RwLock<HashMap<String, AgentHealth>>>,
    task_queue: Arc<Mutex<VecDeque<TaskRequest>>>,
    running_tasks: Arc<RwLock<HashMap<Uuid, TaskExecution>>>,
    completed_tasks: Arc<RwLock<HashMap<Uuid, TaskResult>>>,
    execution_semaphore: Arc<Semaphore>,
    health_check_interval: Duration,
    max_retry_attempts: usize,
    task_timeout_default: Duration,
}

/// Active task execution context
#[derive(Debug)]
struct TaskExecution {
    task: TaskRequest,
    agent_name: String,
    started_at: Instant,
    cancellation_token: tokio_util::sync::CancellationToken,
}

impl AgentCoordinator {
    /// Create a new agent coordinator
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            agents: Arc::new(RwLock::new(HashMap::new())),
            agent_health: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            execution_semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            health_check_interval: Duration::from_secs(30),
            max_retry_attempts: 3,
            task_timeout_default: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Register an agent with the coordinator
    pub async fn register_agent(&self, agent: Arc<dyn Agent>) -> Result<()> {
        let agent_name = agent.name().to_string();
        
        // Add to agent registry
        {
            let mut agents = self.agents.write().await;
            agents.insert(agent_name.clone(), agent);
        }

        // Initialize health tracking
        {
            let mut health = self.agent_health.write().await;
            health.insert(agent_name.clone(), AgentHealth {
                agent_name: agent_name.clone(),
                status: AgentStatus::Idle,
                success_rate: 1.0,
                average_response_time: Duration::from_millis(100),
                current_load: 0,
                max_concurrent: 5, // Default, can be configured per agent
                last_health_check: Utc::now(),
            });
        }

        tracing::info!("Registered agent: {}", agent_name);
        Ok(())
    }

    /// Submit a task for execution
    pub async fn submit_task(&self, mut task: TaskRequest) -> Result<Uuid> {
        // Assign unique ID if not provided
        if task.id.is_nil() {
            task.id = Uuid::new_v4();
        }

        // Set default timeout if not specified
        if task.timeout.is_none() {
            task.timeout = Some(self.task_timeout_default);
        }

        tracing::info!("Submitting task {} with priority {:?}", task.id, task.priority);

        // Add to priority queue
        {
            let mut queue = self.task_queue.lock().await;
            
            // Insert based on priority (higher priority first)
            let insert_position = queue.iter().position(|t| t.priority < task.priority)
                .unwrap_or(queue.len());
            
            queue.insert(insert_position, task.clone());
        }

        // Start task processing (if not already running)
        self.process_queue();

        Ok(task.id)
    }

    /// Process the task queue
    fn process_queue(&self) {
        // Just start the queue processing without recursive calls
        let coordinator = self.clone();
        tokio::spawn(async move {
            coordinator.run_queue_loop().await;
        });
    }

    /// Main queue processing loop
    async fn run_queue_loop(&self) {
        loop {
            // Try to acquire execution permit
            let permit = match self.execution_semaphore.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    // No capacity available, wait a bit
                    sleep(Duration::from_millis(100)).await;
                    continue;
                }
            };

            // Get next task from queue
            let task = {
                let mut queue = self.task_queue.lock().await;
                queue.pop_front()
            };

            let task = match task {
                Some(task) => task,
                None => {
                    // No tasks available, release permit and wait
                    drop(permit);
                    sleep(Duration::from_millis(500)).await;
                    continue;
                }
            };

            // Find suitable agent for the task
            let agent_name = match self.select_agent(&task).await {
                Ok(name) => name,
                Err(e) => {
                    tracing::error!("Failed to select agent for task {}: {}", task.id, e);
                    self.mark_task_failed(task.id, format!("Agent selection failed: {}", e)).await;
                    drop(permit);
                    continue;
                }
            };

            // Execute the task
            let coordinator = self.clone();
            let task_clone = task.clone();
            tokio::spawn(async move {
                coordinator.execute_task(task_clone, agent_name, permit).await;
            });
        }
    }

    /// Select the best agent for a task
    async fn select_agent(&self, task: &TaskRequest) -> Result<String> {
        let agents = self.agents.read().await;
        let health = self.agent_health.read().await;

        let mut suitable_agents = Vec::new();

        // Find agents that can handle the task
        for (agent_name, agent) in agents.iter() {
            // Check if agent has required capabilities
            let agent_capabilities = agent.capabilities();
            let can_handle_capabilities = task.required_capabilities.iter()
                .all(|req_cap| agent_capabilities.contains(req_cap));

            if !can_handle_capabilities {
                continue;
            }

            // Check if agent can handle the specific task context
            if !agent.can_handle(&task.context).await {
                continue;
            }

            // Check agent health and availability
            if let Some(agent_health) = health.get(agent_name) {
                match &agent_health.status {
                    AgentStatus::Idle => {
                        suitable_agents.push((agent_name.clone(), agent_health.clone()));
                    }
                    AgentStatus::Running { .. } => {
                        // Check if agent can handle concurrent tasks
                        if agent_health.current_load < agent_health.max_concurrent {
                            suitable_agents.push((agent_name.clone(), agent_health.clone()));
                        }
                    }
                    _ => {
                        // Agent not available
                        continue;
                    }
                }
            }
        }

        if suitable_agents.is_empty() {
            return Err(anyhow::anyhow!("No suitable agents available for task"));
        }

        // Prefer agents specified in the task
        if !task.preferred_agents.is_empty() {
            for preferred in &task.preferred_agents {
                if let Some((agent_name, _)) = suitable_agents.iter()
                    .find(|(name, _)| name == preferred) {
                    return Ok(agent_name.clone());
                }
            }
        }

        // Select best agent based on load balancing and performance
        suitable_agents.sort_by(|a, b| {
            // Primary: current load (lower is better)
            let load_cmp = a.1.current_load.cmp(&b.1.current_load);
            if load_cmp != std::cmp::Ordering::Equal {
                return load_cmp;
            }

            // Secondary: success rate (higher is better)
            b.1.success_rate.partial_cmp(&a.1.success_rate).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(suitable_agents[0].0.clone())
    }

    /// Execute a task with the selected agent
    async fn execute_task(
        &self,
        task: TaskRequest,
        agent_name: String,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let task_id = task.id;
        let start_time = Instant::now();

        tracing::info!("Executing task {} with agent {}", task_id, agent_name);

        // Update agent status
        self.update_agent_status(&agent_name, AgentStatus::Running {
            task_id,
            started_at: Utc::now(),
        }).await;

        // Create cancellation token for timeout handling
        let cancellation_token = tokio_util::sync::CancellationToken::new();

        // Store running task
        {
            let mut running = self.running_tasks.write().await;
            running.insert(task_id, TaskExecution {
                task: task.clone(),
                agent_name: agent_name.clone(),
                started_at: start_time,
                cancellation_token: cancellation_token.clone(),
            });
        }

        // Execute with timeout
        let result = if let Some(timeout) = task.timeout {
            tokio::select! {
                result = self.execute_task_inner(&task, &agent_name) => result,
                _ = sleep(timeout) => Err(anyhow::anyhow!("Task timeout")),
                _ = cancellation_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
            }
        } else {
            tokio::select! {
                result = self.execute_task_inner(&task, &agent_name) => result,
                _ = cancellation_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
            }
        };

        let execution_time = start_time.elapsed();

        // Process result
        let task_result = match result {
            Ok(result_value) => {
                tracing::info!("Task {} completed successfully in {:?}", task_id, execution_time);
                self.update_agent_success(&agent_name, execution_time).await;
                
                TaskResult {
                    task_id,
                    agent_name: agent_name.clone(),
                    status: TaskStatus::Completed,
                    result: Some(result_value),
                    error: None,
                    execution_time,
                    completed_at: Utc::now(),
                }
            }
            Err(e) => {
                tracing::error!("Task {} failed: {}", task_id, e);
                self.update_agent_failure(&agent_name, execution_time).await;

                // Check if task should be retried
                if task.retry_count < self.max_retry_attempts {
                    let mut retry_task = task.clone();
                    retry_task.retry_count += 1;
                    retry_task.id = Uuid::new_v4(); // New ID for retry
                    
                    tracing::info!("Retrying task {} as {} (attempt {})", 
                        task_id, retry_task.id, retry_task.retry_count + 1);
                    
                    // Re-queue the task
                    let _ = self.submit_task(retry_task).await;
                }

                TaskResult {
                    task_id,
                    agent_name: agent_name.clone(),
                    status: TaskStatus::Failed,
                    result: None,
                    error: Some(e.to_string()),
                    execution_time,
                    completed_at: Utc::now(),
                }
            }
        };

        // Store completed task and clean up
        self.complete_task(task_result).await;
    }

    /// Execute the actual task
    async fn execute_task_inner(
        &self,
        task: &TaskRequest,
        agent_name: &str,
    ) -> Result<serde_json::Value> {
        let agents = self.agents.read().await;
        let agent = agents.get(agent_name)
            .ok_or_else(|| anyhow::anyhow!("Agent {} not found", agent_name))?;

        let mut context = task.context.clone();
        context.task_id = Some(task.id);
        
        agent.execute(&mut context).await
    }

    /// Update agent status
    async fn update_agent_status(&self, agent_name: &str, status: AgentStatus) {
        let mut health = self.agent_health.write().await;
        if let Some(agent_health) = health.get_mut(agent_name) {
            let old_load = match &agent_health.status {
                AgentStatus::Running { .. } => agent_health.current_load,
                _ => 0,
            };

            let new_load = match &status {
                AgentStatus::Running { .. } => old_load + 1,
                AgentStatus::Idle => old_load.saturating_sub(1),
                _ => old_load,
            };

            agent_health.status = status;
            agent_health.current_load = new_load;
            agent_health.last_health_check = Utc::now();
        }
    }

    /// Update agent success metrics
    async fn update_agent_success(&self, agent_name: &str, execution_time: Duration) {
        let mut health = self.agent_health.write().await;
        if let Some(agent_health) = health.get_mut(agent_name) {
            // Update success rate (exponential moving average)
            agent_health.success_rate = agent_health.success_rate * 0.9 + 0.1;
            
            // Update average response time (exponential moving average)
            agent_health.average_response_time = Duration::from_millis(
                (agent_health.average_response_time.as_millis() as f64 * 0.9 + 
                 execution_time.as_millis() as f64 * 0.1) as u64
            );

            agent_health.status = AgentStatus::Idle;
            agent_health.current_load = agent_health.current_load.saturating_sub(1);
        }
    }

    /// Update agent failure metrics
    async fn update_agent_failure(&self, agent_name: &str, execution_time: Duration) {
        let mut health = self.agent_health.write().await;
        if let Some(agent_health) = health.get_mut(agent_name) {
            // Update success rate (exponential moving average)
            agent_health.success_rate = agent_health.success_rate * 0.9;
            
            // Update average response time
            agent_health.average_response_time = Duration::from_millis(
                (agent_health.average_response_time.as_millis() as f64 * 0.9 + 
                 execution_time.as_millis() as f64 * 0.1) as u64
            );

            agent_health.status = AgentStatus::Idle;
            agent_health.current_load = agent_health.current_load.saturating_sub(1);
        }
    }

    /// Mark a task as failed
    async fn mark_task_failed(&self, task_id: Uuid, error: String) {
        let task_result = TaskResult {
            task_id,
            agent_name: "unknown".to_string(),
            status: TaskStatus::Failed,
            result: None,
            error: Some(error),
            execution_time: Duration::from_millis(0),
            completed_at: Utc::now(),
        };

        self.complete_task(task_result).await;
    }

    /// Complete a task and clean up
    async fn complete_task(&self, result: TaskResult) {
        let task_id = result.task_id;

        // Remove from running tasks
        {
            let mut running = self.running_tasks.write().await;
            running.remove(&task_id);
        }

        // Store in completed tasks (with cleanup for old tasks)
        {
            let mut completed = self.completed_tasks.write().await;
            completed.insert(task_id, result);

            // Clean up old completed tasks (keep last 1000)
            if completed.len() > 1000 {
                let oldest_keys: Vec<_> = completed.keys().take(completed.len() - 1000).cloned().collect();
                for key in oldest_keys {
                    completed.remove(&key);
                }
            }
        }
    }

    /// Get task result
    pub async fn get_task_result(&self, task_id: Uuid) -> Option<TaskResult> {
        let completed = self.completed_tasks.read().await;
        completed.get(&task_id).cloned()
    }

    /// Cancel a running task
    pub async fn cancel_task(&self, task_id: Uuid) -> Result<()> {
        let running = self.running_tasks.read().await;
        if let Some(execution) = running.get(&task_id) {
            execution.cancellation_token.cancel();
            tracing::info!("Cancelled task {}", task_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Task {} not found or not running", task_id))
        }
    }

    /// Get agent health status
    pub async fn get_agent_health(&self, agent_name: &str) -> Option<AgentHealth> {
        let health = self.agent_health.read().await;
        health.get(agent_name).cloned()
    }

    /// Get all agent health statuses
    pub async fn get_all_agent_health(&self) -> HashMap<String, AgentHealth> {
        let health = self.agent_health.read().await;
        health.clone()
    }

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        let queue = self.task_queue.lock().await;
        let running = self.running_tasks.read().await;
        let completed = self.completed_tasks.read().await;

        let mut priority_counts = HashMap::new();
        for task in queue.iter() {
            *priority_counts.entry(task.priority).or_insert(0) += 1;
        }

        QueueStats {
            pending_tasks: queue.len(),
            running_tasks: running.len(),
            completed_tasks: completed.len(),
            priority_distribution: priority_counts,
            available_capacity: self.execution_semaphore.available_permits(),
        }
    }

    /// Start periodic health checks
    pub async fn start_health_monitor(&self) {
        let coordinator = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(coordinator.health_check_interval);
            loop {
                interval.tick().await;
                coordinator.perform_health_checks().await;
            }
        });
    }

    /// Perform health checks on all agents
    async fn perform_health_checks(&self) {
        // This could include more sophisticated health checks
        // For now, we'll just update the last health check time
        let mut health = self.agent_health.write().await;
        for agent_health in health.values_mut() {
            agent_health.last_health_check = Utc::now();
        }
    }
}

impl Clone for AgentCoordinator {
    fn clone(&self) -> Self {
        Self {
            agents: self.agents.clone(),
            agent_health: self.agent_health.clone(),
            task_queue: self.task_queue.clone(),
            running_tasks: self.running_tasks.clone(),
            completed_tasks: self.completed_tasks.clone(),
            execution_semaphore: self.execution_semaphore.clone(),
            health_check_interval: self.health_check_interval,
            max_retry_attempts: self.max_retry_attempts,
            task_timeout_default: self.task_timeout_default,
        }
    }
}

/// Queue statistics for monitoring
#[derive(Debug, Serialize, Deserialize)]
pub struct QueueStats {
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub completed_tasks: usize,
    pub priority_distribution: HashMap<Priority, usize>,
    pub available_capacity: usize,
}

/// Coordinator Agent implementation for integration with the system
pub struct CoordinatorAgent {
    coordinator: Arc<AgentCoordinator>,
}

impl CoordinatorAgent {
    pub fn new(coordinator: Arc<AgentCoordinator>) -> Self {
        Self { coordinator }
    }
}

#[async_trait]
impl Agent for CoordinatorAgent {
    fn name(&self) -> &'static str {
        "agent_coordinator"
    }

    fn description(&self) -> &'static str {
        "Coordinates and routes tasks between agents with load balancing and health monitoring"
    }

    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![
            AgentCapability::TaskCoordination,
            AgentCapability::LoadBalancing,
            AgentCapability::HealthMonitoring,
        ]
    }

    async fn execute(&self, context: &mut FlowContext) -> Result<serde_json::Value> {
        let operation = context.metadata.get("coordinator_operation")
            .and_then(|v| v.as_str())
            .unwrap_or("get_stats");

        match operation {
            "get_stats" => {
                let stats = self.coordinator.get_queue_stats().await;
                Ok(serde_json::to_value(stats)?)
            }
            "get_agent_health" => {
                let health = self.coordinator.get_all_agent_health().await;
                Ok(serde_json::to_value(health)?)
            }
            "cancel_task" => {
                if let Some(task_id_str) = context.metadata.get("task_id").and_then(|v| v.as_str()) {
                    if let Ok(task_id) = Uuid::parse_str(task_id_str) {
                        self.coordinator.cancel_task(task_id).await?;
                        Ok(serde_json::json!({"status": "cancelled", "task_id": task_id}))
                    } else {
                        Err(anyhow::anyhow!("Invalid task ID format"))
                    }
                } else {
                    Err(anyhow::anyhow!("Task ID not provided"))
                }
            }
            _ => {
                Err(anyhow::anyhow!("Unknown coordinator operation: {}", operation))
            }
        }
    }

    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.capabilities_required.contains(&AgentCapability::TaskCoordination) ||
        context.metadata.contains_key("coordinator_operation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = AgentCoordinator::new(10);
        assert_eq!(coordinator.execution_semaphore.available_permits(), 10);
    }

    #[tokio::test]
    async fn test_task_priority_queue() {
        // Test priority-based task queuing
    }

    #[tokio::test]
    async fn test_agent_selection() {
        // Test agent selection algorithm
    }
}
