// src/model_discovery/scheduler.rs
//! Scheduler for automated model discovery and updates

use super::{ModelDiscoveryService, ModelDatabase};
use anyhow::Result;
use chrono::{DateTime, Timelike, Utc};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Configuration for the model discovery scheduler with evaluation capabilities
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// How often to run full model discovery (in hours)
    pub discovery_interval_hours: u64,
    /// How often to check model availability (in minutes)
    pub availability_check_interval_minutes: u64,
    /// How often to measure model performance (in hours)
    pub performance_check_interval_hours: u64,
    /// How often to run comprehensive evaluations (in hours)
    pub evaluation_interval_hours: u64,
    /// How many days of historical data to keep
    pub history_retention_days: u32,
    /// Whether to run discovery on startup
    pub run_on_startup: bool,
    /// Whether to run comprehensive evaluation on startup
    pub run_evaluation_on_startup: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            discovery_interval_hours: 24,          // Daily discovery
            availability_check_interval_minutes: 30, // Check availability every 30 minutes
            performance_check_interval_hours: 6,    // Performance check every 6 hours
            evaluation_interval_hours: 48,          // Comprehensive evaluation every 2 days
            history_retention_days: 30,             // Keep 30 days of history
            run_on_startup: true,
            run_evaluation_on_startup: false,       // Evaluation takes longer, skip on startup by default
        }
    }
}

/// Automated model discovery scheduler
pub struct ModelDiscoveryScheduler {
    config: SchedulerConfig,
    discovery_service: Arc<RwLock<ModelDiscoveryService>>,
    database: Arc<ModelDatabase>,
    // Add model client for evaluation
    model_client: Option<Arc<dyn crate::model_clients::ModelClient>>,
    last_discovery: Option<DateTime<Utc>>,
    last_availability_check: Option<DateTime<Utc>>,
    last_performance_check: Option<DateTime<Utc>>,
    last_evaluation: Option<DateTime<Utc>>,
    last_cleanup: Option<DateTime<Utc>>,
}

impl ModelDiscoveryScheduler {
    pub fn new(
        config: SchedulerConfig,
        discovery_service: Arc<RwLock<ModelDiscoveryService>>,
        database: Arc<ModelDatabase>,
    ) -> Self {
        Self {
            config,
            discovery_service,
            database,
            model_client: None,
            last_discovery: None,
            last_availability_check: None,
            last_performance_check: None,
            last_evaluation: None,
            last_cleanup: None,
        }
    }
    
    /// Set a model client for comprehensive evaluation
    pub fn with_model_client(mut self, client: Arc<dyn crate::model_clients::ModelClient>) -> Self {
        self.model_client = Some(client);
        self
    }
    
    /// Start the scheduler with all background tasks
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting model discovery scheduler with config: {:?}", self.config);
        
        // Load existing models from database on startup
        {
            let mut service = self.discovery_service.write().await;
            service.load_from_database().await?;
        }
        
        // Run initial discovery if configured
        if self.config.run_on_startup {
            info!("Running initial model discovery on startup");
            if let Err(e) = self.run_model_discovery().await {
                error!("Failed to run initial model discovery: {}", e);
            }
        }
        
        // Spawn background tasks
        self.spawn_discovery_task().await;
        self.spawn_availability_check_task().await;
        self.spawn_performance_check_task().await;
        self.spawn_evaluation_task().await;  // Add comprehensive evaluation task
        self.spawn_cleanup_task().await;
        
        // Run initial comprehensive evaluation if configured and client is available
        if self.config.run_evaluation_on_startup && self.model_client.is_some() {
            info!("Running initial comprehensive evaluation on startup");
            if let Err(e) = self.run_comprehensive_evaluation().await {
                error!("Failed to run initial comprehensive evaluation: {}", e);
            }
        }
        
        info!("Model discovery scheduler started successfully");
        Ok(())
    }
    
    /// Spawn the main model discovery task
    async fn spawn_discovery_task(&self) {
        let discovery_service = Arc::clone(&self.discovery_service);
        let database = Arc::clone(&self.database);
        let interval_hours = self.config.discovery_interval_hours;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_hours * 3600));
            
            loop {
                interval.tick().await;
                
                info!("Starting scheduled model discovery");
                match Self::run_discovery_task(&discovery_service, &database).await {
                    Ok(model_count) => {
                        info!("Scheduled model discovery completed successfully, found {} models", model_count);
                    }
                    Err(e) => {
                        error!("Scheduled model discovery failed: {}", e);
                    }
                }
            }
        });
    }
    
    /// Spawn the availability check task
    async fn spawn_availability_check_task(&self) {
        let discovery_service = Arc::clone(&self.discovery_service);
        let database = Arc::clone(&self.database);
        let interval_minutes = self.config.availability_check_interval_minutes;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_minutes * 60));
            
            loop {
                interval.tick().await;
                
                debug!("Starting scheduled availability check");
                match Self::run_availability_check(&discovery_service, &database).await {
                    Ok(checked_count) => {
                        debug!("Checked availability for {} models", checked_count);
                    }
                    Err(e) => {
                        warn!("Availability check failed: {}", e);
                    }
                }
            }
        });
    }
    
    /// Spawn the performance check task
    async fn spawn_performance_check_task(&self) {
        let discovery_service = Arc::clone(&self.discovery_service);
        let database = Arc::clone(&self.database);
        let interval_hours = self.config.performance_check_interval_hours;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_hours * 3600));
            
            loop {
                interval.tick().await;
                
                debug!("Starting scheduled performance check");
                match Self::run_performance_check(&discovery_service, &database).await {
                    Ok(checked_count) => {
                        debug!("Checked performance for {} models", checked_count);
                    }
                    Err(e) => {
                        warn!("Performance check failed: {}", e);
                    }
                }
            }
        });
    }
    
    /// Spawn the comprehensive evaluation task
    async fn spawn_evaluation_task(&self) {
        let discovery_service = Arc::clone(&self.discovery_service);
        let database = Arc::clone(&self.database);
        let model_client = self.model_client.clone();
        let interval_hours = self.config.evaluation_interval_hours;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(interval_hours * 3600));
            
            loop {
                interval.tick().await;
                
                if let Some(ref client) = model_client {
                    info!("Starting scheduled comprehensive evaluation");
                    match Self::run_comprehensive_evaluation_task(&discovery_service, &database, client.clone()).await {
                        Ok(evaluation_count) => {
                            info!("Scheduled comprehensive evaluation completed successfully, evaluated {} models", evaluation_count);
                        }
                        Err(e) => {
                            error!("Scheduled comprehensive evaluation failed: {}", e);
                        }
                    }
                } else {
                    debug!("Skipping comprehensive evaluation - no model client configured");
                }
            }
        });
    }
    
    /// Spawn the cleanup task
    async fn spawn_cleanup_task(&self) {
        let database = Arc::clone(&self.database);
        let retention_days = self.config.history_retention_days;
        
        tokio::spawn(async move {
            // Run cleanup daily at 2 AM UTC
            let mut interval = interval(Duration::from_secs(24 * 3600));
            
            loop {
                interval.tick().await;
                
                let now = Utc::now();
                if now.hour() == 2 {  // Run at 2 AM UTC
                    info!("Starting scheduled database cleanup");
                    match database.cleanup_old_data(retention_days).await {
                        Ok(_) => {
                            info!("Database cleanup completed successfully");
                        }
                        Err(e) => {
                            error!("Database cleanup failed: {}", e);
                        }
                    }
                }
            }
        });
    }
    
    /// Run model discovery manually
    pub async fn run_model_discovery(&mut self) -> Result<usize> {
        self.last_discovery = Some(Utc::now());
        Self::run_discovery_task(&self.discovery_service, &self.database).await
    }
    
    /// Run comprehensive evaluation manually
    pub async fn run_comprehensive_evaluation(&mut self) -> Result<usize> {
        if let Some(ref client) = self.model_client {
            self.last_evaluation = Some(Utc::now());
            Self::run_comprehensive_evaluation_task(&self.discovery_service, &self.database, client.clone()).await
        } else {
            warn!("Cannot run comprehensive evaluation - no model client configured");
            Ok(0)
        }
    }
    
    /// Internal discovery task implementation
    async fn run_discovery_task(
        discovery_service: &Arc<RwLock<ModelDiscoveryService>>,
        _database: &Arc<ModelDatabase>,
    ) -> Result<usize> {
        let mut service = discovery_service.write().await;
        let models = service.discover_all_models().await?;
        
        info!("Model discovery found {} models across all providers", models.len());
        
        // Log summary by provider
        let mut provider_counts = std::collections::HashMap::new();
        for model in &models {
            *provider_counts.entry(&model.provider).or_insert(0) += 1;
        }
        
        for (provider, count) in provider_counts {
            info!("  {:?}: {} models", provider, count);
        }
        
        Ok(models.len())
    }
    
    /// Internal availability check implementation
    async fn run_availability_check(
        discovery_service: &Arc<RwLock<ModelDiscoveryService>>,
        _database: &Arc<ModelDatabase>,
    ) -> Result<usize> {
        let service = discovery_service.read().await;
        let models = service.get_evaluated_models().await?;
        let mut checked_count = 0;
        
        for (model, _score) in models {
            // This would typically use the providers to check availability
            // For now, we'll simulate it
            debug!("Checking availability for model: {}", model.id);
            checked_count += 1;
            
            // In a real implementation, you'd call provider.check_model_availability()
            // and update the service with the results
        }
        
        Ok(checked_count)
    }
    
    /// Internal performance check implementation
    async fn run_performance_check(
        discovery_service: &Arc<RwLock<ModelDiscoveryService>>,
        database: &Arc<ModelDatabase>,
    ) -> Result<usize> {
        let service = discovery_service.read().await;
        let models = service.get_evaluated_models().await?;
        let mut checked_count = 0;
        
        for (model, _score) in models {
            // Skip performance checks for unavailable models
            if !model.availability.is_available {
                continue;
            }
            
            debug!("Checking performance for model: {}", model.id);
            checked_count += 1;
            
            // In a real implementation, you'd run performance benchmarks
            // and update the service with the results
        }
        
        Ok(checked_count)
    }
    
    /// Internal comprehensive evaluation task implementation
    async fn run_comprehensive_evaluation_task(
        discovery_service: &Arc<RwLock<ModelDiscoveryService>>,
        _database: &Arc<ModelDatabase>,
        model_client: Arc<dyn crate::model_clients::ModelClient>,
    ) -> Result<usize> {
        info!("Starting comprehensive model evaluation");
        
        let mut service = discovery_service.write().await;
        let evaluation_results = service.discover_and_evaluate_models(model_client).await?;
        
        let evaluated_count = evaluation_results.iter()
            .filter(|(_, eval)| eval.is_some())
            .count();
        
        info!("Comprehensive evaluation completed:");
        info!("  Total models: {}", evaluation_results.len());
        info!("  Successfully evaluated: {}", evaluated_count);
        info!("  Failed evaluations: {}", evaluation_results.len() - evaluated_count);
        
        // Log evaluation summary
        let summary = service.get_evaluation_summary();
        if summary.evaluated_models > 0 {
            info!("  Average score: {:.1}/100", summary.average_score);
            info!("  Highest score: {:.1}/100 ({})", 
                summary.highest_score, 
                summary.best_model.as_deref().unwrap_or("unknown")
            );
        }
        
        Ok(evaluated_count)
    }
    
    /// Get the last evaluation timestamp
    pub fn last_evaluation(&self) -> Option<DateTime<Utc>> {
        self.last_evaluation
    }
    
    /// Check if comprehensive evaluation is overdue
    pub fn is_evaluation_overdue(&self) -> bool {
        match self.last_evaluation {
            Some(last) => {
                let hours_since = Utc::now().signed_duration_since(last).num_hours();
                hours_since >= self.config.evaluation_interval_hours as i64
            }
            None => true, // Never evaluated
        }
    }
    
    /// Get scheduler status
    pub fn get_status(&self) -> SchedulerStatus {
        SchedulerStatus {
            config: self.config.clone(),
            last_discovery: self.last_discovery,
            last_availability_check: self.last_availability_check,
            last_performance_check: self.last_performance_check,
            last_evaluation: self.last_evaluation,
            last_cleanup: self.last_cleanup,
            next_discovery: self.last_discovery.map(|d| d + chrono::Duration::hours(self.config.discovery_interval_hours as i64)),
            next_availability_check: self.last_availability_check.map(|d| d + chrono::Duration::minutes(self.config.availability_check_interval_minutes as i64)),
            next_performance_check: self.last_performance_check.map(|d| d + chrono::Duration::hours(self.config.performance_check_interval_hours as i64)),
            next_evaluation: self.last_evaluation.map(|d| d + chrono::Duration::hours(self.config.evaluation_interval_hours as i64)),
        }
    }
}

/// Status information for the scheduler
#[derive(Debug, Clone)]
pub struct SchedulerStatus {
    pub config: SchedulerConfig,
    pub last_discovery: Option<DateTime<Utc>>,
    pub last_availability_check: Option<DateTime<Utc>>,
    pub last_performance_check: Option<DateTime<Utc>>,
    pub last_evaluation: Option<DateTime<Utc>>,
    pub last_cleanup: Option<DateTime<Utc>>,
    pub next_discovery: Option<DateTime<Utc>>,
    pub next_availability_check: Option<DateTime<Utc>>,
    pub next_performance_check: Option<DateTime<Utc>>,
    pub next_evaluation: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.discovery_interval_hours, 24);
        assert_eq!(config.availability_check_interval_minutes, 30);
        assert_eq!(config.performance_check_interval_hours, 6);
        assert_eq!(config.evaluation_interval_hours, 48);
        assert_eq!(config.history_retention_days, 30);
        assert!(config.run_on_startup);
        assert!(!config.run_evaluation_on_startup);
    }
    
    #[test]
    fn test_scheduler_status() {
        let config = SchedulerConfig::default();
        let now = Utc::now();
        
        let status = SchedulerStatus {
            config: config.clone(),
            last_discovery: Some(now),
            last_availability_check: Some(now),
            last_performance_check: Some(now),
            last_cleanup: Some(now),
            next_discovery: Some(now + chrono::Duration::hours(24)),
            next_availability_check: Some(now + chrono::Duration::minutes(30)),
            next_performance_check: Some(now + chrono::Duration::hours(6)),
            next_evaluation: Some(now + chrono::Duration::hours(48)),
        };
        
        assert!(status.next_discovery.is_some());
        assert!(status.next_availability_check.is_some());
        assert!(status.next_performance_check.is_some());
        assert!(status.next_evaluation.is_some());
    }
}
