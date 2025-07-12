// src/model_discovery/database.rs
//! Database integration for model discovery persistence

use super::{ModelInfo, ModelProviderType, PerformanceMetrics, QualityScores, BenchmarkResult, ModelAvailability};
use crate::agents::ModelCapability;
use crate::orchestrator::model_router::PerformanceRecord;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde_json;
use sqlx::{Pool, Sqlite, Row};
use std::collections::HashMap;
use tracing::{debug, error, info};

/// Database manager for model discovery data
#[derive(Clone)]
pub struct ModelDatabase {
    pool: Pool<Sqlite>,
}

impl ModelDatabase {
    pub fn new(pool: Pool<Sqlite>) -> Self {
        Self { pool }
    }
    
    /// Initialize the model discovery tables
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing model discovery database tables");
        
        // Create model_info table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS model_info (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                version TEXT,
                model_family TEXT NOT NULL,
                parameter_count INTEGER,
                context_window INTEGER NOT NULL,
                max_output_tokens INTEGER,
                architecture TEXT,
                training_cutoff TEXT,
                capabilities TEXT NOT NULL, -- JSON array
                strengths TEXT NOT NULL, -- JSON array
                weaknesses TEXT NOT NULL, -- JSON array
                ideal_use_cases TEXT NOT NULL, -- JSON array
                supported_formats TEXT NOT NULL, -- JSON array
                performance_metrics TEXT NOT NULL, -- JSON object
                quality_scores TEXT NOT NULL, -- JSON object
                benchmark_results TEXT NOT NULL, -- JSON array
                availability TEXT NOT NULL, -- JSON object
                cost_info TEXT NOT NULL, -- JSON object
                deployment_info TEXT NOT NULL, -- JSON object
                discovered_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                metadata TEXT NOT NULL -- JSON object
            )
        "#)
        .execute(&self.pool)
        .await
        .context("Failed to create model_info table")?;
        
        // Create model_performance_history table for tracking performance over time
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS model_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                measured_at TEXT NOT NULL,
                tokens_per_second REAL NOT NULL,
                latency_ms INTEGER NOT NULL,
                throughput_requests_per_minute REAL NOT NULL,
                memory_usage_gb REAL,
                gpu_utilization_percent REAL,
                cpu_utilization_percent REAL,
                first_token_latency_ms INTEGER,
                FOREIGN KEY (model_id) REFERENCES model_info (id)
            )
        "#)
        .execute(&self.pool)
        .await
        .context("Failed to create model_performance_history table")?;
        
        // Create model_availability_history table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS model_availability_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                is_available BOOLEAN NOT NULL,
                response_time_ms INTEGER,
                uptime_percent REAL,
                FOREIGN KEY (model_id) REFERENCES model_info (id)
            )
        "#)
        .execute(&self.pool)
        .await
        .context("Failed to create model_availability_history table")?;
        
        // Create model_recommendations table for storing recommendation scores
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS model_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                use_case TEXT NOT NULL,
                capabilities TEXT NOT NULL, -- JSON array
                score REAL NOT NULL,
                calculated_at TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model_info (id)
            )
        "#)
        .execute(&self.pool)
        .await
        .context("Failed to create model_recommendations table")?;
        
        info!("Model discovery database tables initialized successfully");
        Ok(())
    }
    
    /// Store or update model information
    pub async fn store_model(&self, model: &ModelInfo) -> Result<()> {
        debug!("Storing model information for: {}", model.id);
        
        let capabilities_json = serde_json::to_string(&model.capabilities)
            .context("Failed to serialize capabilities")?;
        let strengths_json = serde_json::to_string(&model.strengths)
            .context("Failed to serialize strengths")?;
        let weaknesses_json = serde_json::to_string(&model.weaknesses)
            .context("Failed to serialize weaknesses")?;
        let use_cases_json = serde_json::to_string(&model.ideal_use_cases)
            .context("Failed to serialize use cases")?;
        let formats_json = serde_json::to_string(&model.supported_formats)
            .context("Failed to serialize supported formats")?;
        let performance_json = serde_json::to_string(&model.performance_metrics)
            .context("Failed to serialize performance metrics")?;
        let quality_json = serde_json::to_string(&model.quality_scores)
            .context("Failed to serialize quality scores")?;
        let benchmarks_json = serde_json::to_string(&model.benchmark_results)
            .context("Failed to serialize benchmark results")?;
        let availability_json = serde_json::to_string(&model.availability)
            .context("Failed to serialize availability")?;
        let cost_json = serde_json::to_string(&model.cost_info)
            .context("Failed to serialize cost info")?;
        let deployment_json = serde_json::to_string(&model.deployment_info)
            .context("Failed to serialize deployment info")?;
        let metadata_json = serde_json::to_string(&model.metadata)
            .context("Failed to serialize metadata")?;
        
        sqlx::query(r#"
            INSERT OR REPLACE INTO model_info (
                id, name, provider, version, model_family, parameter_count,
                context_window, max_output_tokens, architecture, training_cutoff,
                capabilities, strengths, weaknesses, ideal_use_cases, supported_formats,
                performance_metrics, quality_scores, benchmark_results, availability,
                cost_info, deployment_info, discovered_at, last_updated, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(&model.id)
        .bind(&model.name)
        .bind(serde_json::to_string(&model.provider)?)
        .bind(&model.version)
        .bind(&model.model_family)
        .bind(model.parameter_count.map(|p| p as i64))
        .bind(model.context_window as i64)
        .bind(model.max_output_tokens.map(|t| t as i64))
        .bind(&model.architecture)
        .bind(model.training_cutoff.map(|d| d.to_rfc3339()))
        .bind(capabilities_json)
        .bind(strengths_json)
        .bind(weaknesses_json)
        .bind(use_cases_json)
        .bind(formats_json)
        .bind(performance_json)
        .bind(quality_json)
        .bind(benchmarks_json)
        .bind(availability_json)
        .bind(cost_json)
        .bind(deployment_json)
        .bind(model.discovered_at.to_rfc3339())
        .bind(model.last_updated.to_rfc3339())
        .bind(metadata_json)
        .execute(&self.pool)
        .await
        .context("Failed to insert model into database")?;
        
        // Store performance history
        self.store_performance_history(&model.id, &model.performance_metrics, Utc::now()).await?;
        
        // Store availability history
        self.store_availability_history(&model.id, &model.availability).await?;
        
        debug!("Successfully stored model: {}", model.id);
        Ok(())
    }
    
    /// Retrieve model information by ID
    pub async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        debug!("Retrieving model: {}", model_id);
        
        let row = sqlx::query("SELECT * FROM model_info WHERE id = ?")
            .bind(model_id)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to query model from database")?;
        
        match row {
            Some(row) => {
                let model = self.row_to_model_info(row)?;
                Ok(Some(model))
            }
            None => Ok(None),
        }
    }
    
    /// Get all models from database
    pub async fn get_all_models(&self) -> Result<Vec<ModelInfo>> {
        debug!("Retrieving all models from database");
        
        let rows = sqlx::query("SELECT * FROM model_info ORDER BY last_updated DESC")
            .fetch_all(&self.pool)
            .await
            .context("Failed to query all models from database")?;
        
        let mut models = Vec::new();
        for row in rows {
            match self.row_to_model_info(row) {
                Ok(model) => models.push(model),
                Err(e) => error!("Failed to parse model from database row: {}", e),
            }
        }
        
        info!("Retrieved {} models from database", models.len());
        Ok(models)
    }
    
    /// Get models by capability
    pub async fn get_models_by_capability(&self, capability: &ModelCapability) -> Result<Vec<ModelInfo>> {
        debug!("Retrieving models with capability: {:?}", capability);
        
        let capability_json = serde_json::to_string(capability)?;
        
        // This is a simplified search - in a real implementation, you'd want better JSON querying
        let rows = sqlx::query("SELECT * FROM model_info WHERE capabilities LIKE ?")
            .bind(format!("%{}%", capability_json.trim_matches('"')))
            .fetch_all(&self.pool)
            .await
            .context("Failed to query models by capability")?;
        
        let mut models = Vec::new();
        for row in rows {
            match self.row_to_model_info(row) {
                Ok(model) => {
                    if model.capabilities.contains(capability) {
                        models.push(model);
                    }
                }
                Err(e) => error!("Failed to parse model from database row: {}", e),
            }
        }
        
        debug!("Found {} models with capability {:?}", models.len(), capability);
        Ok(models)
    }
    
    /// Get models by provider
    pub async fn get_models_by_provider(&self, provider: &ModelProviderType) -> Result<Vec<ModelInfo>> {
        debug!("Retrieving models from provider: {:?}", provider);
        
        let provider_json = serde_json::to_string(provider)?;
        
        let rows = sqlx::query("SELECT * FROM model_info WHERE provider = ?")
            .bind(provider_json)
            .fetch_all(&self.pool)
            .await
            .context("Failed to query models by provider")?;
        
        let mut models = Vec::new();
        for row in rows {
            match self.row_to_model_info(row) {
                Ok(model) => models.push(model),
                Err(e) => error!("Failed to parse model from database row: {}", e),
            }
        }
        
        debug!("Found {} models from provider {:?}", models.len(), provider);
        Ok(models)
    }
    
    /// Store performance measurement in history
    pub async fn store_performance_history(
        &self,
        model_id: &str,
        metrics: &PerformanceMetrics,
        measured_at: DateTime<Utc>,
    ) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO model_performance_history (
                model_id, measured_at, tokens_per_second, latency_ms,
                throughput_requests_per_minute, memory_usage_gb,
                gpu_utilization_percent, cpu_utilization_percent,
                first_token_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(model_id)
        .bind(measured_at.to_rfc3339())
        .bind(metrics.tokens_per_second)
        .bind(metrics.latency_ms as i64)
        .bind(metrics.throughput_requests_per_minute)
        .bind(metrics.memory_usage_gb)
        .bind(metrics.gpu_utilization_percent)
        .bind(metrics.cpu_utilization_percent)
        .bind(metrics.first_token_latency_ms.map(|l| l as i64))
        .execute(&self.pool)
        .await
        .context("Failed to store performance history")?;
        
        Ok(())
    }
    
    /// Store availability check in history
    pub async fn store_availability_history(
        &self,
        model_id: &str,
        availability: &ModelAvailability,
    ) -> Result<()> {
        sqlx::query(r#"
            INSERT INTO model_availability_history (
                model_id, checked_at, is_available, response_time_ms, uptime_percent
            ) VALUES (?, ?, ?, ?, ?)
        "#)
        .bind(model_id)
        .bind(availability.last_checked.to_rfc3339())
        .bind(availability.is_available)
        .bind(availability.response_time_ms.map(|t| t as i64))
        .bind(availability.uptime_percent)
        .execute(&self.pool)
        .await
        .context("Failed to store availability history")?;
        
        Ok(())
    }
    
    /// Get performance history for a model
    pub async fn get_performance_history(
        &self,
        model_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<(DateTime<Utc>, PerformanceMetrics)>> {
        let limit = limit.unwrap_or(100);
        
        let rows = sqlx::query(r#"
            SELECT * FROM model_performance_history
            WHERE model_id = ?
            ORDER BY measured_at DESC
            LIMIT ?
        "#)
        .bind(model_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .context("Failed to query performance history")?;
        
        let mut history = Vec::new();
        for row in rows {
            let measured_at = DateTime::parse_from_rfc3339(&row.get::<String, _>("measured_at"))?
                .with_timezone(&Utc);
            
            let metrics = PerformanceMetrics {
                tokens_per_second: row.get("tokens_per_second"),
                latency_ms: row.get::<i64, _>("latency_ms") as u64,
                throughput_requests_per_minute: row.get("throughput_requests_per_minute"),
                memory_usage_gb: row.get("memory_usage_gb"),
                gpu_utilization_percent: row.get("gpu_utilization_percent"),
                cpu_utilization_percent: row.get("cpu_utilization_percent"),
                first_token_latency_ms: row.get::<Option<i64>, _>("first_token_latency_ms")
                    .map(|l| l as u64),
            };
            
            history.push((measured_at, metrics));
        }
        
        Ok(history)
    }
    
    /// Clean up old historical data
    pub async fn cleanup_old_data(&self, days_to_keep: u32) -> Result<()> {
        let cutoff_date = Utc::now() - chrono::Duration::days(days_to_keep as i64);
        let cutoff_str = cutoff_date.to_rfc3339();
        
        // Clean up old performance history
        let deleted_performance = sqlx::query(
            "DELETE FROM model_performance_history WHERE measured_at < ?"
        )
        .bind(&cutoff_str)
        .execute(&self.pool)
        .await
        .context("Failed to cleanup old performance history")?
        .rows_affected();
        
        // Clean up old availability history
        let deleted_availability = sqlx::query(
            "DELETE FROM model_availability_history WHERE checked_at < ?"
        )
        .bind(&cutoff_str)
        .execute(&self.pool)
        .await
        .context("Failed to cleanup old availability history")?
        .rows_affected();
        
        info!(
            "Cleaned up {} performance records and {} availability records older than {} days",
            deleted_performance, deleted_availability, days_to_keep
        );
        
        Ok(())
    }
    
    /// Record performance metrics for a model
    pub async fn record_performance(&self, record: &PerformanceRecord) -> Result<()> {
        let timestamp_chrono = chrono::Utc::now(); // Convert Instant to DateTime
        
        sqlx::query(r#"
            INSERT INTO model_performance_history 
            (model_id, measured_at, response_time_ms, tokens_per_second, success_rate, error_rate, quality_score, cost_per_token, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind("default_model") // We'll need model_id from somewhere else
        .bind(timestamp_chrono)
        .bind(record.metrics.execution_time_ms as i64)
        .bind(0.0) // tokens_per_second - calculate from metrics
        .bind(1.0) // success_rate - default to success
        .bind(0.0) // error_rate
        .bind(record.metrics.quality_score)
        .bind(0.001) // cost_per_token - default
        .bind(record.metrics.tokens_used as i64) // input_tokens
        .bind(record.metrics.tokens_used as i64) // output_tokens
        .execute(&self.pool)
        .await
        .context("Failed to record performance metrics")?;
        
        Ok(())
    }

    /// Convert database row to ModelInfo
    fn row_to_model_info(&self, row: sqlx::sqlite::SqliteRow) -> Result<ModelInfo> {
        let capabilities: Vec<ModelCapability> = serde_json::from_str(&row.get::<String, _>("capabilities"))?;
        let strengths = serde_json::from_str(&row.get::<String, _>("strengths"))?;
        let weaknesses = serde_json::from_str(&row.get::<String, _>("weaknesses"))?;
        let ideal_use_cases = serde_json::from_str(&row.get::<String, _>("ideal_use_cases"))?;
        let supported_formats = serde_json::from_str(&row.get::<String, _>("supported_formats"))?;
        let performance_metrics = serde_json::from_str(&row.get::<String, _>("performance_metrics"))?;
        let quality_scores = serde_json::from_str(&row.get::<String, _>("quality_scores"))?;
        let benchmark_results = serde_json::from_str(&row.get::<String, _>("benchmark_results"))?;
        let availability = serde_json::from_str(&row.get::<String, _>("availability"))?;
        let cost_info = serde_json::from_str(&row.get::<String, _>("cost_info"))?;
        let deployment_info = serde_json::from_str(&row.get::<String, _>("deployment_info"))?;
        let metadata = serde_json::from_str(&row.get::<String, _>("metadata"))?;
        
        let provider: ModelProviderType = serde_json::from_str(&row.get::<String, _>("provider"))?;
        
        let training_cutoff = row.get::<Option<String>, _>("training_cutoff")
            .map(|s| DateTime::parse_from_rfc3339(&s).map(|dt| dt.with_timezone(&Utc)))
            .transpose()?;
        
        let discovered_at = DateTime::parse_from_rfc3339(&row.get::<String, _>("discovered_at"))?
            .with_timezone(&Utc);
        let last_updated = DateTime::parse_from_rfc3339(&row.get::<String, _>("last_updated"))?
            .with_timezone(&Utc);
        
        Ok(ModelInfo {
            id: row.get("id"),
            name: row.get("name"),
            provider,
            version: row.get("version"),
            model_family: row.get("model_family"),
            parameter_count: row.get::<Option<i64>, _>("parameter_count").map(|p| p as u64),
            context_window: row.get::<i64, _>("context_window") as u32,
            max_output_tokens: row.get::<Option<i64>, _>("max_output_tokens").map(|t| t as u32),
            architecture: row.get("architecture"),
            training_cutoff,
            capabilities,
            strengths,
            weaknesses,
            ideal_use_cases,
            supported_formats,
            performance_metrics,
            quality_scores,
            benchmark_results,
            availability,
            cost_info,
            deployment_info,
            discovered_at,
            last_updated,
            metadata,
        })
    }
}
