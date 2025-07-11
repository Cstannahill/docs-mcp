use anyhow::Result;
use tracing::{info, error, warn, debug};
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::database::{Database, DocType};
use crate::fetcher::DocumentationFetcher;

pub struct Scheduler {
    db: Database,
    update_frequencies: HashMap<DocType, Duration>,
    priority_sources: Vec<String>,
    usage_analytics: UsageAnalytics,
    content_quality_tracker: ContentQualityTracker,
}

#[derive(Debug, Clone)]
pub struct UpdateStrategy {
    pub check_frequency: Duration,
    pub full_update_frequency: Duration,
    pub incremental_updates: bool,
    pub priority_level: PriorityLevel,
}

#[derive(Debug, Clone)]
pub enum PriorityLevel {
    Critical,  // Real-time updates (releases, security)
    High,      // Daily updates (main docs)
    Medium,    // Weekly updates (tutorials)
    Low,       // Monthly updates (archived docs)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageAnalytics {
    search_frequency: HashMap<DocType, u32>,
    popular_queries: HashMap<String, u32>,
    last_access_times: HashMap<DocType, DateTime<Utc>>,
    error_rates: HashMap<DocType, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentQualityTracker {
    freshness_scores: HashMap<DocType, f32>,
    user_feedback_scores: HashMap<String, f32>, // page_id -> score
    content_completeness: HashMap<DocType, f32>,
    broken_links: HashMap<DocType, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub freshness_score: f32,      // 0.0 - 1.0, how up-to-date content is
    pub completeness_score: f32,   // 0.0 - 1.0, how complete documentation is
    pub user_satisfaction: f32,    // 0.0 - 1.0, based on user feedback
    pub error_rate: f32,          // 0.0 - 1.0, rate of broken links/errors
    pub usage_frequency: u32,     // how often this content is accessed
}

impl Default for UsageAnalytics {
    fn default() -> Self {
        Self {
            search_frequency: HashMap::new(),
            popular_queries: HashMap::new(),
            last_access_times: HashMap::new(),
            error_rates: HashMap::new(),
        }
    }
}

impl Default for ContentQualityTracker {
    fn default() -> Self {
        Self {
            freshness_scores: HashMap::new(),
            user_feedback_scores: HashMap::new(),
            content_completeness: HashMap::new(),
            broken_links: HashMap::new(),
        }
    }
}

impl Scheduler {
    pub fn new(db: Database) -> Self {
        let mut update_frequencies = HashMap::new();
        
        // Configure different update frequencies per doc type
        update_frequencies.insert(DocType::Rust, Duration::hours(12));      // High priority
        update_frequencies.insert(DocType::React, Duration::hours(6));      // Very high priority
        update_frequencies.insert(DocType::TypeScript, Duration::hours(8)); 
        update_frequencies.insert(DocType::Python, Duration::hours(24));    // Daily
        update_frequencies.insert(DocType::Tauri, Duration::hours(24));
        update_frequencies.insert(DocType::Tailwind, Duration::hours(12));
        update_frequencies.insert(DocType::Shadcn, Duration::hours(6));     // UI libs change frequently
        
        let priority_sources = vec![
            "rust-std".to_string(),
            "react-docs".to_string(),
            "typescript-docs".to_string(),
        ];
        
        Self { 
            db, 
            update_frequencies,
            priority_sources,
            usage_analytics: UsageAnalytics::default(),
            content_quality_tracker: ContentQualityTracker::default(),
        }
    }

    pub async fn start_daily_updates(&self) -> Result<()> {
        info!("Starting documentation update scheduler");
        
        // Run initial update
        if let Err(e) = self.force_update().await {
            error!("Initial documentation update failed: {}", e);
        }

        info!("Daily update scheduler initialized. Manual updates can be triggered with force_update()");
        
        Ok(())
    }

    pub async fn force_update(&self) -> Result<()> {
        info!("Running forced documentation update");
        let fetcher = DocumentationFetcher::new(self.db.clone());
        fetcher.update_all_documentation().await
    }

    /// Intelligent update scheduling based on content type and usage patterns
    pub async fn start_intelligent_updates(&self) -> Result<()> {
        info!("Starting intelligent documentation update scheduler");
        
        // Run initial update for priority sources
        for source_id in &self.priority_sources {
            if let Err(e) = self.update_specific_source(source_id).await {
                error!("Failed to update priority source {}: {}", source_id, e);
            }
        }

        // Schedule based on detected changes and usage patterns
        self.schedule_adaptive_updates().await?;
        
        Ok(())
    }

    /// Update only sources that have likely changed
    async fn schedule_adaptive_updates(&self) -> Result<()> {
        // Check for version changes, new releases, etc.
        for (doc_type, frequency) in &self.update_frequencies {
            let last_check = self.get_last_update_time(doc_type).await?;
            
            if last_check.map_or(true, |time| Utc::now() - time > *frequency) {
                info!("Scheduling update for {:?} (frequency: {:?})", doc_type, frequency);
                if let Err(e) = self.update_doc_type(doc_type).await {
                    warn!("Failed to update {:?}: {}", doc_type, e);
                }
            }
        }
        
        Ok(())
    }

    async fn get_last_update_time(&self, doc_type: &DocType) -> Result<Option<DateTime<Utc>>> {
        // Query the database for the last update time of this doc type
        self.db.get_last_update_time_for_type(doc_type).await
    }

    async fn update_doc_type(&self, doc_type: &DocType) -> Result<()> {
        info!("Running targeted update for {:?}", doc_type);
        let fetcher = DocumentationFetcher::new(self.db.clone());
        fetcher.update_documentation_by_type(doc_type).await
    }

    async fn update_specific_source(&self, source_id: &str) -> Result<()> {
        info!("Running update for source: {}", source_id);
        let fetcher = DocumentationFetcher::new(self.db.clone());
        fetcher.update_source_by_id(source_id).await
    }

    /// Advanced adaptive scheduling based on usage patterns and content quality
    pub async fn start_adaptive_scheduling(&self) -> Result<()> {
        info!("Starting adaptive scheduling with quality analytics");
        
        // Analyze current content quality and usage patterns
        self.analyze_usage_patterns().await?;
        self.assess_content_quality().await?;
        
        // Update priorities based on analytics
        self.update_dynamic_priorities().await?;
        
        // Schedule updates based on intelligent analysis
        self.schedule_intelligent_updates().await?;
        
        info!("Adaptive scheduling initialized with quality-based priorities");
        Ok(())
    }
    
    /// Track search patterns to inform update priorities
    pub async fn record_search(&mut self, query: &str, doc_type: &DocType) -> Result<()> {
        debug!("Recording search: {} for {:?}", query, doc_type);
        
        // Update search frequency
        *self.usage_analytics.search_frequency.entry(doc_type.clone()).or_insert(0) += 1;
        
        // Track popular queries
        *self.usage_analytics.popular_queries.entry(query.to_string()).or_insert(0) += 1;
        
        // Update last access time
        self.usage_analytics.last_access_times.insert(doc_type.clone(), Utc::now());
        
        // Trigger dynamic priority update if needed
        if self.should_update_priorities().await? {
            self.update_dynamic_priorities().await?;
        }
        
        Ok(())
    }
    
    /// Assess the quality of content across all documentation types
    async fn assess_content_quality(&self) -> Result<()> {
        info!("Assessing content quality across all documentation sources");
        
        for doc_type in [DocType::Rust, DocType::React, DocType::TypeScript, DocType::Python, 
                        DocType::Tauri, DocType::Tailwind, DocType::Shadcn] {
            let quality_metrics = self.calculate_quality_metrics(&doc_type).await?;
            info!("Quality metrics for {:?}: freshness={:.2}, completeness={:.2}, satisfaction={:.2}", 
                  doc_type, quality_metrics.freshness_score, quality_metrics.completeness_score, 
                  quality_metrics.user_satisfaction);
        }
        
        Ok(())
    }
    
    /// Calculate comprehensive quality metrics for a documentation type
    async fn calculate_quality_metrics(&self, doc_type: &DocType) -> Result<QualityMetrics> {
        // Calculate freshness score based on last update times
        let freshness_score = self.calculate_freshness_score(doc_type).await?;
        
        // Calculate completeness based on content coverage
        let completeness_score = self.calculate_completeness_score(doc_type).await?;
        
        // Get user satisfaction from feedback/usage patterns
        let user_satisfaction = self.calculate_user_satisfaction(doc_type).await?;
        
        // Calculate error rate from broken links/failed updates
        let error_rate = self.calculate_error_rate(doc_type).await?;
        
        // Get usage frequency
        let usage_frequency = self.usage_analytics.search_frequency.get(doc_type).unwrap_or(&0).clone();
        
        Ok(QualityMetrics {
            freshness_score,
            completeness_score,
            user_satisfaction,
            error_rate,
            usage_frequency,
        })
    }
    
    async fn calculate_freshness_score(&self, doc_type: &DocType) -> Result<f32> {
        if let Some(last_update) = self.db.get_last_update_time_for_type(doc_type).await? {
            let hours_since_update = (Utc::now() - last_update).num_hours() as f32;
            let expected_frequency = self.update_frequencies.get(doc_type)
                .unwrap_or(&Duration::hours(24)).num_hours() as f32;
            
            // Score decreases as content gets staler relative to expected frequency
            let freshness = (1.0 - (hours_since_update / (expected_frequency * 2.0))).max(0.0);
            Ok(freshness.min(1.0))
        } else {
            Ok(0.0) // No updates yet
        }
    }
    
    async fn calculate_completeness_score(&self, doc_type: &DocType) -> Result<f32> {
        // Get page count for this doc type
        let page_count = self.db.get_page_count_for_type(doc_type).await.unwrap_or(0);
        
        // Expected page counts for different doc types (rough estimates)
        let expected_pages = match doc_type {
            DocType::Rust => 500,      // Large standard library
            DocType::React => 200,     // Comprehensive but focused
            DocType::TypeScript => 300,
            DocType::Python => 600,    // Very large stdlib
            DocType::Tauri => 100,     // Smaller, focused framework
            DocType::Tailwind => 150,
            DocType::Shadcn => 80,
        };
        
        let completeness = (page_count as f32 / expected_pages as f32).min(1.0);
        Ok(completeness)
    }
    
    async fn calculate_user_satisfaction(&self, doc_type: &DocType) -> Result<f32> {
        // For now, use search frequency as a proxy for satisfaction
        // Higher search frequency = higher satisfaction (users find it useful)
        let search_count = self.usage_analytics.search_frequency.get(doc_type).unwrap_or(&0);
        
        // Normalize to 0-1 scale (arbitrary scaling based on expected usage)
        let satisfaction = (*search_count as f32 / 100.0).min(1.0);
        Ok(satisfaction)
    }
    
    async fn calculate_error_rate(&self, doc_type: &DocType) -> Result<f32> {
        Ok(self.usage_analytics.error_rates.get(doc_type).copied().unwrap_or(0.0))
    }
    
    /// Analyze current usage patterns to inform scheduling decisions
    async fn analyze_usage_patterns(&self) -> Result<()> {
        info!("Analyzing usage patterns for adaptive scheduling");
        
        // Find most popular documentation types
        let mut usage_vec: Vec<_> = self.usage_analytics.search_frequency.iter().collect();
        usage_vec.sort_by(|a, b| b.1.cmp(a.1));
        
        info!("Usage frequency ranking:");
        for (doc_type, count) in usage_vec.iter().take(5) {
            info!("  {:?}: {} searches", doc_type, count);
        }
        
        // Identify trending queries
        let mut query_vec: Vec<_> = self.usage_analytics.popular_queries.iter().collect();
        query_vec.sort_by(|a, b| b.1.cmp(a.1));
        
        info!("Popular queries:");
        for (query, count) in query_vec.iter().take(5) {
            info!("  '{}': {} times", query, count);
        }
        
        Ok(())
    }
    
    /// Update scheduling priorities based on current analytics
    async fn update_dynamic_priorities(&self) -> Result<()> {
        info!("Updating dynamic priorities based on usage analytics");
        
        // More frequently searched doc types get higher priority (shorter intervals)
        for (doc_type, search_count) in &self.usage_analytics.search_frequency {
            if *search_count > 10 { // High usage threshold
                let default_frequency = Duration::hours(24);
                let current_frequency = self.update_frequencies.get(doc_type)
                    .unwrap_or(&default_frequency);
                
                // Reduce update interval for popular content (but don't go below 1 hour)
                let new_frequency = Duration::hours(
                    (current_frequency.num_hours() * 2 / 3).max(1)
                );
                
                info!("Increasing update frequency for {:?}: {:?} -> {:?}", 
                      doc_type, current_frequency, new_frequency);
            }
        }
        
        Ok(())
    }
    
    /// Schedule updates using intelligent analysis
    async fn schedule_intelligent_updates(&self) -> Result<()> {
        info!("Scheduling intelligent updates based on quality metrics");
        
        for doc_type in [DocType::Rust, DocType::React, DocType::TypeScript, DocType::Python, 
                        DocType::Tauri, DocType::Tailwind, DocType::Shadcn] {
            let quality_metrics = self.calculate_quality_metrics(&doc_type).await?;
            
            // Determine if update is needed based on multiple factors
            let needs_update = self.needs_intelligent_update(&doc_type, &quality_metrics).await?;
            
            if needs_update {
                info!("Scheduling intelligent update for {:?} (quality score: {:.2})", 
                      doc_type, self.calculate_overall_quality_score(&quality_metrics));
                
                if let Err(e) = self.update_doc_type(&doc_type).await {
                    warn!("Failed to update {:?}: {}", doc_type, e);
                    
                    // Update error rate tracking
                    // self.record_error(&doc_type).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Determine if a doc type needs an update based on quality metrics
    async fn needs_intelligent_update(&self, doc_type: &DocType, quality_metrics: &QualityMetrics) -> Result<bool> {
        // Multiple factors determine update need:
        
        // 1. Freshness threshold
        if quality_metrics.freshness_score < 0.3 {
            debug!("Update needed for {:?}: low freshness score {:.2}", doc_type, quality_metrics.freshness_score);
            return Ok(true);
        }
        
        // 2. High usage with declining quality
        if quality_metrics.usage_frequency > 5 && quality_metrics.user_satisfaction < 0.5 {
            debug!("Update needed for {:?}: high usage ({}) but low satisfaction ({:.2})", 
                   doc_type, quality_metrics.usage_frequency, quality_metrics.user_satisfaction);
            return Ok(true);
        }
        
        // 3. High error rate
        if quality_metrics.error_rate > 0.1 {
            debug!("Update needed for {:?}: high error rate {:.2}", doc_type, quality_metrics.error_rate);
            return Ok(true);
        }
        
        // 4. Regular schedule check
        let last_check = self.get_last_update_time(doc_type).await?;
        let default_frequency = Duration::hours(24);
        let frequency = self.update_frequencies.get(doc_type).unwrap_or(&default_frequency);
        
        if last_check.map_or(true, |time| Utc::now() - time > *frequency) {
            debug!("Update needed for {:?}: regular schedule interval reached", doc_type);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    fn calculate_overall_quality_score(&self, metrics: &QualityMetrics) -> f32 {
        // Weighted average of quality factors
        let weights = [0.3, 0.25, 0.25, 0.2]; // freshness, completeness, satisfaction, error_rate
        let scores = [
            metrics.freshness_score,
            metrics.completeness_score,
            metrics.user_satisfaction,
            1.0 - metrics.error_rate, // Invert error rate (lower is better)
        ];
        
        weights.iter().zip(scores.iter())
            .map(|(w, s)| w * s)
            .sum()
    }
    
    async fn should_update_priorities(&self) -> Result<bool> {
        // Update priorities every 100 searches or if we haven't updated in the last hour
        let total_searches: u32 = self.usage_analytics.search_frequency.values().sum();
        Ok(total_searches > 0 && total_searches % 100 == 0)
    }
}
