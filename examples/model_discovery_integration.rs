// examples/model_discovery_integration.rs
//! Example showing how to integrate the model discovery system
//! 
//! This example demonstrates:
//! - Setting up the model discovery service with Ollama provider
//! - Running discovery to find and catalog models
//! - Using the database for persistent storage
//! - Setting up automated scheduling
//! - Querying models by capabilities and use cases

use docs_mcp_server::model_discovery::{
    ModelDiscoveryService, ModelDatabase, OllamaModelProvider,
    ModelDiscoveryScheduler, SchedulerConfig,
    ModelConstraints, UseCase, ModelProviderType
};
use docs_mcp_server::agents::ModelCapability;
use docs_mcp_server::database::Database;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting Model Discovery System Integration Example");

    // 1. Set up database connection
    let database_url = "sqlite:test_data/model_discovery.db";
    let main_db = Database::new(database_url).await?;
    let model_db = ModelDatabase::new(main_db.pool.clone());
    
    // Initialize model discovery tables
    model_db.initialize().await?;
    
    // 2. Create model discovery service with database
    let mut discovery_service = ModelDiscoveryService::with_database(model_db.clone());
    
    // 3. Add Ollama provider
    let ollama_provider = OllamaModelProvider::new("http://127.0.0.1:11434".to_string())?;
    discovery_service.add_provider(Box::new(ollama_provider));
    
    // 4. Run initial discovery
    info!("Running initial model discovery...");
    let discovered_models = discovery_service.discover_all_models().await?;
    info!("Discovered {} models", discovered_models.len());
    
    // Display discovered models
    for model in &discovered_models {
        info!("Model: {} ({:?})", model.name, model.provider);
        info!("  Capabilities: {:?}", model.capabilities);
        info!("  Use Cases: {:?}", model.ideal_use_cases);
        info!("  Performance: {:.1} tokens/sec, {}ms latency", 
            model.performance_metrics.tokens_per_second,
            model.performance_metrics.latency_ms
        );
        info!("  Quality Score: {:.2}", model.quality_scores.overall);
        println!();
    }
    
    // 5. Demonstrate model querying by capabilities
    info!("=== Querying Models by Capabilities ===");
    
    let code_models = discovery_service.get_models_by_capability(&ModelCapability::CodeGeneration);
    info!("Found {} models with CodeGeneration capability:", code_models.len());
    for model in code_models {
        info!("  - {} (Score: {:.2})", model.name, model.quality_scores.overall);
    }
    
    let reasoning_models = discovery_service.get_models_by_capability(&ModelCapability::Reasoning);
    info!("Found {} models with Reasoning capability:", reasoning_models.len());
    for model in reasoning_models {
        info!("  - {} (Score: {:.2})", model.name, model.quality_scores.overall);
    }
    
    // 6. Demonstrate model recommendation system
    info!("=== Model Recommendation System ===");
    
    // Recommend best model for code generation
    let code_recommendation = discovery_service.recommend_model(
        &UseCase::CodeGeneration,
        &[ModelCapability::CodeGeneration, ModelCapability::CodeUnderstanding],
        None
    );
    
    if let Some(model) = code_recommendation {
        info!("Best model for code generation: {} (Score: {:.2})", 
            model.name, model.quality_scores.overall);
        info!("  Strengths: {:?}", model.strengths);
        info!("  Performance: {:.1} tokens/sec", model.performance_metrics.tokens_per_second);
    } else {
        info!("No suitable model found for code generation");
    }
    
    // Recommend with constraints
    let constraints = ModelConstraints {
        max_latency_ms: Some(1000),
        min_tokens_per_second: Some(40.0),
        max_cost_per_token: None,
        required_provider: Some(ModelProviderType::Ollama),
        local_only: true,
        max_memory_gb: Some(8.0),
    };
    
    let fast_model = discovery_service.recommend_model(
        &UseCase::QuestionAnswering,
        &[ModelCapability::TextGeneration, ModelCapability::Reasoning],
        Some(&constraints)
    );
    
    if let Some(model) = fast_model {
        info!("Best fast model for Q&A: {} ({}ms latency, {:.1} tokens/sec)", 
            model.name, 
            model.performance_metrics.latency_ms,
            model.performance_metrics.tokens_per_second
        );
    }
    
    // Get top 3 recommendations
    let top_recommendations = discovery_service.get_top_recommendations(
        &UseCase::Documentation,
        &[ModelCapability::Documentation, ModelCapability::Explanation],
        3,
        None
    );
    
    info!("Top 3 models for documentation:");
    for (i, (model, score)) in top_recommendations.iter().enumerate() {
        info!("  {}. {} (Score: {:.3})", i + 1, model.name, score);
    }
    
    // 7. Set up automated scheduling (in production environment)
    info!("=== Setting up Automated Scheduler ===");
    
    let scheduler_config = SchedulerConfig {
        discovery_interval_hours: 24,     // Daily discovery
        availability_check_interval_minutes: 30, // Every 30 minutes
        performance_check_interval_hours: 6,     // Every 6 hours
        history_retention_days: 30,       // Keep 30 days of history
        run_on_startup: false,           // We already ran discovery
    };
    
    let service_arc = Arc::new(RwLock::new(discovery_service));
    let database_arc = Arc::new(model_db);
    
    let mut scheduler = ModelDiscoveryScheduler::new(
        scheduler_config,
        service_arc.clone(),
        database_arc.clone()
    );
    
    info!("Scheduler configured successfully");
    let status = scheduler.get_status();
    info!("Scheduler status: {:?}", status.config);
    
    // 8. Demonstrate database queries
    info!("=== Database Integration ===");
    
    // Query models by provider from database
    let ollama_models = database_arc.get_models_by_provider(&ModelProviderType::Ollama).await?;
    info!("Found {} Ollama models in database", ollama_models.len());
    
    // Query models by capability from database
    let code_models_db = database_arc.get_models_by_capability(&ModelCapability::CodeGeneration).await?;
    info!("Found {} code generation models in database", code_models_db.len());
    
    // Get performance history for a model (if any)
    if let Some(first_model) = discovered_models.first() {
        let performance_history = database_arc.get_performance_history(&first_model.id, Some(10)).await?;
        info!("Performance history for {}: {} entries", first_model.name, performance_history.len());
    }
    
    // 9. Simulate updates and monitoring
    info!("=== Simulating Model Updates ===");
    
    // In a real application, you would start the scheduler here:
    // scheduler.start().await?;
    
    // For demonstration, we'll show how to manually update model data
    let service = service_arc.read().await;
    if let Some(model) = service.get_all_models().first() {
        info!("Example model update for: {}", model.name);
        info!("  Current performance: {:.1} tokens/sec, {}ms latency",
            model.performance_metrics.tokens_per_second,
            model.performance_metrics.latency_ms
        );
        
        // In practice, you would measure new performance and update:
        // service.update_model_performance(&model.id, new_metrics).await?;
    }
    
    info!("Model Discovery System Integration Example completed successfully!");
    info!("In production, the scheduler would continue running in the background,");
    info!("automatically discovering new models and updating performance metrics.");
    
    Ok(())
}

/// Helper function to display model summary
fn display_model_summary(models: &[docs_mcp_server::model_discovery::ModelInfo]) {
    use std::collections::HashMap;
    
    let mut provider_counts = HashMap::new();
    let mut capability_counts = HashMap::new();
    
    for model in models {
        *provider_counts.entry(&model.provider).or_insert(0) += 1;
        
        for capability in &model.capabilities {
            *capability_counts.entry(capability).or_insert(0) += 1;
        }
    }
    
    info!("=== Model Discovery Summary ===");
    info!("Total models: {}", models.len());
    
    info!("By provider:");
    for (provider, count) in provider_counts {
        info!("  {:?}: {}", provider, count);
    }
    
    info!("By capability:");
    for (capability, count) in capability_counts {
        info!("  {:?}: {}", capability, count);
    }
}

/// Example showing how to use the model discovery system for specific tasks
fn demonstrate_task_specific_usage() {
    info!("=== Task-Specific Model Usage Examples ===");
    
    // Code Generation Task
    info!("For code generation, you would:");
    info!("1. Query models with CodeGeneration + CodeUnderstanding capabilities");
    info!("2. Apply constraints (e.g., local deployment, performance requirements)");
    info!("3. Rank by code generation quality scores");
    info!("4. Select highest-scoring available model");
    
    // Documentation Task
    info!("For documentation generation, you would:");
    info!("1. Query models with Documentation + Explanation capabilities");
    info!("2. Prefer models with strong natural language understanding");
    info!("3. Consider model's training data recency for accuracy");
    info!("4. Balance quality vs. speed based on use case");
    
    // Learning Assistant Task
    info!("For learning assistance, you would:");
    info!("1. Query models with Reasoning + Explanation capabilities");
    info!("2. Prioritize safety and alignment scores");
    info!("3. Consider pedagogical strengths and weaknesses");
    info!("4. Select models with good instruction following");
}
