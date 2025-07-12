// examples/comprehensive_model_evaluation_demo.rs
//! Demonstration of the comprehensive model evaluation system
//! 
//! This example shows how the system:
//! 1. Discovers models from providers (e.g., Ollama)
//! 2. Runs comprehensive evaluations including:
//!    - Performance testing (speed, latency)
//!    - Quality assessment (accuracy, coherence)  
//!    - Task-specific capability testing
//!    - External context gathering via web search
//! 3. Provides detailed evaluation reports and recommendations

use docs_mcp_server::{
    model_discovery::{
        ModelDiscoveryService, OllamaModelProvider, ModelDatabase,
        evaluator::{ModelEvaluator, EvaluationConfig, CriteriaWeights},
        UseCase,
    },
    model_clients::{OllamaClient, ModelClient},
    web_search::WebSearchEngine,
    database::Database,
};
use std::sync::Arc;
use anyhow::Result;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔍 Comprehensive Model Evaluation Demo");
    println!("=====================================");
    println!();
    
    // Step 1: Set up the evaluation environment
    println!("📋 Setting up evaluation environment...");
    
    // Create database for storing evaluation results
    let database = Database::new("docs.db").await?;
    let model_db = ModelDatabase::new(database.pool().clone());
    
    // Create web search engine for external context gathering
    let web_search = Arc::new(WebSearchEngine::new());
    
    // Configure comprehensive evaluation settings
    let eval_config = EvaluationConfig {
        test_timeout_seconds: 120,
        max_retries: 3,
        enable_web_context: true,
        web_search_results: 15,
        enable_performance_tests: true,
        enable_quality_tests: true,
        criteria_weights: CriteriaWeights {
            accuracy: 0.30,          // Prioritize accuracy
            speed: 0.20,             // Speed is important
            quality: 0.25,           // Quality matters
            consistency: 0.15,       // Consistency is key
            safety: 0.05,            // Basic safety
            cost_effectiveness: 0.03,
            context_handling: 0.02,
            instruction_following: 0.0,
        },
    };
    
    // Step 2: Set up model discovery with comprehensive evaluation
    println!("🤖 Setting up model discovery with evaluation capabilities...");
    
    let mut discovery_service = ModelDiscoveryService::with_database(model_db)
        .with_evaluator(web_search.clone())
        .with_evaluation_config(eval_config);
    
    // Add Ollama provider
    let ollama_provider = OllamaModelProvider::new("http://localhost:11434".to_string())?;
    discovery_service.add_provider(Box::new(ollama_provider));
    
    // Step 3: Create model client for testing
    println!("🔗 Setting up model client for testing...");
    let ollama_client = Arc::new(OllamaClient::new("http://localhost:11434".to_string(), "llama2".to_string())?);
    
    // Test connection
    match ollama_client.generate_with_context(&docs_mcp_server::model_clients::ModelRequest {
        prompt: "Hello! Can you respond with 'EVALUATION_TEST_OK'?".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.1),
        ..Default::default()
    }).await {
        Ok(response) => {
            println!("✅ Model client connection successful");
            println!("   Response: {}", response.content.trim());
        }
        Err(e) => {
            println!("❌ Model client connection failed: {}", e);
            println!("   Make sure Ollama is running at http://localhost:11434");
            return Ok(());
        }
    }
    println!();
    
    // Step 4: Discover and evaluate models
    println!("🔍 Starting comprehensive model discovery and evaluation...");
    println!("This will:");
    println!("  • Discover available models from Ollama");
    println!("  • Test each model's performance (speed, latency)");
    println!("  • Evaluate quality across multiple tasks");
    println!("  • Gather external context via web search");
    println!("  • Generate detailed capability assessments");
    println!();
    
    let evaluation_results = discovery_service
        .discover_and_evaluate_models(ollama_client.clone())
        .await?;
    
    // Step 5: Display comprehensive results
    println!("📊 COMPREHENSIVE EVALUATION RESULTS");
    println!("===================================");
    println!();
    
    if evaluation_results.is_empty() {
        println!("⚠️  No models found. Make sure Ollama is running with models installed.");
        println!("   Try: ollama pull codellama:7b");
        return Ok(());
    }
    
    for (i, (model, evaluation)) in evaluation_results.iter().enumerate() {
        println!("🤖 MODEL {} - {}", i + 1, model.name);
        println!("   Provider: {:?}", model.provider);
        println!("   Context Window: {} tokens", model.context_window);
        
        if let Some(eval) = evaluation {
            println!("   📈 EVALUATION RESULTS:");
            println!("      Overall Score: {:.1}/100", eval.overall_score);
            println!("      Evaluation Date: {}", eval.evaluation_timestamp.format("%Y-%m-%d %H:%M UTC"));
            
            // Performance metrics
            println!("   ⚡ PERFORMANCE:");
            println!("      Speed: {:.1} tokens/sec", eval.performance_metrics.tokens_per_second);
            println!("      Latency: {}ms", eval.performance_metrics.latency_ms);
            println!("      Throughput: {:.1} req/min", eval.performance_metrics.throughput_requests_per_minute);
            
            // Quality scores
            println!("   🎯 QUALITY SCORES:");
            println!("      Overall Quality: {:.1}/100", eval.quality_scores.overall);
            println!("      Accuracy: {:.1}/100", eval.quality_scores.accuracy);
            println!("      Coherence: {:.1}/100", eval.quality_scores.coherence);
            println!("      Relevance: {:.1}/100", eval.quality_scores.relevance);
            println!("      Safety: {:.1}/100", eval.quality_scores.safety);
            
            // Capability scores
            if !eval.capability_scores.is_empty() {
                println!("   🧠 CAPABILITY SCORES:");
                for (capability, score) in &eval.capability_scores {
                    println!("      {}: {:.1}/100", capability.replace('_', " ").to_uppercase(), score);
                }
            }
            
            // Test results summary
            println!("   📋 TEST SUMMARY:");
            println!("      Tests Run: {}", eval.evaluation_metadata.total_tests_run);
            println!("      Tests Passed: {}", eval.evaluation_metadata.tests_passed);
            println!("      Pass Rate: {:.1}%", 
                if eval.evaluation_metadata.total_tests_run > 0 {
                    (eval.evaluation_metadata.tests_passed as f64 / eval.evaluation_metadata.total_tests_run as f64) * 100.0
                } else { 0.0 }
            );
            println!("      Evaluation Time: {:.1}s", eval.evaluation_metadata.total_evaluation_time_ms as f64 / 1000.0);
            
            // Strengths and weaknesses
            if !eval.strengths.is_empty() {
                println!("   💪 STRENGTHS:");
                for strength in &eval.strengths {
                    println!("      • {:?}", strength);
                }
            }
            
            if !eval.weaknesses.is_empty() {
                println!("   ⚠️  WEAKNESSES:");
                for weakness in &eval.weaknesses {
                    println!("      • {:?}", weakness);
                }
            }
            
            // Recommended use cases
            if !eval.recommended_use_cases.is_empty() {
                println!("   🎯 RECOMMENDED FOR:");
                for use_case in &eval.recommended_use_cases {
                    println!("      • {:?}", use_case);
                }
            }
            
            // External context insights
            if !eval.external_context.web_search_summary.is_empty() {
                println!("   🌐 EXTERNAL CONTEXT:");
                let summary_lines: Vec<&str> = eval.external_context.web_search_summary
                    .lines()
                    .take(3)
                    .collect();
                for line in summary_lines {
                    if !line.trim().is_empty() {
                        println!("      {}", line.trim());
                    }
                }
                if eval.external_context.popularity_metrics.community_mentions > 0 {
                    println!("      Community Mentions: {}", eval.external_context.popularity_metrics.community_mentions);
                }
            }
            
            // Detailed test results
            if !eval.test_results.is_empty() {
                println!("   🧪 DETAILED TEST RESULTS:");
                for (test_id, result) in &eval.test_results {
                    let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
                    println!("      {} {} - Score: {:.1}/100 ({}ms)", 
                        status, result.test_name, result.score, result.execution_time_ms);
                    
                    if let Some(ref error) = result.error_message {
                        println!("         Error: {}", error);
                    }
                }
            }
            
            // Benchmark results
            if !eval.benchmark_results.is_empty() {
                println!("   📊 BENCHMARK RESULTS:");
                for benchmark in &eval.benchmark_results {
                    println!("      {}: {:.1}", benchmark.benchmark_name, benchmark.score);
                }
            }
        } else {
            println!("   ⚠️  Evaluation failed or skipped");
        }
        
        println!();
    }
    
    // Step 6: Generate overall summary
    println!("📈 EVALUATION SUMMARY");
    println!("====================");
    
    let summary = discovery_service.get_evaluation_summary();
    println!("Total Models Discovered: {}", summary.total_models);
    println!("Models Successfully Evaluated: {}", summary.evaluated_models);
    
    if summary.evaluated_models > 0 {
        println!("Average Score: {:.1}/100", summary.average_score);
        println!("Highest Score: {:.1}/100", summary.highest_score);
        println!("Lowest Score: {:.1}/100", summary.lowest_score);
        
        if let Some(ref best_model) = summary.best_model {
            println!("Best Model: {}", best_model);
        }
    }
    println!();
    
    // Step 7: Demonstrate model recommendations
    println!("🎯 MODEL RECOMMENDATIONS");
    println!("========================");
    
    let use_cases = vec![
        UseCase::CodeGeneration,
        UseCase::CodeAnalysis,
        UseCase::Documentation,
        UseCase::Debugging,
        UseCase::QuestionAnswering,
    ];
    
    for use_case in use_cases {
        if let Some(best_model) = discovery_service.get_best_model_for_use_case(&use_case).await {
            println!("Best for {:?}: {} (Score: {})", 
                use_case, 
                best_model.name,
                best_model.metadata.get("evaluation_score").unwrap_or(&"N/A".to_string())
            );
        } else {
            println!("Best for {:?}: No suitable model found", use_case);
        }
    }
    println!();
    
    // Step 8: Show how to get evaluated models ranked by score
    println!("🏆 MODELS RANKED BY EVALUATION SCORE");
    println!("====================================");
    
    let ranked_models = discovery_service.get_evaluated_models().await?;
    for (i, (model, score)) in ranked_models.iter().enumerate().take(5) {
        match score {
            Some(s) => println!("{}. {} - {:.1}/100", i + 1, model.name, s),
            None => println!("{}. {} - Not evaluated", i + 1, model.name),
        }
    }
    
    println!();
    println!("✅ Comprehensive model evaluation completed!");
    println!("   Evaluation results have been stored in the database for future reference.");
    println!("   You can re-run evaluations as models are updated or new tests are added.");
    
    Ok(())
}
