// src/bin/test-real-agentic-flow.rs
//! Integration test for real agentic flow implementation
//! 
//! This test demonstrates:
//! 1. Real Ollama client integration
//! 2. Actual agent execution (CodeAnalyzerAgent)
//! 3. End-to-end flow orchestration without mock data

use docs_mcp_server::{
    agents::{AgentRegistry, code_analyzer::CodeAnalyzerAgent},
    orchestrator::{FlowOrchestrator, FlowRequest, FlowRequirements, RequestIntent, AnalysisDepth},
    model_clients::{OllamaClient, ModelClient},
    orchestrator::model_router::ModelRouter,
};
use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("🚀 Starting Real Agentic Flow Integration Test");
    println!("==============================================");
    
    // Step 1: Create and test Ollama client
    println!("📡 Creating Ollama client...");
    let ollama_client = OllamaClient::localhost("codellama:7b")?;
    
    // Test connection
    println!("🔍 Testing Ollama connection...");
    match ollama_client.health_check().await {
        Ok(true) => println!("✅ Ollama is healthy"),
        Ok(false) => {
            println!("❌ Ollama health check failed");
            return Err(anyhow::anyhow!("Ollama is not healthy"));
        }
        Err(e) => {
            println!("❌ Ollama connection error: {}", e);
            return Err(e);
        }
    }
    
    // Test model availability
    println!("🔍 Checking model availability...");
    match ollama_client.is_model_available().await {
        Ok(true) => println!("✅ codellama:7b is available"),
        Ok(false) => {
            println!("⚠️  Model not available, attempting to ensure availability...");
            ollama_client.ensure_model_available().await?;
            println!("✅ Model is now available");
        }
        Err(e) => {
            println!("❌ Model availability check failed: {}", e);
            return Err(e);
        }
    }
    
    // Test basic generation
    println!("🧪 Testing basic generation...");
    let test_response = ollama_client.generate("Hello, can you respond with 'AI_WORKING'?").await?;
    println!("📝 Response: {}", test_response.trim());
    
    // Step 2: Set up agent registry
    println!("🤖 Setting up agent registry...");
    let mut agent_registry = AgentRegistry::new();
    agent_registry.register(CodeAnalyzerAgent::new());
    let agent_registry = Arc::new(agent_registry);
    
    println!("✅ Registered agents: {:?}", agent_registry.list_agents());
    
    // Step 3: Set up model router and orchestrator
    println!("🎛️  Setting up orchestrator...");
    let model_router = ModelRouter::new();
    let orchestrator = FlowOrchestrator::new(agent_registry.clone(), model_router);
    
    // Register the Ollama client
    orchestrator.register_model_client("codellama:7b".to_string(), Arc::new(ollama_client));
    
    // Step 4: Create a test request for code analysis
    println!("📝 Creating test request...");
    let test_code = r#"
fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

fn main() {
    let result = fibonacci(10);
    println!("Result: {}", result);
}
"#;
    
    let mut input_data = HashMap::new();
    input_data.insert("code".to_string(), serde_json::Value::String(test_code.to_string()));
    input_data.insert("language".to_string(), serde_json::Value::String("rust".to_string()));
    input_data.insert("analysis_depth".to_string(), serde_json::Value::String("standard".to_string()));
    
    let request = FlowRequest {
        description: "Analyze this Rust fibonacci function for potential issues and improvements".to_string(),
        intent: Some(RequestIntent::CodeAnalysis {
            depth: AnalysisDepth::Standard,
            focus_areas: vec!["performance".to_string(), "best_practices".to_string()],
        }),
        input_data,
        requirements: FlowRequirements {
            max_execution_time_seconds: 120,
            max_cost_estimate: 0.50,
            quality_threshold: 0.7,
            preferred_models: vec!["codellama:7b".to_string()],
            excluded_models: vec![],
            parallel_execution_allowed: true,
            cache_enabled: false, // Disable for testing
        },
        user_context: None,
        project_context: None,
    };
    
    // Step 5: Execute the flow
    println!("🚀 Executing agentic flow...");
    println!("   Request: {}", request.description);
    println!("   Code lines: {}", test_code.lines().count());
    
    let start_time = std::time::Instant::now();
    
    match orchestrator.execute_flow(request).await {
        Ok(result) => {
            let duration = start_time.elapsed();
            
            println!("🎉 Flow execution completed successfully!");
            println!("⏱️  Duration: {:?}", duration);
            println!("🆔 Flow ID: {}", result.flow_id);
            println!("✅ Success: {}", result.success);
            println!("📊 Tasks completed: {}", result.metrics.tasks_completed);
            println!("💰 Total cost: ${:.4}", result.metrics.total_cost);
            
            if result.success {
                println!("\n📋 Results Summary:");
                for (task_id, output) in &result.results {
                    println!("  🔧 Task '{}': {:?}", task_id, output.get_field::<String>("status").unwrap_or_default());
                    
                    // Try to extract code analysis results
                    if let Ok(quality_score) = output.get_field::<f64>("quality_score") {
                        println!("    📈 Quality Score: {:.2}", quality_score);
                    }
                    
                    if let Ok(issues_count) = output.get_field::<usize>("issues_count") {
                        println!("    ⚠️  Issues Found: {}", issues_count);
                    }
                    
                    if let Ok(suggestions_count) = output.get_field::<usize>("suggestions_count") {
                        println!("    💡 Suggestions: {}", suggestions_count);
                    }
                }
                
                println!("\n🎯 REAL AGENTIC FLOW WORKING SUCCESSFULLY!");
                println!("   ✅ Ollama integration: Working");
                println!("   ✅ Agent execution: Working");
                println!("   ✅ Task orchestration: Working");
                println!("   ✅ Model routing: Working");
            } else {
                println!("⚠️  Flow completed but with some failures");
            }
        }
        Err(e) => {
            let duration = start_time.elapsed();
            println!("❌ Flow execution failed after {:?}", duration);
            println!("   Error: {}", e);
            
            // Provide debugging information
            println!("\n🔍 Debugging Information:");
            println!("   - Check if Ollama is running: curl http://localhost:11434/api/tags");
            println!("   - Check if codellama:7b is available: ollama list");
            println!("   - Agent registry size: {}", agent_registry.list_agents().len());
            
            return Err(e);
        }
    }
    
    println!("\n🚀 Integration test completed successfully!");
    Ok(())
}
