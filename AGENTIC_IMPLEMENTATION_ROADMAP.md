# Agentic Flow Implementation Roadmap

## 🎯 Current State: Honest Assessment

After cleaning up misleading mock implementations, the project now has:

✅ **Working Foundation**:

- Data structures and type definitions
- Task template system that generates task lists
- Dependency analysis and pipeline construction
- Agent registry with capability-based lookup
- Basic intent classification (keyword matching)

❌ **Not Implemented (Causes Real Failures)**:

- Agent execution (returns errors instead of fake success)
- Model client connections to real AI services
- Real metrics collection
- Error handling and retry logic

## 🛠️ Implementation Priority Queue

### Phase 1: Core Agent Execution (Essential)

**Goal**: Make ONE agent actually work end-to-end

1. **Real Model Client Implementation**

   ```rust
   // src/model_clients/ollama_client.rs (NEW FILE)
   pub struct OllamaClient {
       base_url: String,
       model_name: String,
       client: reqwest::Client,
   }

   impl ModelClient for OllamaClient {
       async fn generate(&self, prompt: &str) -> Result<String> {
           // Real HTTP call to Ollama API
           let response = self.client
               .post(&format!("{}/api/generate", self.base_url))
               .json(&serde_json::json!({
                   "model": self.model_name,
                   "prompt": prompt,
                   "stream": false
               }))
               .send()
               .await?;
           // Parse real response
       }
   }
   ```

2. **Fix ExecutionEngine::execute_task**

   ```rust
   // src/orchestrator/mod.rs - line 1240
   async fn execute_task(&self, task: &ExecutableTask, ...) -> Result<AgentOutput> {
       // Get actual agent from registry
       let agent = self.agent_registry.get(&task.agent_name)
           .ok_or_else(|| anyhow::anyhow!("Agent not found: {}", task.agent_name))?;

       // Get real model client
       let model_client = self.get_model_client(&task.model_assignment.model_selection.model_name)?;

       // Build task input from dependencies
       let task_input = self.build_task_input(task, previous_results)?;

       // REAL EXECUTION
       let result = agent.execute(task_input, context, model_client).await?;

       Ok(result)
   }
   ```

3. **Complete CodeAnalyzerAgent Implementation**
   ```rust
   // src/agents/code_analyzer.rs - enhance existing execute method
   async fn execute(&self, input: AgentInput, context: &FlowContext, model_client: Arc<dyn ModelClient>) -> Result<AgentOutput> {
       let code = input.get_field::<String>("code")?;
       let language = input.get_field::<String>("language")?;

       // Real code analysis using language parser
       let structure = self.analyze_code_structure(&code, &language).await?;

       // Real AI analysis using actual model
       let semantic_analysis = self.ai_semantic_analysis(&code, &structure, model_client, context).await?;

       // Return real analysis results
       Ok(AgentOutput::new()
           .with_field("analysis", semantic_analysis)
           .with_field("structure", structure)
           .with_field("language", language))
   }
   ```

### Phase 2: Model Router Enhancement

**Goal**: Intelligent model selection based on real capabilities

1. **Real Model Capability Detection**

   ```rust
   // Query actual models for their capabilities
   async fn detect_model_capabilities(&self, model_name: &str) -> Result<Vec<ModelCapability>> {
       match model_name {
           "codellama:13b" => {
               // Test model with capability-specific prompts
               self.test_code_generation_capability().await?;
               Ok(vec![ModelCapability::CodeGeneration, ModelCapability::CodeUnderstanding])
           }
           // ... test other models
       }
   }
   ```

2. **Performance-Based Selection**
   ```rust
   // Track real performance metrics
   pub struct ModelPerformanceTracker {
       response_times: HashMap<String, VecDeque<Duration>>,
       success_rates: HashMap<String, (u32, u32)>, // (successes, total)
       token_costs: HashMap<String, VecDeque<f64>>,
   }
   ```

### Phase 3: Agent Ecosystem Expansion

**Goal**: Add specialized agents for different tasks

1. **DocumentationAgent**

   ```rust
   pub struct DocumentationAgent {
       style_templates: HashMap<DocumentationStyle, String>,
       audience_adapters: HashMap<SkillLevel, StyleAdapter>,
   }
   ```

2. **SecurityAuditAgent**

   ```rust
   pub struct SecurityAuditAgent {
       vulnerability_patterns: HashMap<String, Vec<SecurityPattern>>,
       cwe_mapping: HashMap<String, Vec<String>>,
   }
   ```

3. **TestGeneratorAgent**
   ```rust
   pub struct TestGeneratorAgent {
       test_frameworks: HashMap<String, TestFramework>,
       coverage_analyzers: HashMap<String, CoverageAnalyzer>,
   }
   ```

### Phase 4: Error Handling & Reliability

**Goal**: Production-ready error handling and retry logic

1. **Retry Mechanisms**

   ```rust
   pub struct RetryPolicy {
       max_attempts: u32,
       backoff_strategy: BackoffStrategy,
       retryable_errors: HashSet<ErrorType>,
   }
   ```

2. **Circuit Breaker Pattern**

   ```rust
   pub struct ModelCircuitBreaker {
       failure_threshold: u32,
       timeout_duration: Duration,
       current_state: CircuitState,
   }
   ```

3. **Graceful Degradation**
   ```rust
   // Fallback to simpler models or local analysis when primary models fail
   async fn execute_with_fallback(&self, task: &Task) -> Result<AgentOutput> {
       match self.primary_execution(task).await {
           Ok(result) => Ok(result),
           Err(_) => self.fallback_execution(task).await,
       }
   }
   ```

## 🧪 Testing Strategy

### Integration Tests (Priority 1)

```rust
#[tokio::test]
async fn test_real_code_analysis_flow() {
    let ollama_client = OllamaClient::new("http://localhost:11434", "codellama:13b");
    let mut registry = AgentRegistry::new();
    registry.register(CodeAnalyzerAgent::new());

    let orchestrator = FlowOrchestrator::new(Arc::new(registry), ModelRouter::new());

    let request = FlowRequest {
        description: "Analyze this Rust function".to_string(),
        input_data: HashMap::from([
            ("code".to_string(), json!("fn add(a: i32, b: i32) -> i32 { a + b }")),
            ("language".to_string(), json!("rust")),
        ]),
        requirements: FlowRequirements::default(),
        intent: Some(RequestIntent::CodeAnalysis {
            depth: AnalysisDepth::Standard,
            focus_areas: vec![]
        }),
        user_context: None,
        project_context: None,
    };

    let result = orchestrator.execute_flow(request).await.unwrap();
    assert!(result.success);
    assert!(!result.results.is_empty());
}
```

### Unit Tests for Each Component

```rust
#[tokio::test]
async fn test_ollama_client_connection() {
    let client = OllamaClient::new("http://localhost:11434", "codellama:13b");
    let response = client.generate("Explain this code: fn main() {}").await;
    assert!(response.is_ok());
}

#[test]
fn test_code_analyzer_rust_parsing() {
    let analyzer = CodeAnalyzerAgent::new();
    let structure = analyzer.analyze_code_structure("fn test() {}", "rust").await.unwrap();
    assert_eq!(structure.language, "rust");
    assert_eq!(structure.functions.len(), 1);
}
```

## 📋 Immediate Next Steps

1. **Create Ollama client** (`src/model_clients/ollama_client.rs`)
2. **Fix ExecutionEngine::execute_task** to call real agents
3. **Test with one simple code analysis task**
4. **Verify error handling works correctly**
5. **Add integration test to validate end-to-end flow**

## 🎯 Success Criteria

**Phase 1 Complete When**:

- ✅ Can analyze a simple Rust function using CodeAnalyzerAgent
- ✅ Real HTTP calls to Ollama API succeed
- ✅ Pipeline execution works without mock data
- ✅ Errors are properly propagated and handled
- ✅ Integration test passes consistently

**Future Phases**:

- Multiple agents working together in complex workflows
- Intelligent model routing based on performance data
- Production-ready error handling and monitoring
- Real-time cost and performance optimization

This roadmap focuses on **working implementation over architectural sophistication**, ensuring each phase delivers real functionality that can be tested and validated.
