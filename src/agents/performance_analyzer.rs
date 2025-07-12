// src/agents/performance_analyzer.rs
//! Agent for analyzing code performance

use super::*;
use async_trait::async_trait;
use anyhow::Result;

use crate::agents::context_manager::ContextManager;
use std::sync::Arc;

pub struct PerformanceAnalyzerAgent {
    pub context_manager: Arc<ContextManager>,
}

#[async_trait]
impl Agent for PerformanceAnalyzerAgent {
    fn name(&self) -> &'static str {
        "PerformanceAnalyzer"
    }
    fn description(&self) -> &'static str {
        "Analyzes code for performance bottlenecks and optimization opportunities."
    }
    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::PerformanceAnalysis]
    }
    async fn execute(&self, context: &mut FlowContext) -> Result<serde_json::Value> {
        let session_id = context.session_id.clone();
        let short_term_context = self.context_manager.get_context(&session_id, Some(2048), None).await?;
        let mut context_text = String::new();
        for turn in &short_term_context {
            context_text.push_str(&turn.user_message);
            context_text.push_str(" ");
            context_text.push_str(&turn.assistant_response);
            context_text.push_str(" ");
        }
        let code: String = context.input.get_field("code")?;
        let full_code = format!("{} {}", context_text, code);
        // TODO: Use model client for real performance analysis
        let analysis = format!("No performance issues found in: {}", full_code);
        self.context_manager.add_turn(
            &session_id,
            code.clone(),
            "[PerformanceAnalyzerAgent output: performance analysis generated]".to_string(),
            None,
            std::collections::HashMap::new(),
        ).await?;
        Ok(serde_json::json!({ "performance_analysis": analysis }))
    }
    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.input.has_field("code")
    }
}
