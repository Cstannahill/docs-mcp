// src/agents/test_generator.rs
//! Agent for generating tests from code or requirements

use super::*;
use async_trait::async_trait;
use anyhow::Result;

use crate::agents::context_manager::ContextManager;
use std::sync::Arc;

pub struct TestGeneratorAgent {
    pub context_manager: Arc<ContextManager>,
}

#[async_trait]
impl Agent for TestGeneratorAgent {
    fn name(&self) -> &'static str {
        "TestGenerator"
    }
    fn description(&self) -> &'static str {
        "Generates unit and integration tests for code."
    }
    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::TestCreation]
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
        let code: String = context.get_global_context::<String>("code").ok_or_else(|| anyhow::anyhow!("Missing 'code' in global context"))?;
        let full_code = format!("{} {}", context_text, code);
        // TODO: Use model client for real test generation
        let test = format!("#[test]\nfn test_generated() {{ /* test for: {} */ }}", full_code);
        self.context_manager.add_turn(
            &session_id,
            code.clone(),
            "[TestGeneratorAgent output: test generated]".to_string(),
            None,
            std::collections::HashMap::new(),
        ).await?;
        Ok(serde_json::json!({ "test_code": test }))
    }
    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.metadata.contains_key("code")
    }
}
