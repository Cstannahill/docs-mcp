// src/agents/doc_generator.rs
//! Agent for generating documentation from code or text

use super::*;
use async_trait::async_trait;
use anyhow::Result;

use crate::agents::context_manager::ContextManager;
use std::sync::Arc;

pub struct DocGeneratorAgent {
    pub context_manager: Arc<ContextManager>,
}

#[async_trait]
impl Agent for DocGeneratorAgent {
    fn name(&self) -> &'static str {
        "DocGenerator"
    }
    fn description(&self) -> &'static str {
        "Generates documentation for code, APIs, or text."
    }
    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::DocumentGeneration]
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
        // TODO: Use model client for real doc generation
        let doc = format!("/// Documentation for code: {}", full_code);
        self.context_manager.add_turn(
            &session_id,
            code.clone(),
            "[DocGeneratorAgent output: documentation generated]".to_string(),
            None,
            std::collections::HashMap::new(),
        ).await?;
        Ok(serde_json::json!({ "documentation": doc }))
    }
    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.input.has_field("code")
    }
}
