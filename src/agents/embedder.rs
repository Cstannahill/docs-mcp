// src/agents/embedder.rs
//! Agent for generating embeddings from text or code

use super::*;
use async_trait::async_trait;
use anyhow::Result;

use crate::agents::context_manager::ContextManager;
use std::sync::Arc;

pub struct EmbedderAgent {
    pub context_manager: Arc<ContextManager>,
}

#[async_trait]
impl Agent for EmbedderAgent {
    fn name(&self) -> &'static str {
        "Embedder"
    }
    fn description(&self) -> &'static str {
        "Generates vector embeddings for text or code."
    }
    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::SemanticAnalysis]
    }
    async fn execute(&self, context: &mut FlowContext) -> Result<serde_json::Value> {
        // Session management: get session_id from FlowContext
        let session_id = context.session_id.clone();

        // Fetch relevant context (short-term and long-term)
        let short_term_context = self.context_manager.get_context(&session_id, Some(2048), None).await?;
        let long_term_context = self.context_manager.get_context(&session_id, Some(8192), None).await?;

        // Example: concatenate recent turns for embedding
        let mut context_text = String::new();
        for turn in &short_term_context {
            context_text.push_str(&turn.user_message);
            context_text.push_str(" ");
            context_text.push_str(&turn.assistant_response);
            context_text.push_str(" ");
        }

        let text: String = context.input.get_field("text")?;
        let full_text = format!("{} {}", context_text, text);

        // TODO: Use model client for real embedding generation
        let embedding = vec![0.0; 768]; // Dummy embedding

        // Update context with this turn (simulate agent output)
        self.context_manager.add_turn(
            &session_id,
            text.clone(),
            "[EmbedderAgent output: embedding generated]".to_string(),
            None,
            std::collections::HashMap::new(),
        ).await?;

        Ok(serde_json::json!({ "embedding": embedding }))
    }
    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.input.has_field("text")
    }
}
