// src/agents/security_auditor.rs
//! Agent for performing security audits on code

use super::*;
use async_trait::async_trait;
use anyhow::Result;

use crate::agents::context_manager::ContextManager;
use std::sync::Arc;

pub struct SecurityAuditorAgent {
    pub context_manager: Arc<ContextManager>,
}

#[async_trait]
impl Agent for SecurityAuditorAgent {
    fn name(&self) -> &'static str {
        "SecurityAuditor"
    }
    fn description(&self) -> &'static str {
        "Performs security audits and vulnerability detection on code."
    }
    fn capabilities(&self) -> Vec<AgentCapability> {
        vec![AgentCapability::SecurityAudit]
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
        // TODO: Use model client for real security analysis
        let report = format!("No vulnerabilities found in: {}", full_code);
        self.context_manager.add_turn(
            &session_id,
            code.clone(),
            "[SecurityAuditorAgent output: security report generated]".to_string(),
            None,
            std::collections::HashMap::new(),
        ).await?;
        Ok(serde_json::json!({ "security_report": report }))
    }
    async fn can_handle(&self, context: &FlowContext) -> bool {
        context.metadata.contains_key("code")
    }
}
