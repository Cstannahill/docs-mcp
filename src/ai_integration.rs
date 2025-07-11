use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn, debug};
use chrono::Utc;

use crate::database::{Database, DocumentPage, DocType, SearchQuery, SearchFilters, RankingPreferences, SearchResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIContext {
    pub current_language: Option<String>,
    pub project_type: Option<String>,
    pub current_file_path: Option<String>,
    pub recent_queries: Vec<String>,
    pub user_skill_level: SkillLevel,
    pub preferred_explanation_style: ExplanationStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SkillLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationStyle {
    Concise,      // Brief, direct answers
    Detailed,     // Comprehensive explanations
    ExampleFocused, // Prioritize code examples
    Conceptual,   // Focus on understanding concepts
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedResponse {
    pub primary_content: DocumentPage,
    pub related_examples: Vec<CodeExample>,
    pub conceptual_context: Vec<ConceptLink>,
    pub difficulty_notes: Option<String>,
    pub common_pitfalls: Vec<String>,
    pub best_practices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    pub title: String,
    pub code: String,
    pub language: String,
    pub explanation: String,
    pub complexity_level: SkillLevel,
    pub source_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptLink {
    pub concept: String,
    pub related_page_id: String,
    pub relationship_type: RelationshipType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    Prerequisite,
    Alternative,
    Extension,
    Comparison,
    Implementation,
}

pub struct AIIntegrationEngine {
    db: Database,
    context_weights: HashMap<String, f32>,
    intent_patterns: HashMap<String, QueryIntent>,
}

impl AIIntegrationEngine {
    pub fn new(db: Database) -> Self {
        let mut context_weights = HashMap::new();
        context_weights.insert("language_match".to_string(), 2.0);
        context_weights.insert("project_type_match".to_string(), 1.5);
        context_weights.insert("recent_query_similarity".to_string(), 1.2);
        context_weights.insert("skill_level_appropriate".to_string(), 1.8);
        
        let mut intent_patterns = HashMap::new();
        intent_patterns.insert("how to".to_string(), QueryIntent::Learning);
        intent_patterns.insert("tutorial".to_string(), QueryIntent::Learning);
        intent_patterns.insert("guide".to_string(), QueryIntent::Learning);
        intent_patterns.insert("example".to_string(), QueryIntent::ProblemSolving);
        intent_patterns.insert("error".to_string(), QueryIntent::Debugging);
        intent_patterns.insert("debug".to_string(), QueryIntent::Debugging);
        intent_patterns.insert("compare".to_string(), QueryIntent::Comparison);
        intent_patterns.insert("vs".to_string(), QueryIntent::Comparison);
        intent_patterns.insert("api".to_string(), QueryIntent::ApiReference);
        intent_patterns.insert("reference".to_string(), QueryIntent::ApiReference);
        intent_patterns.insert("config".to_string(), QueryIntent::Configuration);
        intent_patterns.insert("setup".to_string(), QueryIntent::Configuration);
        
        Self { 
            db, 
            context_weights,
            intent_patterns,
        }
    }

    /// Generate enhanced responses based on AI context
    pub async fn generate_contextual_response(
        &self,
        query: &str,
        context: &AIContext,
    ) -> Result<EnhancedResponse> {
        // 1. Analyze query intent and context
        let query_intent = self.analyze_query_intent(query, context).await?;
        
        // 2. Retrieve relevant documentation with context weighting
        let primary_content = self.get_contextual_content(query, context).await?;
        
        // 3. Find related examples based on skill level and style
        let related_examples = self.get_relevant_examples(
            &primary_content, 
            context.user_skill_level.clone(),
            context.preferred_explanation_style.clone()
        ).await?;
        
        // 4. Build conceptual context map
        let conceptual_context = self.build_concept_map(&primary_content).await?;
        
        // 5. Generate adaptive explanations
        let (difficulty_notes, common_pitfalls, best_practices) = 
            self.generate_adaptive_guidance(&primary_content, context).await?;

        Ok(EnhancedResponse {
            primary_content,
            related_examples,
            conceptual_context,
            difficulty_notes,
            common_pitfalls,
            best_practices,
        })
    }

    async fn analyze_query_intent(&self, query: &str, context: &AIContext) -> Result<QueryIntent> {
        debug!("Analyzing query intent for: {}", query);
        
        let query_lower = query.to_lowercase();
        
        // Check for explicit intent patterns
        for (pattern, intent) in &self.intent_patterns {
            if query_lower.contains(pattern) {
                info!("Detected intent {:?} from pattern '{}'", intent, pattern);
                return Ok(intent.clone());
            }
        }
        
        // Contextual intent analysis
        if let Some(file_path) = &context.current_file_path {
            if file_path.ends_with(".test.js") || file_path.ends_with(".spec.ts") {
                return Ok(QueryIntent::Debugging);
            }
        }
        
        // Check recent queries for patterns
        for recent_query in &context.recent_queries {
            if recent_query.to_lowercase().contains("error") {
                return Ok(QueryIntent::Debugging);
            }
        }
        
        // Default to learning for beginners, API reference for experts
        match context.user_skill_level {
            SkillLevel::Beginner => Ok(QueryIntent::Learning),
            SkillLevel::Expert => Ok(QueryIntent::ApiReference),
            _ => Ok(QueryIntent::ProblemSolving),
        }
    }

    async fn get_contextual_content(&self, query: &str, context: &AIContext) -> Result<DocumentPage> {
        debug!("Getting contextual content for query: {}", query);
        
        // Build context-aware search filters
        let mut filters = SearchFilters {
            doc_types: None,
            content_types: None,
            language: context.current_language.clone(),
            difficulty_level: Some(self.skill_level_to_string(&context.user_skill_level)),
            last_updated_after: None,
        };
        
        // Filter by project type and current language
        if let Some(lang) = &context.current_language {
            filters.doc_types = Some(self.language_to_doc_types(lang));
        }
        
        // Adjust content type based on explanation style
        filters.content_types = Some(match context.preferred_explanation_style {
            ExplanationStyle::ExampleFocused => vec!["example".to_string(), "tutorial".to_string()],
            ExplanationStyle::Detailed => vec!["guide".to_string(), "tutorial".to_string()],
            ExplanationStyle::Concise => vec!["api".to_string(), "reference".to_string()],
            ExplanationStyle::Conceptual => vec!["guide".to_string(), "tutorial".to_string()],
        });
        
        // Build ranking preferences based on context
        let ranking_preferences = RankingPreferences {
            prioritize_recent: true,
            prioritize_official: context.user_skill_level == SkillLevel::Beginner,
            prioritize_examples: matches!(context.preferred_explanation_style, ExplanationStyle::ExampleFocused),
            context_similarity_weight: self.calculate_context_weight(context),
        };
        
        let search_query = SearchQuery {
            query: query.to_string(),
            filters,
            ranking_preferences,
        };
        
        // Perform enhanced search with context weighting
        let search_results = self.db.enhanced_search(&search_query).await?;
        
        if let Some(result) = search_results.first() {
            info!("Found contextual content: {} (score: {:.2})", result.page.title, result.relevance_score);
            Ok(result.page.clone())
        } else {
            // Fallback to basic search if no contextual results
            warn!("No contextual results found, falling back to basic search");
            let basic_results = self.db.search_documents(query, None).await?;
            basic_results
                .first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No content found for query: {}", query))
        }
    }
    
    fn skill_level_to_string(&self, skill_level: &SkillLevel) -> String {
        match skill_level {
            SkillLevel::Beginner => "beginner".to_string(),
            SkillLevel::Intermediate => "intermediate".to_string(),
            SkillLevel::Advanced => "advanced".to_string(),
            SkillLevel::Expert => "expert".to_string(),
        }
    }
    
    fn language_to_doc_types(&self, language: &str) -> Vec<DocType> {
        match language.to_lowercase().as_str() {
            "rust" => vec![DocType::Rust, DocType::Tauri],
            "javascript" | "js" => vec![DocType::React, DocType::TypeScript],
            "typescript" | "ts" => vec![DocType::TypeScript, DocType::React],
            "python" | "py" => vec![DocType::Python],
            "css" => vec![DocType::Tailwind, DocType::Shadcn],
            _ => vec![DocType::React, DocType::TypeScript], // Default fallback
        }
    }
    
    fn calculate_context_weight(&self, context: &AIContext) -> f32 {
        let mut weight = 1.0;
        
        if context.current_language.is_some() {
            weight += self.context_weights.get("language_match").unwrap_or(&0.0);
        }
        
        if context.project_type.is_some() {
            weight += self.context_weights.get("project_type_match").unwrap_or(&0.0);
        }
        
        if !context.recent_queries.is_empty() {
            weight += self.context_weights.get("recent_query_similarity").unwrap_or(&0.0);
        }
        
        weight += self.context_weights.get("skill_level_appropriate").unwrap_or(&0.0);
        
        weight
    }

    async fn get_relevant_examples(
        &self,
        page: &DocumentPage,
        skill_level: SkillLevel,
        style: ExplanationStyle,
    ) -> Result<Vec<CodeExample>> {
        debug!("Finding relevant examples for page: {}", page.title);
        
        let mut examples = Vec::new();
        
        // Extract code blocks from the content
        let code_blocks = self.extract_code_blocks(&page.content, &page.markdown_content);
        
        for (idx, code_block) in code_blocks.iter().enumerate() {
            let complexity = self.assess_code_complexity(&code_block.code);
            
            // Filter by skill level appropriateness
            let complexity_match = match (&skill_level, &complexity) {
                (SkillLevel::Beginner, SkillLevel::Beginner | SkillLevel::Intermediate) => true,
                (SkillLevel::Intermediate, SkillLevel::Beginner | SkillLevel::Intermediate | SkillLevel::Advanced) => true,
                (SkillLevel::Advanced, _) => true,
                (SkillLevel::Expert, _) => true,
                _ => false,
            };
            
            if complexity_match {
                let explanation = self.generate_code_explanation(&code_block.code, &style);
                
                examples.push(CodeExample {
                    title: format!("Example {}: {}", idx + 1, self.infer_code_purpose(&code_block.code)),
                    code: code_block.code.clone(),
                    language: code_block.language.clone(),
                    explanation,
                    complexity_level: complexity,
                    source_url: page.url.clone(),
                });
            }
        }
        
        // Limit examples based on style preference
        let limit = match style {
            ExplanationStyle::Concise => 2,
            ExplanationStyle::ExampleFocused => 5,
            _ => 3,
        };
        
        examples.truncate(limit);
        info!("Found {} relevant examples for skill level {:?}", examples.len(), skill_level);
        Ok(examples)
    }
    
    fn extract_code_blocks(&self, content: &str, markdown_content: &str) -> Vec<ExtractedCodeBlock> {
        let mut blocks = Vec::new();
        
        // Extract from markdown first (more structured)
        if !markdown_content.is_empty() {
            blocks.extend(self.extract_from_markdown(markdown_content));
        }
        
        // Fallback to HTML content extraction
        if blocks.is_empty() {
            blocks.extend(self.extract_from_html(content));
        }
        
        blocks
    }
    
    fn extract_from_markdown(&self, markdown: &str) -> Vec<ExtractedCodeBlock> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = markdown.lines().collect();
        let mut i = 0;
        
        while i < lines.len() {
            let line = lines[i].trim();
            if line.starts_with("```") {
                let language = line[3..].trim().to_string();
                let language = if language.is_empty() { "text".to_string() } else { language };
                
                i += 1;
                let mut code_lines = Vec::new();
                
                while i < lines.len() && !lines[i].trim().starts_with("```") {
                    code_lines.push(lines[i]);
                    i += 1;
                }
                
                if !code_lines.is_empty() {
                    blocks.push(ExtractedCodeBlock {
                        code: code_lines.join("\n"),
                        language,
                    });
                }
            }
            i += 1;
        }
        
        blocks
    }
    
    fn extract_from_html(&self, html: &str) -> Vec<ExtractedCodeBlock> {
        let mut blocks = Vec::new();
        
        // Simple regex-based extraction for code elements
        if html.contains("<code>") || html.contains("<pre>") {
            // This is a simplified implementation
            // In a real scenario, you'd use a proper HTML parser
            let code_pattern = regex::Regex::new(r"<(?:code|pre)[^>]*>(.*?)</(?:code|pre)>").unwrap();
            
            for captures in code_pattern.captures_iter(html) {
                if let Some(code_match) = captures.get(1) {
                    let code = html_escape::decode_html_entities(code_match.as_str()).to_string();
                    if code.len() > 10 && code.lines().count() > 1 {
                        blocks.push(ExtractedCodeBlock {
                            code,
                            language: "text".to_string(),
                        });
                    }
                }
            }
        }
        
        blocks
    }
    
    fn assess_code_complexity(&self, code: &str) -> SkillLevel {
        let line_count = code.lines().count();
        let has_advanced_patterns = code.contains("async") || 
                                   code.contains("await") || 
                                   code.contains("generic") ||
                                   code.contains("trait") ||
                                   code.contains("impl") ||
                                   code.contains("macro");
        
        let has_intermediate_patterns = code.contains("function") ||
                                       code.contains("class") ||
                                       code.contains("interface") ||
                                       code.contains("struct");
        
        match (line_count, has_advanced_patterns, has_intermediate_patterns) {
            (n, true, _) if n > 20 => SkillLevel::Expert,
            (n, true, _) if n > 10 => SkillLevel::Advanced,
            (n, _, true) if n > 15 => SkillLevel::Advanced,
            (n, _, true) if n > 5 => SkillLevel::Intermediate,
            (n, _, _) if n > 10 => SkillLevel::Intermediate,
            _ => SkillLevel::Beginner,
        }
    }
    
    fn generate_code_explanation(&self, code: &str, style: &ExplanationStyle) -> String {
        match style {
            ExplanationStyle::Concise => {
                format!("Code snippet demonstrating key concepts.")
            },
            ExplanationStyle::Detailed => {
                format!("This code example shows the implementation details and explains each step of the process. The code demonstrates best practices and common patterns used in this context.")
            },
            ExplanationStyle::ExampleFocused => {
                format!("Working example that you can copy and modify for your use case. Focus on the practical implementation details.")
            },
            ExplanationStyle::Conceptual => {
                format!("This example illustrates the underlying concepts and principles. Understanding this pattern will help you apply similar solutions in different contexts.")
            },
        }
    }
    
    fn infer_code_purpose(&self, code: &str) -> String {
        if code.contains("test") || code.contains("assert") || code.contains("expect") {
            return "Testing".to_string();
        }
        if code.contains("config") || code.contains("setup") || code.contains("init") {
            return "Configuration".to_string();
        }
        if code.contains("async") || code.contains("await") || code.contains("fetch") {
            return "Async Operation".to_string();
        }
        if code.contains("component") || code.contains("render") || code.contains("jsx") {
            return "Component".to_string();
        }
        if code.contains("function") || code.contains("fn ") {
            return "Function Implementation".to_string();
        }
        
        "Code Example".to_string()
    }

    async fn build_concept_map(&self, page: &DocumentPage) -> Result<Vec<ConceptLink>> {
        debug!("Building concept map for page: {}", page.title);
        let mut concept_links = Vec::new();
        
        // Extract key concepts from the page
        let concepts = self.extract_key_concepts(page);
        
        for concept in concepts {
            // Find related pages for each concept
            let related_pages = self.find_related_concept_pages(&concept, page).await?;
            
            for (related_page_id, relationship_type) in related_pages {
                concept_links.push(ConceptLink {
                    concept: concept.clone(),
                    related_page_id,
                    relationship_type,
                });
            }
        }
        
        info!("Built concept map with {} links", concept_links.len());
        Ok(concept_links)
    }
    
    fn extract_key_concepts(&self, page: &DocumentPage) -> Vec<String> {
        let mut concepts = Vec::new();
        
        // Extract concepts from title
        let title_words: Vec<&str> = page.title.split_whitespace().collect();
        for word in title_words {
            if word.len() > 3 && !word.chars().all(|c| c.is_ascii_lowercase()) {
                concepts.push(word.to_string());
            }
        }
        
        // Look for common programming concepts in content
        let programming_concepts = vec![
            "async", "await", "promise", "callback", "closure", "function",
            "class", "interface", "struct", "enum", "trait", "impl",
            "component", "hook", "state", "props", "context", "redux",
            "router", "middleware", "handler", "controller", "service",
            "api", "rest", "graphql", "database", "query", "mutation",
            "test", "mock", "stub", "assertion", "validation", "error",
            "performance", "optimization", "cache", "memory", "thread",
        ];
        
        let content_lower = page.content.to_lowercase();
        for concept in programming_concepts {
            if content_lower.contains(concept) {
                concepts.push(concept.to_string());
            }
        }
        
        // Remove duplicates and limit
        concepts.sort();
        concepts.dedup();
        concepts.truncate(10);
        
        concepts
    }
    
    async fn find_related_concept_pages(&self, concept: &str, current_page: &DocumentPage) -> Result<Vec<(String, RelationshipType)>> {
        let mut related_pages = Vec::new();
        
        // Search for pages containing this concept
        let search_results = self.db.search_documents(concept, None).await?;
        
        for page in search_results.iter().take(5) {
            if page.id == current_page.id {
                continue; // Skip self
            }
            
            let relationship_type = self.determine_relationship_type(concept, current_page, page);
            related_pages.push((page.id.clone(), relationship_type));
        }
        
        Ok(related_pages)
    }
    
    fn determine_relationship_type(&self, _concept: &str, current_page: &DocumentPage, related_page: &DocumentPage) -> RelationshipType {
        // Determine relationship based on content analysis
        if current_page.title.to_lowercase().contains("basic") && 
           related_page.title.to_lowercase().contains("advanced") {
            return RelationshipType::Extension;
        }
        
        if current_page.title.to_lowercase().contains("guide") && 
           related_page.title.to_lowercase().contains("tutorial") {
            return RelationshipType::Prerequisite;
        }
        
        if (current_page.title.contains("React") && related_page.title.contains("Vue")) ||
           (current_page.title.contains("Rust") && related_page.title.contains("Go")) {
            return RelationshipType::Alternative;
        }
        
        if current_page.content.len() < related_page.content.len() {
            RelationshipType::Extension
        } else {
            RelationshipType::Implementation
        }
    }

    async fn generate_adaptive_guidance(
        &self,
        page: &DocumentPage,
        context: &AIContext,
    ) -> Result<(Option<String>, Vec<String>, Vec<String>)> {
        debug!("Generating adaptive guidance for page: {}", page.title);
        
        // Generate difficulty notes based on skill level
        let difficulty_notes = self.generate_difficulty_notes(page, &context.user_skill_level);
        
        // Generate common pitfalls
        let common_pitfalls = self.generate_common_pitfalls(page, context);
        
        // Generate best practices
        let best_practices = self.generate_best_practices(page, context);
        
        info!("Generated adaptive guidance: {} pitfalls, {} best practices", 
              common_pitfalls.len(), best_practices.len());
        
        Ok((difficulty_notes, common_pitfalls, best_practices))
    }
    
    fn generate_difficulty_notes(&self, page: &DocumentPage, skill_level: &SkillLevel) -> Option<String> {
        let content_lower = page.content.to_lowercase();
        let has_advanced_concepts = content_lower.contains("async") || 
                                   content_lower.contains("generic") ||
                                   content_lower.contains("lifetime") ||
                                   content_lower.contains("closure");
        
        let has_intermediate_concepts = content_lower.contains("class") ||
                                       content_lower.contains("interface") ||
                                       content_lower.contains("inheritance");
        
        match (skill_level, has_advanced_concepts, has_intermediate_concepts) {
            (SkillLevel::Beginner, true, _) => {
                Some("⚠️ This topic involves advanced concepts. Consider reviewing basic concepts first, then return to this when you're more comfortable with the fundamentals.".to_string())
            },
            (SkillLevel::Beginner, _, true) => {
                Some("💡 This topic builds on intermediate concepts. Make sure you understand the basics before diving deep.".to_string())
            },
            (SkillLevel::Intermediate, true, _) => {
                Some("🎯 This is an advanced topic. Take your time to understand each concept thoroughly.".to_string())
            },
            (SkillLevel::Expert, false, false) => {
                Some("📚 This covers foundational concepts you likely already know, but may contain useful implementation details.".to_string())
            },
            _ => None,
        }
    }
    
    fn generate_common_pitfalls(&self, page: &DocumentPage, context: &AIContext) -> Vec<String> {
        let mut pitfalls = Vec::new();
        let content_lower = page.content.to_lowercase();
        
        // React-specific pitfalls
        if content_lower.contains("useeffect") {
            pitfalls.push("🔄 Don't forget to include all dependencies in the useEffect dependency array to avoid stale closures.".to_string());
            pitfalls.push("⚠️ Be careful with infinite re-renders when dependencies change on every render.".to_string());
        }
        
        if content_lower.contains("usestate") {
            pitfalls.push("🔄 Remember that setState is asynchronous - don't rely on the state value immediately after calling setState.".to_string());
        }
        
        // Rust-specific pitfalls
        if content_lower.contains("borrow") || content_lower.contains("ownership") {
            pitfalls.push("🦀 Watch out for borrowing conflicts - you can't have mutable and immutable borrows at the same time.".to_string());
            pitfalls.push("🔒 Be mindful of lifetimes when returning references from functions.".to_string());
        }
        
        if content_lower.contains("async") {
            pitfalls.push("⏰ Async functions must be awaited or they won't execute. Don't forget the .await.".to_string());
            pitfalls.push("🔄 Be careful with shared state in async contexts - consider using Arc<Mutex<T>> for thread safety.".to_string());
        }
        
        // TypeScript-specific pitfalls
        if content_lower.contains("type") || content_lower.contains("interface") {
            pitfalls.push("📝 TypeScript types are erased at runtime - don't rely on them for runtime type checking.".to_string());
            pitfalls.push("🔄 Use type guards when you need to narrow types at runtime.".to_string());
        }
        
        // General programming pitfalls
        if content_lower.contains("api") || content_lower.contains("fetch") {
            pitfalls.push("🌐 Always handle network errors and loading states in API calls.".to_string());
            pitfalls.push("🔒 Don't expose sensitive data like API keys in client-side code.".to_string());
        }
        
        // Skill-level specific warnings
        match context.user_skill_level {
            SkillLevel::Beginner => {
                pitfalls.push("💡 Start with small, working examples before building complex solutions.".to_string());
            },
            SkillLevel::Intermediate => {
                pitfalls.push("🎯 Focus on understanding the underlying patterns, not just memorizing syntax.".to_string());
            },
            _ => {},
        }
        
        pitfalls.truncate(4); // Limit to most important pitfalls
        pitfalls
    }
    
    fn generate_best_practices(&self, page: &DocumentPage, context: &AIContext) -> Vec<String> {
        let mut practices = Vec::new();
        let content_lower = page.content.to_lowercase();
        
        // React best practices
        if content_lower.contains("component") {
            practices.push("📦 Keep components small and focused on a single responsibility.".to_string());
            practices.push("🎯 Use meaningful prop names and consider TypeScript for better type safety.".to_string());
            practices.push("♻️ Extract reusable logic into custom hooks.".to_string());
        }
        
        if content_lower.contains("state") {
            practices.push("🏗️ Lift state up only when multiple components need it - prefer local state when possible.".to_string());
            practices.push("📊 Consider using useReducer for complex state logic.".to_string());
        }
        
        // Rust best practices
        if content_lower.contains("function") || content_lower.contains("fn ") {
            practices.push("📝 Use descriptive function and variable names - Rust encourages clarity over brevity.".to_string());
            practices.push("⚡ Prefer iterators over loops for better performance and readability.".to_string());
        }
        
        if content_lower.contains("error") {
            practices.push("🛡️ Use Result<T, E> for recoverable errors and panic! only for unrecoverable situations.".to_string());
            practices.push("🔧 Implement proper error handling with the ? operator for cleaner code.".to_string());
        }
        
        // General programming best practices
        if content_lower.contains("test") {
            practices.push("✅ Write tests for your public API and edge cases.".to_string());
            practices.push("🔄 Use descriptive test names that explain what's being tested.".to_string());
        }
        
        if content_lower.contains("performance") || content_lower.contains("optimization") {
            practices.push("⚡ Measure before optimizing - premature optimization is the root of all evil.".to_string());
            practices.push("📊 Profile your code to identify actual bottlenecks.".to_string());
        }
        
        // Documentation and maintainability
        practices.push("📖 Write self-documenting code with clear names and structure.".to_string());
        practices.push("🔄 Follow consistent code formatting and style guidelines.".to_string());
        
        // Skill-level specific practices
        match context.user_skill_level {
            SkillLevel::Beginner => {
                practices.push("🏗️ Focus on making it work first, then improve it.".to_string());
                practices.push("📚 Read the official documentation - it's usually the best resource.".to_string());
            },
            SkillLevel::Expert => {
                practices.push("👥 Consider the team and future maintainers when writing code.".to_string());
                practices.push("🏛️ Design for extensibility and maintainability.".to_string());
            },
            _ => {},
        }
        
        practices.truncate(5); // Limit to most important practices
        practices
    }
}

#[derive(Debug, Clone)]
struct ExtractedCodeBlock {
    code: String,
    language: String,
}

#[derive(Debug, Clone)]
pub enum QueryIntent {
    Learning,
    ProblemSolving,
    ApiReference,
    Debugging,
    Comparison,
    Configuration,
}
