use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::database::{
    Database, LearningPath, LearningPathStep, UserLearningProgress, 
    DocumentPage, DocType, DifficultyLevel, ContentSuggestion, 
    SuggestionType, UserContext, DocumentRelationship, RelationshipType
};

pub struct LearningPathEngine {
    db: Database,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveTutorial {
    pub id: String,
    pub title: String,
    pub description: String,
    pub steps: Vec<TutorialStep>,
    pub prerequisites: Vec<String>,
    pub estimated_duration_minutes: i32,
    pub difficulty_level: DifficultyLevel,
    pub doc_type: DocType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialStep {
    pub id: String,
    pub title: String,
    pub content: String,
    pub step_type: TutorialStepType,
    pub page_references: Vec<String>,
    pub interactive_elements: Vec<InteractiveElement>,
    pub validation: Option<StepValidation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TutorialStepType {
    Reading,
    CodeExample,
    Exercise,
    Quiz,
    Project,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    pub element_type: ElementType,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementType {
    CodeBlock,
    LiveDemo,
    Quiz,
    Checklist,
    VideoEmbed,
    ExternalLink,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepValidation {
    pub validation_type: ValidationType,
    pub criteria: Vec<String>,
    pub feedback: HashMap<String, String>, // outcome -> feedback message
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    CodeOutput,
    QuizScore,
    ManualCheck,
    FileExists,
    TimeSpent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRecommendation {
    pub suggested_content: Vec<ContentSuggestion>,
    pub next_steps: Vec<String>,
    pub skill_gaps: Vec<SkillGap>,
    pub adaptive_difficulty: DifficultyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillGap {
    pub topic: String,
    pub current_level: f32,
    pub target_level: f32,
    pub recommended_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLearningProfile {
    pub session_id: String,
    pub skill_assessments: HashMap<String, f32>,
    pub completed_paths: Vec<String>,
    pub current_paths: Vec<String>,
    pub learning_velocity: f32, // pages per hour
    pub preferred_difficulty: DifficultyLevel,
    pub strengths: Vec<String>,
    pub areas_for_improvement: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

impl LearningPathEngine {
    pub fn new(db: Database) -> Self {
        Self { db }
    }

    /// Generate personalized learning paths based on user context and goals
    pub async fn generate_personalized_path(
        &self,
        user_context: &UserContext,
        target_topic: &str,
        difficulty: DifficultyLevel,
    ) -> Result<LearningPath> {
        // Analyze user's current knowledge level
        let current_skills = self.assess_user_skills(user_context).await?;
        
        // Find relevant pages for the target topic
        let relevant_pages = self.db.search_pages_by_topic(target_topic).await?;
        
        // Create learning path with progressive difficulty
        let path_id = format!("path_{}_{}_{}", 
            user_context.session_id, 
            target_topic.replace(" ", "_"),
            chrono::Utc::now().timestamp()
        );
        
        let learning_path = LearningPath {
            id: path_id.clone(),
            title: format!("Personalized {} Learning Path", target_topic),
            description: format!("A customized learning journey for {} tailored to your current skill level", target_topic),
            difficulty_level: difficulty,
            estimated_duration_minutes: self.estimate_path_duration(&relevant_pages).await?,
            doc_type: self.infer_doc_type(target_topic),
            created_by: "ai_generator".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Create ordered steps
        let ordered_pages = self.order_pages_by_difficulty(&relevant_pages, &difficulty).await?;
        let mut steps = Vec::new();
        
        for (index, page) in ordered_pages.iter().enumerate() {
            let step = LearningPathStep {
                id: None,
                path_id: path_id.clone(),
                step_order: index as i32 + 1,
                page_id: page.id.clone(),
                title: page.title.clone(),
                description: Some(self.generate_step_description(page, index).await?),
                is_optional: self.is_step_optional(page, &difficulty).await?,
                estimated_duration_minutes: Some(self.estimate_reading_time(page).await?),
                created_at: Utc::now(),
            };
            steps.push(step);
        }

        // Store the learning path and steps
        self.db.create_learning_path(&learning_path).await?;
        for step in steps {
            self.db.create_learning_path_step(&step).await?;
        }

        Ok(learning_path)
    }

    /// Generate adaptive recommendations based on user progress
    pub async fn generate_recommendations(
        &self,
        user_context: &UserContext,
    ) -> Result<LearningRecommendation> {
        let profile = self.build_user_profile(user_context).await?;
        
        // Analyze current progress
        let current_progress = self.analyze_learning_progress(&profile).await?;
        
        // Identify skill gaps
        let skill_gaps = self.identify_skill_gaps(&profile).await?;
        
        // Generate content suggestions
        let suggested_content = self.generate_content_suggestions(&profile, &skill_gaps).await?;
        
        // Determine next steps
        let next_steps = self.suggest_next_steps(&profile, &current_progress).await?;
        
        // Adapt difficulty based on performance
        let adaptive_difficulty = self.calculate_adaptive_difficulty(&profile).await?;

        Ok(LearningRecommendation {
            suggested_content,
            next_steps,
            skill_gaps,
            adaptive_difficulty,
        })
    }

    /// Create interactive tutorial from documentation
    pub async fn create_interactive_tutorial(
        &self,
        topic: &str,
        pages: Vec<DocumentPage>,
        difficulty: DifficultyLevel,
    ) -> Result<InteractiveTutorial> {
        let tutorial_id = format!("tutorial_{}_{}", 
            topic.replace(" ", "_"), 
            chrono::Utc::now().timestamp()
        );

        let mut steps = Vec::new();
        
        for (index, page) in pages.iter().enumerate() {
            // Generate different step types based on content
            let step_type = self.determine_step_type(page, index).await?;
            
            // Create interactive elements based on content
            let interactive_elements = self.generate_interactive_elements(page, &step_type).await?;
            
            // Add validation if appropriate
            let validation = if matches!(step_type, TutorialStepType::Exercise | TutorialStepType::Quiz) {
                Some(self.create_step_validation(&step_type, page).await?)
            } else {
                None
            };

            let step = TutorialStep {
                id: format!("step_{}_{}", tutorial_id, index),
                title: page.title.clone(),
                content: self.adapt_content_for_tutorial(page, &difficulty).await?,
                step_type,
                page_references: vec![page.id.clone()],
                interactive_elements,
                validation,
            };
            
            steps.push(step);
        }

        let tutorial = InteractiveTutorial {
            id: tutorial_id,
            title: format!("Interactive {} Tutorial", topic),
            description: format!("Learn {} through hands-on examples and exercises", topic),
            steps,
            prerequisites: self.identify_prerequisites(&pages).await?,
            estimated_duration_minutes: self.estimate_tutorial_duration(&pages).await?,
            difficulty_level: difficulty,
            doc_type: self.infer_doc_type(topic),
        };

        Ok(tutorial)
    }

    /// Track user progress and update learning profile
    pub async fn track_progress(
        &self,
        session_id: &str,
        path_id: &str,
        step_id: i64,
        completed: bool,
        time_spent_seconds: Option<i32>,
        notes: Option<String>,
    ) -> Result<()> {
        let progress = UserLearningProgress {
            id: None,
            session_id: session_id.to_string(),
            path_id: path_id.to_string(),
            step_id,
            completed,
            completion_time: if completed { Some(Utc::now()) } else { None },
            notes,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.db.update_learning_progress(&progress).await?;
        
        // Update user profile based on progress
        self.update_user_profile_from_progress(session_id, &progress).await?;
        
        // Generate new recommendations if step completed
        if completed {
            self.trigger_recommendation_update(session_id).await?;
        }

        Ok(())
    }

    /// Build comprehensive user learning profile
    async fn build_user_profile(&self, user_context: &UserContext) -> Result<UserLearningProfile> {
        let interactions = &user_context.recent_interactions;
        
        // Calculate learning velocity
        let learning_velocity = self.calculate_learning_velocity(interactions).await?;
        
        // Assess skills based on interaction history
        let skill_assessments = self.assess_skills_from_interactions(interactions).await?;
        
        // Get completed and current paths
        let completed_paths = self.db.get_completed_learning_paths(&user_context.session_id).await?;
        let current_paths = user_context.current_learning_paths.clone();
        
        // Identify strengths and areas for improvement
        let (strengths, areas_for_improvement) = self.analyze_performance_patterns(interactions).await?;

        Ok(UserLearningProfile {
            session_id: user_context.session_id.clone(),
            skill_assessments,
            completed_paths,
            current_paths,
            learning_velocity,
            preferred_difficulty: user_context.skill_level.clone().unwrap_or(DifficultyLevel::Intermediate),
            strengths,
            areas_for_improvement,
            last_updated: Utc::now(),
        })
    }

    /// Assess user skills from interaction patterns
    async fn assess_user_skills(&self, user_context: &UserContext) -> Result<HashMap<String, f32>> {
        let mut skills = HashMap::new();
        
        for interaction in &user_context.recent_interactions {
            if let Some(page) = self.db.get_page(&interaction.page_id).await? {
                // Extract topics from page content
                let topics = self.extract_topics_from_page(&page).await?;
                
                for topic in topics {
                    let current_level = skills.get(&topic).unwrap_or(&0.0);
                    
                    // Increase skill level based on interaction type and duration
                    let skill_increment = match interaction.interaction_type {
                        crate::database::InteractionType::View => {
                            if let Some(duration) = interaction.duration_seconds {
                                (duration as f32 / 300.0).min(0.1) // Max 0.1 for 5+ minutes
                            } else {
                                0.02
                            }
                        }
                        crate::database::InteractionType::Bookmark => 0.15,
                        crate::database::InteractionType::Copy => 0.05,
                        crate::database::InteractionType::Rate => 0.1,
                        _ => 0.01,
                    };
                    
                    skills.insert(topic, (current_level + skill_increment).min(1.0));
                }
            }
        }
        
        Ok(skills)
    }

    /// Generate content suggestions based on learning profile
    async fn generate_content_suggestions(
        &self,
        profile: &UserLearningProfile,
        skill_gaps: &[SkillGap],
    ) -> Result<Vec<ContentSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Suggest content for skill gaps
        for gap in skill_gaps {
            let relevant_pages = self.db.search_pages_by_topic(&gap.topic).await?;
            
            for page in relevant_pages.into_iter().take(3) {
                let suggestion = ContentSuggestion {
                    id: None,
                    session_id: profile.session_id.clone(),
                    suggested_page_id: page.id,
                    suggestion_type: SuggestionType::Prerequisite,
                    confidence_score: gap.target_level - gap.current_level,
                    reason: Some(format!("Recommended to improve {} skills", gap.topic)),
                    shown: false,
                    clicked: false,
                    created_at: Utc::now(),
                };
                suggestions.push(suggestion);
            }
        }
        
        Ok(suggestions)
    }

    // Helper methods (simplified implementations)
    
    async fn estimate_path_duration(&self, pages: &[DocumentPage]) -> Result<i32> {
        let total_content_length: usize = pages.iter().map(|p| p.content.len()).sum();
        // Assume 200 words per minute reading speed, average 5 chars per word
        Ok((total_content_length / (200 * 5)) as i32)
    }

    fn infer_doc_type(&self, topic: &str) -> DocType {
        let topic_lower = topic.to_lowercase();
        if topic_lower.contains("rust") {
            DocType::Rust
        } else if topic_lower.contains("typescript") || topic_lower.contains("javascript") {
            DocType::TypeScript
        } else if topic_lower.contains("python") {
            DocType::Python
        } else if topic_lower.contains("react") {
            DocType::React
        } else if topic_lower.contains("tauri") {
            DocType::Tauri
        } else {
            DocType::Rust // Default
        }
    }

    async fn order_pages_by_difficulty(&self, pages: &[DocumentPage], target_difficulty: &DifficultyLevel) -> Result<Vec<DocumentPage>> {
        let mut pages_with_scores: Vec<(DocumentPage, f32)> = Vec::new();
        
        for page in pages {
            let difficulty_score = self.calculate_difficulty_score(page).await?;
            pages_with_scores.push((page.clone(), difficulty_score));
        }
        
        // Sort by difficulty score (ascending for progressive learning)
        pages_with_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(pages_with_scores.into_iter().map(|(page, _)| page).collect())
    }

    async fn calculate_difficulty_score(&self, page: &DocumentPage) -> Result<f32> {
        // Simple heuristic based on content complexity
        let content = &page.content.to_lowercase();
        
        let complex_terms = ["advanced", "optimization", "performance", "architecture"];
        let beginner_terms = ["introduction", "basic", "getting started", "hello world"];
        
        let complex_count = complex_terms.iter()
            .map(|term| content.matches(term).count())
            .sum::<usize>() as f32;
            
        let beginner_count = beginner_terms.iter()
            .map(|term| content.matches(term).count())
            .sum::<usize>() as f32;
        
        // Return score from 0.0 (beginner) to 1.0 (advanced)
        if complex_count + beginner_count == 0.0 {
            0.5 // Default to intermediate
        } else {
            complex_count / (complex_count + beginner_count)
        }
    }

    // Additional helper methods would be implemented here...
    // For brevity, I'm including just the signatures:
    
    async fn generate_step_description(&self, page: &DocumentPage, index: usize) -> Result<String> {
        Ok(format!("Step {}: Learn about {}", index + 1, page.title))
    }
    
    async fn is_step_optional(&self, page: &DocumentPage, difficulty: &DifficultyLevel) -> Result<bool> {
        Ok(false) // Simplified - could analyze content to determine if optional
    }
    
    async fn estimate_reading_time(&self, page: &DocumentPage) -> Result<i32> {
        Ok((page.content.len() / 1000).max(1) as i32) // Rough estimate
    }
    
    async fn analyze_learning_progress(&self, profile: &UserLearningProfile) -> Result<HashMap<String, f32>> {
        Ok(HashMap::new()) // Simplified implementation
    }
    
    async fn identify_skill_gaps(&self, profile: &UserLearningProfile) -> Result<Vec<SkillGap>> {
        Ok(Vec::new()) // Simplified implementation
    }
    
    async fn suggest_next_steps(&self, profile: &UserLearningProfile, progress: &HashMap<String, f32>) -> Result<Vec<String>> {
        Ok(Vec::new()) // Simplified implementation
    }
    
    async fn calculate_adaptive_difficulty(&self, profile: &UserLearningProfile) -> Result<DifficultyLevel> {
        Ok(profile.preferred_difficulty.clone()) // Simplified implementation
    }
    
    async fn determine_step_type(&self, page: &DocumentPage, index: usize) -> Result<TutorialStepType> {
        Ok(TutorialStepType::Reading) // Simplified implementation
    }
    
    async fn generate_interactive_elements(&self, page: &DocumentPage, step_type: &TutorialStepType) -> Result<Vec<InteractiveElement>> {
        Ok(Vec::new()) // Simplified implementation
    }
    
    async fn create_step_validation(&self, step_type: &TutorialStepType, page: &DocumentPage) -> Result<StepValidation> {
        Ok(StepValidation {
            validation_type: ValidationType::ManualCheck,
            criteria: vec!["Read the content".to_string()],
            feedback: HashMap::new(),
        })
    }
    
    async fn adapt_content_for_tutorial(&self, page: &DocumentPage, difficulty: &DifficultyLevel) -> Result<String> {
        Ok(page.content.clone()) // Simplified implementation
    }
    
    async fn identify_prerequisites(&self, pages: &[DocumentPage]) -> Result<Vec<String>> {
        Ok(Vec::new()) // Simplified implementation
    }
    
    async fn estimate_tutorial_duration(&self, pages: &[DocumentPage]) -> Result<i32> {
        Ok(pages.len() as i32 * 10) // 10 minutes per page estimate
    }
    
    async fn update_user_profile_from_progress(&self, session_id: &str, progress: &UserLearningProgress) -> Result<()> {
        Ok(()) // Simplified implementation
    }
    
    async fn trigger_recommendation_update(&self, session_id: &str) -> Result<()> {
        Ok(()) // Simplified implementation
    }
    
    async fn calculate_learning_velocity(&self, interactions: &[crate::database::UserInteraction]) -> Result<f32> {
        Ok(1.0) // Simplified implementation
    }
    
    async fn assess_skills_from_interactions(&self, interactions: &[crate::database::UserInteraction]) -> Result<HashMap<String, f32>> {
        Ok(HashMap::new()) // Simplified implementation
    }
    
    async fn analyze_performance_patterns(&self, interactions: &[crate::database::UserInteraction]) -> Result<(Vec<String>, Vec<String>)> {
        Ok((Vec::new(), Vec::new())) // Simplified implementation
    }
    
    async fn extract_topics_from_page(&self, page: &DocumentPage) -> Result<Vec<String>> {
        // Simple topic extraction based on keywords
        let content = page.content.to_lowercase();
        let mut topics = Vec::new();
        
        // Extract programming language topics
        if content.contains("rust") { topics.push("rust".to_string()); }
        if content.contains("python") { topics.push("python".to_string()); }
        if content.contains("typescript") { topics.push("typescript".to_string()); }
        if content.contains("javascript") { topics.push("javascript".to_string()); }
        
        // Extract concept topics
        if content.contains("function") { topics.push("functions".to_string()); }
        if content.contains("variable") { topics.push("variables".to_string()); }
        if content.contains("loop") { topics.push("loops".to_string()); }
        if content.contains("class") { topics.push("classes".to_string()); }
        
        Ok(topics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_learning_path_creation() {
        // This would test the learning path generation logic
        // Implementation would depend on having a test database setup
    }

    #[test]
    fn test_difficulty_score_calculation() {
        // Test the difficulty scoring algorithm
    }
}
