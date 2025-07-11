use anyhow::Result;
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use std::collections::HashMap;
use crate::database::{
    Database, DocumentPage, DocumentQualityMetrics, UserContext, UserInteraction,
    RankingFactors, SearchFilters, DocType, DifficultyLevel
};

pub struct AdvancedRankingEngine {
    db: Database,
    weights: RankingWeights,
}

#[derive(Debug, Clone)]
pub struct RankingWeights {
    pub text_relevance: f32,
    pub semantic_similarity: f32,
    pub quality_score: f32,
    pub freshness_score: f32,
    pub user_engagement: f32,
    pub context_relevance: f32,
}

impl Default for RankingWeights {
    fn default() -> Self {
        Self {
            text_relevance: 0.3,
            semantic_similarity: 0.25,
            quality_score: 0.2,
            freshness_score: 0.1,
            user_engagement: 0.1,
            context_relevance: 0.05,
        }
    }
}

impl AdvancedRankingEngine {
    pub fn new(db: Database) -> Self {
        Self {
            db,
            weights: RankingWeights::default(),
        }
    }

    /// Compute comprehensive ranking score for a document
    pub async fn compute_ranking_score(
        &self,
        page: &DocumentPage,
        query: &str,
        semantic_similarity: f32,
        user_context: &UserContext,
        filters: &SearchFilters,
    ) -> Result<(f32, RankingFactors)> {
        let text_relevance = self.compute_text_relevance(page, query);
        let quality_score = self.get_quality_score(&page.id).await?;
        let freshness_score = self.compute_freshness_score(page);
        let user_engagement = self.compute_user_engagement(&page.id, user_context).await?;
        let context_relevance = self.compute_context_relevance(page, user_context, filters).await?;

        let factors = RankingFactors {
            text_relevance,
            semantic_similarity,
            quality_score,
            freshness_score,
            user_engagement,
            context_relevance,
        };

        let total_score = self.weights.text_relevance * text_relevance
            + self.weights.semantic_similarity * semantic_similarity
            + self.weights.quality_score * quality_score
            + self.weights.freshness_score * freshness_score
            + self.weights.user_engagement * user_engagement
            + self.weights.context_relevance * context_relevance;

        Ok((total_score, factors))
    }

    /// Compute text-based relevance using TF-IDF and keyword matching
    fn compute_text_relevance(&self, page: &DocumentPage, query: &str) -> f32 {
        let query_terms: Vec<&str> = query.to_lowercase()
            .split_whitespace()
            .collect();

        if query_terms.is_empty() {
            return 0.0;
        }

        let content_lower = page.content.to_lowercase();
        let title_lower = page.title.to_lowercase();
        
        let mut relevance_score = 0.0;
        let total_terms = query_terms.len() as f32;

        for term in &query_terms {
            // Title matches are more important
            let title_matches = title_lower.matches(term).count() as f32;
            let content_matches = content_lower.matches(term).count() as f32;
            
            // Compute TF (term frequency)
            let title_tf = title_matches / title_lower.split_whitespace().count().max(1) as f32;
            let content_tf = content_matches / content_lower.split_whitespace().count().max(1) as f32;
            
            // Weight title matches higher
            let term_score = (title_tf * 3.0 + content_tf) / 4.0;
            relevance_score += term_score;
        }

        // Normalize by number of query terms
        let final_score = relevance_score / total_terms;
        
        // Apply boost for exact phrase matches
        if query.len() > 1 && (content_lower.contains(&query.to_lowercase()) || title_lower.contains(&query.to_lowercase())) {
            final_score * 1.5
        } else {
            final_score
        }.min(1.0)
    }

    /// Get quality score from metrics or compute if not available
    async fn get_quality_score(&self, page_id: &str) -> Result<f32> {
        match self.db.get_quality_metrics(page_id).await? {
            Some(metrics) => {
                // Combine multiple quality factors
                let weighted_score = 
                    metrics.freshness_score * 0.3 +
                    metrics.completeness_score * 0.3 +
                    metrics.accuracy_score * 0.2 +
                    (metrics.user_rating_avg / 5.0) * 0.2; // Normalize rating to 0-1
                
                Ok(weighted_score.min(1.0))
            }
            None => {
                // Compute basic quality score if metrics not available
                Ok(0.7) // Default score
            }
        }
    }

    /// Compute freshness score based on last update time
    fn compute_freshness_score(&self, page: &DocumentPage) -> f32 {
        let now = Utc::now();
        let age = now.signed_duration_since(page.last_updated);
        
        // Fresh content (< 30 days) gets full score
        if age < ChronoDuration::days(30) {
            return 1.0;
        }
        
        // Decay function: score decreases exponentially with age
        let days_old = age.num_days() as f32;
        let decay_rate = 0.01; // Adjust to control decay speed
        
        (-(decay_rate * (days_old - 30.0))).exp().max(0.1)
    }

    /// Compute user engagement score based on interaction history
    async fn compute_user_engagement(&self, page_id: &str, user_context: &UserContext) -> Result<f32> {
        // Global engagement metrics
        let view_count = self.db.get_page_view_count(page_id).await?.unwrap_or(0);
        let bookmark_count = self.db.get_page_bookmark_count(page_id).await?.unwrap_or(0);
        
        // Normalize view count (log scale to prevent outliers from dominating)
        let view_score = if view_count > 0 {
            (view_count as f32).ln() / 10.0 // Adjust divisor as needed
        } else {
            0.0
        }.min(1.0);
        
        // Bookmark ratio (high bookmarking indicates quality)
        let bookmark_ratio = if view_count > 0 {
            bookmark_count as f32 / view_count as f32
        } else {
            0.0
        };
        
        // Personal engagement (has user interacted with this page before?)
        let personal_interaction = user_context.recent_interactions
            .iter()
            .any(|interaction| interaction.page_id == page_id);
        
        let personal_boost = if personal_interaction { 0.2 } else { 0.0 };
        
        Ok((view_score * 0.6 + bookmark_ratio * 0.4 + personal_boost).min(1.0))
    }

    /// Compute context relevance based on user's current learning state
    async fn compute_context_relevance(
        &self,
        page: &DocumentPage,
        user_context: &UserContext,
        filters: &SearchFilters,
    ) -> Result<f32> {
        let mut relevance_score = 0.0;
        
        // Doc type preference
        if user_context.preferred_doc_types.contains(&filters.doc_types.get(0).cloned().unwrap_or(DocType::Rust)) {
            relevance_score += 0.3;
        }
        
        // Skill level matching
        if let Some(user_level) = &user_context.skill_level {
            let page_difficulty = self.estimate_page_difficulty(page).await?;
            let level_match = match (user_level, page_difficulty) {
                (DifficultyLevel::Beginner, DifficultyLevel::Beginner) => 0.8,
                (DifficultyLevel::Intermediate, DifficultyLevel::Intermediate) => 0.8,
                (DifficultyLevel::Advanced, DifficultyLevel::Advanced) => 0.8,
                (DifficultyLevel::Beginner, DifficultyLevel::Intermediate) => 0.4,
                (DifficultyLevel::Intermediate, DifficultyLevel::Advanced) => 0.4,
                _ => 0.1,
            };
            relevance_score += level_match * 0.4;
        }
        
        // Learning path context
        if !user_context.current_learning_paths.is_empty() {
            let is_in_learning_path = self.db.is_page_in_learning_paths(
                &page.id,
                &user_context.current_learning_paths
            ).await?;
            
            if is_in_learning_path {
                relevance_score += 0.3;
            }
        }
        
        Ok(relevance_score.min(1.0))
    }

    /// Estimate page difficulty based on content analysis
    async fn estimate_page_difficulty(&self, page: &DocumentPage) -> Result<DifficultyLevel> {
        let content = &page.content.to_lowercase();
        let title = &page.title.to_lowercase();
        
        // Simple heuristics for difficulty estimation
        let beginner_indicators = [
            "introduction", "getting started", "basic", "beginner", "first steps",
            "hello world", "tutorial", "quick start", "overview"
        ];
        
        let advanced_indicators = [
            "advanced", "optimization", "performance", "internals", "deep dive",
            "architecture", "design patterns", "best practices", "production"
        ];
        
        let beginner_score = beginner_indicators.iter()
            .map(|&term| {
                (title.matches(term).count() * 2 + content.matches(term).count()) as f32
            })
            .sum::<f32>();
        
        let advanced_score = advanced_indicators.iter()
            .map(|&term| {
                (title.matches(term).count() * 2 + content.matches(term).count()) as f32
            })
            .sum::<f32>();
        
        if beginner_score > advanced_score * 1.5 {
            Ok(DifficultyLevel::Beginner)
        } else if advanced_score > beginner_score * 1.5 {
            Ok(DifficultyLevel::Advanced)
        } else {
            Ok(DifficultyLevel::Intermediate)
        }
    }

    /// Update ranking weights based on user feedback and analytics
    pub async fn update_weights_from_analytics(&mut self) -> Result<()> {
        // Analyze search analytics to optimize weights
        let analytics = self.db.get_search_analytics_summary().await?;
        
        // Simple adaptive weighting based on click-through rates
        // In a real system, this would use more sophisticated ML techniques
        
        if let Some(ctr_by_factor) = analytics.get("click_through_rates") {
            // Adjust weights based on which factors correlate with clicks
            // This is a simplified example
            self.weights.text_relevance *= 0.95 + 0.1 * ctr_by_factor.get("text").unwrap_or(&0.5);
            self.weights.semantic_similarity *= 0.95 + 0.1 * ctr_by_factor.get("semantic").unwrap_or(&0.5);
            // ... adjust other weights similarly
        }
        
        // Ensure weights still sum to reasonable values
        let total_weight = self.weights.text_relevance + self.weights.semantic_similarity + 
                          self.weights.quality_score + self.weights.freshness_score + 
                          self.weights.user_engagement + self.weights.context_relevance;
        
        if total_weight > 0.0 {
            self.weights.text_relevance /= total_weight;
            self.weights.semantic_similarity /= total_weight;
            self.weights.quality_score /= total_weight;
            self.weights.freshness_score /= total_weight;
            self.weights.user_engagement /= total_weight;
            self.weights.context_relevance /= total_weight;
        }
        
        Ok(())
    }

    /// Compute diversity score to avoid returning too similar results
    pub fn compute_diversity_penalty(&self, results: &[DocumentPage], current_page: &DocumentPage) -> f32 {
        let mut similarity_sum = 0.0;
        let mut count = 0;
        
        for existing_page in results {
            // Simple content similarity based on shared terms
            let shared_terms = self.count_shared_terms(&existing_page.content, &current_page.content);
            let max_terms = existing_page.content.split_whitespace().count()
                .max(current_page.content.split_whitespace().count());
            
            if max_terms > 0 {
                similarity_sum += shared_terms as f32 / max_terms as f32;
                count += 1;
            }
        }
        
        if count > 0 {
            let avg_similarity = similarity_sum / count as f32;
            // Return penalty (higher similarity = higher penalty)
            avg_similarity.powf(2.0) // Square to emphasize high similarity
        } else {
            0.0
        }
    }

    /// Count shared terms between two content pieces
    fn count_shared_terms(&self, content1: &str, content2: &str) -> usize {
        let terms1: std::collections::HashSet<&str> = content1
            .to_lowercase()
            .split_whitespace()
            .collect();
        
        let terms2: std::collections::HashSet<&str> = content2
            .to_lowercase()
            .split_whitespace()
            .collect();
        
        terms1.intersection(&terms2).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_text_relevance_computation() {
        let engine = AdvancedRankingEngine::new(Database::new_in_memory().await.unwrap());
        
        let page = DocumentPage {
            id: "test".to_string(),
            source_id: "test".to_string(),
            title: "Rust Programming Guide".to_string(),
            url: "test".to_string(),
            content: "This is a comprehensive guide to Rust programming language".to_string(),
            markdown_content: "".to_string(),
            last_updated: Utc::now(),
            path: "".to_string(),
            section: None,
        };
        
        let relevance = engine.compute_text_relevance(&page, "Rust programming");
        assert!(relevance > 0.0);
        assert!(relevance <= 1.0);
    }

    #[test]
    fn test_freshness_score() {
        let engine = AdvancedRankingEngine::new(Database::new_in_memory().await.unwrap());
        
        let fresh_page = DocumentPage {
            id: "test".to_string(),
            source_id: "test".to_string(),
            title: "Test".to_string(),
            url: "test".to_string(),
            content: "test".to_string(),
            markdown_content: "".to_string(),
            last_updated: Utc::now(),
            path: "".to_string(),
            section: None,
        };
        
        let old_page = DocumentPage {
            last_updated: Utc::now() - ChronoDuration::days(365),
            ..fresh_page.clone()
        };
        
        let fresh_score = engine.compute_freshness_score(&fresh_page);
        let old_score = engine.compute_freshness_score(&old_page);
        
        assert!(fresh_score > old_score);
        assert_eq!(fresh_score, 1.0); // Fresh content should get full score
    }
}
