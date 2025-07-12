use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{SqlitePool, Row};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationSource {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub doc_type: DocType,
    pub last_updated: Option<DateTime<Utc>>,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum DocType {
    Rust,
    Tauri,
    React,
    TypeScript,
    Python,
    Tailwind,
    Shadcn,
}

impl DocType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocType::Rust => "rust",
            DocType::Tauri => "tauri",
            DocType::React => "react",
            DocType::TypeScript => "typescript",
            DocType::Python => "python",
            DocType::Tailwind => "tailwind",
            DocType::Shadcn => "shadcn",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentPage {
    pub id: String,
    pub source_id: String,
    pub title: String,
    pub url: String,
    pub content: String,
    pub markdown_content: String,
    pub last_updated: DateTime<Utc>,
    pub path: String,
    pub section: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub filters: SearchFilters,
    pub ranking_preferences: RankingPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    pub doc_types: Option<Vec<DocType>>,
    pub content_types: Option<Vec<String>>, // "api", "tutorial", "example"
    pub language: Option<String>,
    pub difficulty_level: Option<String>,   // "beginner", "intermediate", "advanced"
    pub last_updated_after: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingPreferences {
    pub prioritize_recent: bool,
    pub prioritize_official: bool,
    pub prioritize_examples: bool,
    pub context_similarity_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub page: DocumentPage,
    pub relevance_score: f32,
    pub matched_snippets: Vec<ContentSnippet>,
    pub related_pages: Vec<String>, // page IDs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSnippet {
    pub content: String,
    pub highlight_ranges: Vec<(usize, usize)>,
    pub snippet_type: SnippetType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnippetType {
    CodeExample,
    Definition,
    Usage,
    Configuration,
}

#[derive(Debug, Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePool::connect(database_url).await?;
        
        // Create tables manually since migrations might have issues
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS documentation_sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                last_updated DATETIME,
                version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS document_pages (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                markdown_content TEXT NOT NULL,
                last_updated DATETIME NOT NULL,
                path TEXT NOT NULL,
                section TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&pool)
        .await?;

        // Create indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_document_pages_source_id ON document_pages(source_id)")
            .execute(&pool)
            .await?;
        
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_document_pages_path ON document_pages(path)")
            .execute(&pool)
            .await?;
        
        Ok(Self { pool })
    }

    /// Create a new in-memory database (for testing)
    pub async fn new_in_memory() -> Result<Self> {
        Self::new(":memory:").await
    }

    pub async fn add_source(&self, source: &DocumentationSource) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO documentation_sources 
            (id, name, base_url, doc_type, last_updated, version)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&source.id)
        .bind(&source.name)
        .bind(&source.base_url)
        .bind(source.doc_type.as_str())
        .bind(&source.last_updated)
        .bind(&source.version)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_sources(&self) -> Result<Vec<DocumentationSource>> {
        let rows = sqlx::query(
            "SELECT id, name, base_url, doc_type, last_updated, version FROM documentation_sources"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut sources = Vec::new();
        for row in rows {
            let doc_type_str: String = row.get("doc_type");
            let doc_type = match doc_type_str.as_str() {
                "rust" => DocType::Rust,
                "tauri" => DocType::Tauri,
                "react" => DocType::React,
                "typescript" => DocType::TypeScript,
                "python" => DocType::Python,
                "tailwind" => DocType::Tailwind,
                "shadcn" => DocType::Shadcn,
                _ => continue,
            };

            sources.push(DocumentationSource {
                id: row.get("id"),
                name: row.get("name"),
                base_url: row.get("base_url"),
                doc_type,
                last_updated: row.get("last_updated"),
                version: row.get("version"),
            });
        }

        Ok(sources)
    }

    pub async fn add_document(&self, doc: &DocumentPage) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO document_pages 
            (id, source_id, title, url, content, markdown_content, last_updated, path, section)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&doc.id)
        .bind(&doc.source_id)
        .bind(&doc.title)
        .bind(&doc.url)
        .bind(&doc.content)
        .bind(&doc.markdown_content)
        .bind(&doc.last_updated)
        .bind(&doc.path)
        .bind(&doc.section)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn search_documents(&self, query: &str, source_type: Option<&str>) -> Result<Vec<DocumentPage>> {
        let (sql, query_args) = if let Some(doc_type) = source_type {
            (
                r#"
                SELECT dp.id, dp.source_id, dp.title, dp.url, dp.content, dp.markdown_content, 
                       dp.last_updated, dp.path, dp.section
                FROM document_pages dp
                JOIN documentation_sources ds ON dp.source_id = ds.id
                WHERE (dp.title LIKE ? OR dp.content LIKE ? OR dp.markdown_content LIKE ?)
                AND ds.doc_type = ?
                ORDER BY 
                    CASE 
                        WHEN dp.title LIKE ? THEN 1
                        WHEN dp.content LIKE ? THEN 2
                        ELSE 3
                    END,
                    dp.title
                LIMIT 50
                "#,
                vec![
                    format!("%{}%", query),
                    format!("%{}%", query),
                    format!("%{}%", query),
                    doc_type.to_string(),
                    format!("%{}%", query),
                    format!("%{}%", query),
                ]
            )
        } else {
            (
                r#"
                SELECT id, source_id, title, url, content, markdown_content, last_updated, path, section
                FROM document_pages
                WHERE title LIKE ? OR content LIKE ? OR markdown_content LIKE ?
                ORDER BY 
                    CASE 
                        WHEN title LIKE ? THEN 1
                        WHEN content LIKE ? THEN 2
                        ELSE 3
                    END,
                    title
                LIMIT 50
                "#,
                vec![
                    format!("%{}%", query),
                    format!("%{}%", query),
                    format!("%{}%", query),
                    format!("%{}%", query),
                    format!("%{}%", query),
                ]
            )
        };

        let mut db_query = sqlx::query(sql);
        for arg in query_args {
            db_query = db_query.bind(arg);
        }

        let rows = db_query.fetch_all(&self.pool).await?;

        let mut documents = Vec::new();
        for row in rows {
            documents.push(DocumentPage {
                id: row.get("id"),
                source_id: row.get("source_id"),
                title: row.get("title"),
                url: row.get("url"),
                content: row.get("content"),
                markdown_content: row.get("markdown_content"),
                last_updated: row.get("last_updated"),
                path: row.get("path"),
                section: row.get("section"),
            });
        }

        Ok(documents)
    }

    pub async fn enhanced_search(&self, search_query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let mut where_conditions = Vec::<String>::new();
        let mut query_args = Vec::new();
        let mut join_sources = false;

        // Base text search
        where_conditions.push("(dp.title LIKE ? OR dp.content LIKE ? OR dp.markdown_content LIKE ?)".to_string());
        let search_pattern = format!("%{}%", search_query.query);
        query_args.push(search_pattern.clone());
        query_args.push(search_pattern.clone());
        query_args.push(search_pattern);

        // Apply filters
        if let Some(ref doc_types) = search_query.filters.doc_types {
            if !doc_types.is_empty() {
                join_sources = true;
                let type_placeholders = doc_types.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let type_condition = format!("ds.doc_type IN ({})", type_placeholders);
                where_conditions.push(type_condition);
                for doc_type in doc_types {
                    query_args.push(doc_type.as_str().to_string());
                }
            }
        }

        if let Some(ref content_types) = search_query.filters.content_types {
            if !content_types.is_empty() {
                // Filter by content type indicators in title/content
                let mut content_conditions = Vec::new();
                for content_type in content_types {
                    match content_type.as_str() {
                        "api" => {
                            content_conditions.push("(dp.title LIKE '%API%' OR dp.title LIKE '%Reference%' OR dp.content LIKE '%function%' OR dp.content LIKE '%method%')");
                        },
                        "tutorial" => {
                            content_conditions.push("(dp.title LIKE '%Tutorial%' OR dp.title LIKE '%Guide%' OR dp.title LIKE '%Getting Started%')");
                        },
                        "example" => {
                            content_conditions.push("(dp.content LIKE '%example%' OR dp.content LIKE '%```%' OR dp.markdown_content LIKE '%```%')");
                        },
                        _ => {}
                    }
                }
                if !content_conditions.is_empty() {
                    let content_condition = format!("({})", content_conditions.join(" OR "));
                    where_conditions.push(content_condition);
                }
            }
        }

        if let Some(ref last_updated_after) = search_query.filters.last_updated_after {
            where_conditions.push("dp.last_updated >= ?".to_string());
            query_args.push(last_updated_after.format("%Y-%m-%d %H:%M:%S").to_string());
        }

        // Build the query
        let from_clause = if join_sources {
            "document_pages dp JOIN documentation_sources ds ON dp.source_id = ds.id"
        } else {
            "document_pages dp"
        };

        let where_clause = if where_conditions.is_empty() {
            "".to_string()
        } else {
            format!("WHERE {}", where_conditions.join(" AND "))
        };

        // Enhanced ranking based on preferences
        let mut order_conditions = Vec::new();
        
        // Base relevance scoring
        order_conditions.push("CASE WHEN dp.title LIKE ? THEN 1 WHEN dp.content LIKE ? THEN 2 ELSE 3 END");
        query_args.push(format!("%{}%", search_query.query));
        query_args.push(format!("%{}%", search_query.query));

        // Recency boost
        if search_query.ranking_preferences.prioritize_recent {
            order_conditions.push("dp.last_updated DESC");
        }

        // Example prioritization
        if search_query.ranking_preferences.prioritize_examples {
            order_conditions.push("CASE WHEN dp.content LIKE '%```%' THEN 1 ELSE 2 END");
        }

        let order_clause = format!("ORDER BY {}", order_conditions.join(", "));

        let sql = format!(
            "SELECT dp.id, dp.source_id, dp.title, dp.url, dp.content, dp.markdown_content, dp.last_updated, dp.path, dp.section {} {} {} LIMIT 50",
            format!("FROM {}", from_clause),
            where_clause,
            order_clause
        );

        let mut db_query = sqlx::query(&sql);
        for arg in query_args {
            db_query = db_query.bind(arg);
        }

        let rows = db_query.fetch_all(&self.pool).await?;

        let mut results = Vec::new();
        for row in rows {
            let page = DocumentPage {
                id: row.get("id"),
                source_id: row.get("source_id"),
                title: row.get("title"),
                url: row.get("url"),
                content: row.get("content"),
                markdown_content: row.get("markdown_content"),
                last_updated: row.get("last_updated"),
                path: row.get("path"),
                section: row.get("section"),
            };

            // Calculate relevance score
            let relevance_score = self.calculate_relevance_score(&page, &search_query.query, &search_query.ranking_preferences);

            // Extract matched snippets
            let matched_snippets = self.extract_snippets(&page, &search_query.query);

            // Find related pages (simplified for now)
            let related_pages = self.find_related_pages(&page).await?;

            results.push(SearchResult {
                page,
                relevance_score,
                matched_snippets,
                related_pages,
            });
        }

        // Sort by relevance score if needed
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    fn calculate_relevance_score(&self, page: &DocumentPage, query: &str, preferences: &RankingPreferences) -> f32 {
        let mut score = 0.0;

        // Title match boost
        if page.title.to_lowercase().contains(&query.to_lowercase()) {
            score += 10.0;
        }

        // Content match
        if page.content.to_lowercase().contains(&query.to_lowercase()) {
            score += 5.0;
        }

        // Recency boost
        if preferences.prioritize_recent {
            let days_old = (Utc::now() - page.last_updated).num_days();
            score += (30.0 - days_old.min(30) as f32) / 30.0 * 3.0;
        }

        // Example boost
        if preferences.prioritize_examples && (page.content.contains("```") || page.markdown_content.contains("```")) {
            score += 5.0;
        }

        // Context similarity (simplified)
        score += preferences.context_similarity_weight * 2.0;

        score
    }

    fn extract_snippets(&self, page: &DocumentPage, query: &str) -> Vec<ContentSnippet> {
        let mut snippets = Vec::new();
        let query_lower = query.to_lowercase();

        // Search in content
        if let Some(pos) = page.content.to_lowercase().find(&query_lower) {
            let start = pos.saturating_sub(100);
            let end = (pos + query.len() + 100).min(page.content.len());
            let snippet_text = &page.content[start..end];
            
            snippets.push(ContentSnippet {
                content: snippet_text.to_string(),
                highlight_ranges: vec![(pos - start, pos - start + query.len())],
                snippet_type: if snippet_text.contains("```") { SnippetType::CodeExample } else { SnippetType::Definition },
            });
        }

        // Limit to top 3 snippets
        snippets.truncate(3);
        snippets
    }

    async fn find_related_pages(&self, page: &DocumentPage) -> Result<Vec<String>> {
        // Find pages from the same source or with similar topics
        let rows = sqlx::query(
            "SELECT id FROM document_pages WHERE source_id = ? AND id != ? LIMIT 5"
        )
        .bind(&page.source_id)
        .bind(&page.id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|row| row.get::<String, _>("id")).collect())
    }

    /// Get last update time for a documentation type
    pub async fn get_last_update_time_for_type(&self, doc_type: &DocType) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            "SELECT MAX(last_updated) as last_update FROM documentation_sources WHERE doc_type = ?"
        )
        .bind(doc_type.as_str())
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(row.get("last_update"))
        } else {
            Ok(None)
        }
    }

    /// Update source last_updated timestamp
    pub async fn update_source_timestamp(&self, source_id: &str) -> Result<()> {
        sqlx::query(
            "UPDATE documentation_sources SET last_updated = ? WHERE id = ?"
        )
        .bind(Utc::now())
        .bind(source_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Clear all documents for a specific source
    pub async fn clear_source_documents(&self, source_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM document_pages WHERE source_id = ?")
            .bind(source_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Get a specific document by source ID and path
    pub async fn get_document_by_path(&self, source_id: &str, path: &str) -> Result<Option<DocumentPage>> {
        let row = sqlx::query(
            "SELECT id, source_id, title, url, content, markdown_content, last_updated, path, section 
             FROM document_pages 
             WHERE source_id = ? AND path = ?"
        )
        .bind(source_id)
        .bind(path)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            Ok(Some(DocumentPage {
                id: row.get("id"),
                source_id: row.get("source_id"),
                title: row.get("title"),
                url: row.get("url"),
                content: row.get("content"),
                markdown_content: row.get("markdown_content"),
                last_updated: row.get("last_updated"),
                path: row.get("path"),
                section: row.get("section"),
            }))
        } else {
            Ok(None)
        }
    }

    /// Get the count of pages for a specific documentation type
    pub async fn get_page_count_for_type(&self, doc_type: &DocType) -> Result<i32> {
        let count = sqlx::query(
            "SELECT COUNT(*) as count FROM document_pages d 
             JOIN documentation_sources s ON d.source_id = s.id 
             WHERE s.doc_type = ?",
        )
        .bind(doc_type.as_str())
        .fetch_one(&self.pool)
        .await?;
        
        let count: i64 = count.get("count");
        Ok(count as i32)
    }

    /// Count documents for a specific source
    pub async fn count_source_documents(&self, source_id: &str) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM document_pages WHERE source_id = ?")
            .bind(source_id)
            .fetch_one(&self.pool)
            .await?;
        
        Ok(row.get("count"))
    }

    // Vector Embeddings Methods
    pub async fn store_document_embedding(&self, embedding: &DocumentEmbedding) -> Result<i64> {
        let query = r#"
            INSERT INTO document_embeddings (page_id, embedding, chunk_index, embedding_model, chunk_text, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#;
        
        let id = sqlx::query(query)
            .bind(&embedding.page_id)
            .bind(&embedding.embedding) // Already a JSON string
            .bind(embedding.chunk_index)
            .bind(&embedding.embedding_model)
            .bind(&embedding.chunk_text)
            .bind(embedding.created_at)
            .bind(embedding.created_at)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();
            
        Ok(id)
    }

    pub async fn get_document_embeddings(&self, page_id: &str) -> Result<Vec<DocumentEmbedding>> {
        let query = r#"
            SELECT id, page_id, embedding, chunk_index, total_chunks, embedding_model, created_at
            FROM document_embeddings 
            WHERE page_id = ?1
            ORDER BY chunk_index
        "#;
        
        let embeddings = sqlx::query_as::<_, DocumentEmbedding>(query)
            .bind(page_id)
            .fetch_all(&self.pool)
            .await?;
            
        Ok(embeddings)
    }

    pub async fn search_similar_embeddings(&self, embedding: &[u8], limit: Option<i32>) -> Result<Vec<(String, f32)>> {
        // Note: SQLite doesn't have native vector similarity search
        // This is a simplified implementation - in production, consider using a vector database
        let query = r#"
            SELECT DISTINCT page_id
            FROM document_embeddings
            LIMIT ?1
        "#;
        
        let limit = limit.unwrap_or(10);
        let rows = sqlx::query(query)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
            
        let mut results = Vec::new();
        for row in rows {
            let page_id: String = row.get("page_id");
            // Simplified similarity score - in practice, you'd calculate cosine similarity
            results.push((page_id, 0.8));
        }
        
        Ok(results)
    }

    // User Interaction Methods
    pub async fn record_user_interaction(&self, interaction: &UserInteraction) -> Result<i64> {
        let query = r#"
            INSERT INTO user_interactions (session_id, page_id, interaction_type, duration_seconds, metadata, timestamp)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#;
        
        let metadata_json = serde_json::to_string(&interaction.metadata).unwrap_or_default();
        
        let id = sqlx::query(query)
            .bind(&interaction.session_id)
            .bind(&interaction.page_id)
            .bind(&interaction.interaction_type)
            .bind(interaction.duration_seconds)
            .bind(metadata_json)
            .bind(interaction.timestamp)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();
            
        Ok(id)
    }

    pub async fn get_user_interactions(&self, session_id: &str, limit: Option<i32>) -> Result<Vec<UserInteraction>> {
        let query = r#"
            SELECT id, session_id, page_id, interaction_type, duration_seconds, metadata, timestamp
            FROM user_interactions 
            WHERE session_id = ?1
            ORDER BY timestamp DESC
            LIMIT ?2
        "#;
        
        let limit = limit.unwrap_or(100);
        let interactions = sqlx::query_as::<_, UserInteraction>(query)
            .bind(session_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
            
        Ok(interactions)
    }

    // Learning Path Methods
    pub async fn create_learning_path(&self, path: &LearningPath) -> Result<i64> {
        let query = r#"
            INSERT INTO learning_paths (id, title, description, difficulty_level, estimated_duration_minutes, doc_type, created_by, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#;
        
        sqlx::query(query)
            .bind(&path.id)
            .bind(&path.title)
            .bind(&path.description)
            .bind(&path.difficulty_level)
            .bind(path.estimated_duration_minutes)
            .bind(&path.doc_type)
            .bind(&path.created_by)
            .bind(path.created_at)
            .bind(path.updated_at)
            .execute(&self.pool)
            .await?;
            
        Ok(1) // Return success indicator
    }

    pub async fn create_learning_path_step(&self, step: &LearningPathStep) -> Result<i64> {
        let query = r#"
            INSERT INTO learning_path_steps (path_id, step_order, page_id, title, description, is_optional, estimated_duration_minutes, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#;
        
        let id = sqlx::query(query)
            .bind(&step.path_id)
            .bind(step.step_order)
            .bind(&step.page_id)
            .bind(&step.title)
            .bind(&step.description)
            .bind(step.is_optional)
            .bind(step.estimated_duration_minutes)
            .bind(step.created_at)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();
            
        Ok(id)
    }

    pub async fn get_learning_path(&self, path_id: &str) -> Result<Option<LearningPath>> {
        let query = r#"
            SELECT id, title, description, difficulty_level, estimated_duration_minutes, doc_type, created_by, created_at, updated_at
            FROM learning_paths 
            WHERE id = ?1
        "#;
        
        let path = sqlx::query_as::<_, LearningPath>(query)
            .bind(path_id)
            .fetch_optional(&self.pool)
            .await?;
            
        Ok(path)
    }

    pub async fn get_learning_path_steps(&self, path_id: &str) -> Result<Vec<LearningPathStep>> {
        let query = r#"
            SELECT id, path_id, step_order, page_id, title, description, is_optional, estimated_duration_minutes, created_at
            FROM learning_path_steps 
            WHERE path_id = ?1
            ORDER BY step_order
        "#;
        
        let steps = sqlx::query_as::<_, LearningPathStep>(query)
            .bind(path_id)
            .fetch_all(&self.pool)
            .await?;
            
        Ok(steps)
    }

    pub async fn get_completed_learning_paths(&self, session_id: &str) -> Result<Vec<String>> {
        let query = r#"
            SELECT DISTINCT p.path_id
            FROM user_learning_progress p
            JOIN learning_path_steps s ON p.step_id = s.id
            WHERE p.session_id = ?1 AND p.completed = true
            GROUP BY p.path_id
            HAVING COUNT(*) = (
                SELECT COUNT(*) 
                FROM learning_path_steps 
                WHERE path_id = p.path_id AND is_optional = false
            )
        "#;
        
        let rows = sqlx::query(query)
            .bind(session_id)
            .fetch_all(&self.pool)
            .await?;
            
        let paths = rows.into_iter()
            .map(|row| row.get::<String, _>("path_id"))
            .collect();
            
        Ok(paths)
    }

    // User Learning Progress Methods
    pub async fn update_learning_progress(&self, progress: &UserLearningProgress) -> Result<()> {
        let query = r#"
            INSERT OR REPLACE INTO user_learning_progress 
            (session_id, path_id, step_id, completed, completion_time, notes, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#;
        
        sqlx::query(query)
            .bind(&progress.session_id)
            .bind(&progress.path_id)
            .bind(progress.step_id)
            .bind(progress.completed)
            .bind(progress.completion_time)
            .bind(&progress.notes)
            .bind(progress.created_at)
            .bind(progress.updated_at)
            .execute(&self.pool)
            .await?;
            
        Ok(())
    }

    pub async fn get_user_progress(&self, session_id: &str, path_id: &str) -> Result<Vec<UserLearningProgress>> {
        let query = r#"
            SELECT id, session_id, path_id, step_id, completed, completion_time, notes, created_at, updated_at
            FROM user_learning_progress 
            WHERE session_id = ?1 AND path_id = ?2
            ORDER BY step_id
        "#;
        
        let progress = sqlx::query_as::<_, UserLearningProgress>(query)
            .bind(session_id)
            .bind(path_id)
            .fetch_all(&self.pool)
            .await?;
            
        Ok(progress)
    }

    // Document Quality Metrics Methods
    pub async fn store_quality_metrics(&self, metrics: &DocumentQualityMetrics) -> Result<()> {
        let query = r#"
            INSERT OR REPLACE INTO document_quality_metrics 
            (page_id, readability_score, completeness_score, accuracy_score, freshness_score, overall_score, calculated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#;
        
        sqlx::query(query)
            .bind(&metrics.page_id)
            .bind(metrics.readability_score)
            .bind(metrics.completeness_score)
            .bind(metrics.accuracy_score)
            .bind(metrics.freshness_score)
            .bind(metrics.overall_score)
            .bind(metrics.calculated_at)
            .execute(&self.pool)
            .await?;
            
        Ok(())
    }

    pub async fn get_quality_metrics(&self, page_id: &str) -> Result<Option<DocumentQualityMetrics>> {
        let query = r#"
            SELECT id, page_id, readability_score, completeness_score, accuracy_score, freshness_score, overall_score, calculated_at
            FROM document_quality_metrics 
            WHERE page_id = ?1
        "#;
        
        let metrics = sqlx::query_as::<_, DocumentQualityMetrics>(query)
            .bind(page_id)
            .fetch_optional(&self.pool)
            .await?;
            
        Ok(metrics)
    }

    // Content Suggestions Methods
    pub async fn store_content_suggestion(&self, suggestion: &ContentSuggestion) -> Result<i64> {
        let query = r#"
            INSERT INTO content_suggestions 
            (session_id, suggested_page_id, suggestion_type, confidence_score, reason, shown, clicked, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#;
        
        let id = sqlx::query(query)
            .bind(&suggestion.session_id)
            .bind(&suggestion.suggested_page_id)
            .bind(&suggestion.suggestion_type)
            .bind(suggestion.confidence_score)
            .bind(&suggestion.reason)
            .bind(suggestion.shown)
            .bind(suggestion.clicked)
            .bind(suggestion.created_at)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();
            
        Ok(id)
    }

    pub async fn get_content_suggestions(&self, session_id: &str, limit: Option<i32>) -> Result<Vec<ContentSuggestion>> {
        let query = r#"
            SELECT id, session_id, suggested_page_id, suggestion_type, confidence_score, reason, shown, clicked, created_at
            FROM content_suggestions 
            WHERE session_id = ?1 AND shown = false
            ORDER BY confidence_score DESC
            LIMIT ?2
        "#;
        
        let limit = limit.unwrap_or(10);
        let suggestions = sqlx::query_as::<_, ContentSuggestion>(query)
            .bind(session_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
            
        Ok(suggestions)
    }

    // Search Analytics Methods
    pub async fn record_search_analytics(&self, analytics: &SearchAnalytics) -> Result<i64> {
        let query = r#"
            INSERT INTO search_analytics 
            (session_id, query, results_count, clicked_page_id, click_position, search_time_ms, filters_used, timestamp)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#;
        
        let filters_json = serde_json::to_string(&analytics.filters_used).unwrap_or_default();
        
        let id = sqlx::query(query)
            .bind(&analytics.session_id)
            .bind(&analytics.query)
            .bind(analytics.results_count)
            .bind(&analytics.clicked_page_id)
            .bind(analytics.click_position)
            .bind(analytics.search_time_ms)
            .bind(filters_json)
            .bind(analytics.timestamp)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();
            
        Ok(id)
    }

    // Search and Topic Methods
    pub async fn search_pages_by_topic(&self, topic: &str) -> Result<Vec<DocumentPage>> {
        let query = r#"
            SELECT id, url, title, content, doc_type, last_updated, content_hash
            FROM document_pages 
            WHERE title LIKE ?1 OR content LIKE ?1
            ORDER BY 
                CASE WHEN title LIKE ?1 THEN 1 ELSE 2 END,
                LENGTH(content) DESC
            LIMIT 50
        "#;
        
        let search_pattern = format!("%{}%", topic);
        let pages = sqlx::query_as::<_, DocumentPage>(query)
            .bind(&search_pattern)
            .fetch_all(&self.pool)
            .await?;
            
        Ok(pages)
    }

    // Document Relationships Methods
    pub async fn create_document_relationship(&self, relationship: &DocumentRelationship) -> Result<i64> {
        let query = r#"
            INSERT INTO document_relationships 
            (source_page_id, target_page_id, relationship_type, strength, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
        "#;
        
        let id = sqlx::query(query)
            .bind(&relationship.source_page_id)
            .bind(&relationship.target_page_id)
            .bind(&relationship.relationship_type)
            .bind(relationship.strength)
            .bind(relationship.created_at)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();
            
        Ok(id)
    }

    pub async fn get_related_documents(&self, page_id: &str, relationship_type: Option<RelationshipType>) -> Result<Vec<DocumentRelationship>> {
        let query = if let Some(ref rel_type) = relationship_type {
            r#"
                SELECT id, source_page_id, target_page_id, relationship_type, strength, created_at
                FROM document_relationships 
                WHERE source_page_id = ?1 AND relationship_type = ?2
                ORDER BY strength DESC
                LIMIT 20
            "#
        } else {
            r#"
                SELECT id, source_page_id, target_page_id, relationship_type, strength, created_at
                FROM document_relationships 
                WHERE source_page_id = ?1
                ORDER BY strength DESC
                LIMIT 20
            "#
        };
        
        let mut query_builder = sqlx::query_as::<_, DocumentRelationship>(query)
            .bind(page_id);
            
        if let Some(rel_type) = relationship_type {
            query_builder = query_builder.bind(rel_type);
        }
        
        let relationships = query_builder
            .fetch_all(&self.pool)
            .await?;
            
        Ok(relationships)
    }

    // Migration method for the new schema
    pub async fn run_advanced_features_migration(&self) -> Result<()> {
        let migration_sql = include_str!("../migrations/002_advanced_features.sql");
        
        // Split the migration into individual statements
        let statements: Vec<&str> = migration_sql
            .split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && !s.starts_with("--"))
            .collect();
        
        for statement in statements {
            if !statement.trim().is_empty() {
                sqlx::query(statement)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("Migration error: {}", e))?;
            }
        }
        
        Ok(())
    }

    // Missing methods for compilation - these need to be implemented
    pub async fn get_page(&self, page_id: &str) -> Result<Option<DocumentPage>> {
        // TODO: Implement actual database query
        Ok(None)
    }

    pub async fn get_page_view_count(&self, page_id: &str) -> Result<Option<i32>> {
        // TODO: Implement actual database query  
        Ok(Some(0))
    }

    pub async fn get_page_bookmark_count(&self, page_id: &str) -> Result<Option<i32>> {
        // TODO: Implement actual database query
        Ok(Some(0))
    }

    pub async fn is_page_in_learning_paths(&self, page_id: &str, user_context: &UserContext) -> Result<bool> {
        // TODO: Implement actual database query
        Ok(false)
    }

    pub async fn get_search_analytics_summary(&self) -> Result<SearchAnalytics> {
        // TODO: Implement actual database query
        Ok(SearchAnalytics {
            id: None,
            session_id: "summary".to_string(),
            query: "summary".to_string(),
            doc_type: None,
            results_count: 0,
            clicked_page_id: None,
            click_position: None,
            search_duration_ms: None,
            filters_used: "{}".to_string(),
            search_time_ms: 0,
            timestamp: Utc::now(),
            created_at: Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentEmbedding {
    pub id: Option<i64>,
    pub page_id: String,
    pub embedding_model: String,
    pub embedding: String, // JSON serialized Vec<f32>
    pub chunk_index: i32,
    pub chunk_text: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct UserInteraction {
    pub id: Option<i64>,
    pub session_id: String,
    pub page_id: String,
    pub interaction_type: InteractionType,
    pub context: Option<String>,
    pub duration_seconds: Option<i32>,
    pub metadata: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum InteractionType {
    View,
    Search,
    Copy,
    Bookmark,
    Rate,
}

impl InteractionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            InteractionType::View => "view",
            InteractionType::Search => "search",
            InteractionType::Copy => "copy",
            InteractionType::Bookmark => "bookmark",
            InteractionType::Rate => "rate",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct LearningPath {
    pub id: String,
    pub title: String,
    pub description: String,
    pub difficulty_level: DifficultyLevel,
    pub estimated_duration_minutes: i32,
    pub doc_type: DocType,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
}

impl DifficultyLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            DifficultyLevel::Beginner => "beginner",
            DifficultyLevel::Intermediate => "intermediate",
            DifficultyLevel::Advanced => "advanced",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct LearningPathStep {
    pub id: Option<i64>,
    pub path_id: String,
    pub step_order: i32,
    pub page_id: String,
    pub title: String,
    pub description: Option<String>,
    pub is_optional: bool,
    pub estimated_duration_minutes: Option<i32>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct UserLearningProgress {
    pub id: Option<i64>,
    pub session_id: String,
    pub path_id: String,
    pub step_id: i64,
    pub completed: bool,
    pub completion_time: Option<DateTime<Utc>>,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentRelationship {
    pub id: Option<i64>,
    pub source_page_id: String,
    pub target_page_id: String,
    pub related_page_id: String,
    pub relationship_type: RelationshipType,
    pub strength: f32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum RelationshipType {
    Prerequisite,
    FollowUp,
    Related,
    Example,
}

impl RelationshipType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationshipType::Prerequisite => "prerequisite",
            RelationshipType::FollowUp => "follow_up",
            RelationshipType::Related => "related",
            RelationshipType::Example => "example",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchAnalytics {
    pub id: Option<i64>,
    pub session_id: String,
    pub query: String,
    pub doc_type: Option<DocType>,
    pub results_count: i32,
    pub clicked_page_id: Option<String>,
    pub click_position: Option<i32>,
    pub search_duration_ms: Option<i32>,
    pub filters_used: String,
    pub search_time_ms: i32,
    pub timestamp: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DocumentQualityMetrics {
    pub page_id: String,
    pub freshness_score: f32,
    pub completeness_score: f32,
    pub accuracy_score: f32,
    pub readability_score: f32,
    pub overall_score: f32,
    pub user_rating_avg: f32,
    pub view_count: i32,
    pub bookmark_count: i32,
    pub calculated_at: DateTime<Utc>,
    pub last_calculated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ContentSuggestion {
    pub id: Option<i64>,
    pub session_id: String,
    pub suggested_page_id: String,
    pub suggestion_type: SuggestionType,
    pub confidence_score: f32,
    pub reason: Option<String>,
    pub shown: bool,
    pub clicked: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum SuggestionType {
    NextStep,
    Related,
    Prerequisite,
    Advanced,
}

impl SuggestionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SuggestionType::NextStep => "next_step",
            SuggestionType::Related => "related",
            SuggestionType::Prerequisite => "prerequisite",
            SuggestionType::Advanced => "advanced",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSearchResult {
    pub page: DocumentPage,
    pub relevance_score: f32,
    pub ranking_factors: RankingFactors,
    pub related_pages: Vec<DocumentPage>,
    pub learning_suggestions: Vec<ContentSuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFactors {
    pub text_relevance: f32,
    pub semantic_similarity: f32,
    pub quality_score: f32,
    pub freshness_score: f32,
    pub user_engagement: f32,
    pub context_relevance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchQuery {
    pub query: String,
    pub embedding: Vec<f32>,
    pub filters: SearchFilters,
    pub user_context: UserContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub session_id: String,
    pub skill_level: Option<DifficultyLevel>,
    pub preferred_doc_types: Vec<DocType>,
    pub recent_interactions: Vec<UserInteraction>,
    pub current_learning_paths: Vec<String>,
}
