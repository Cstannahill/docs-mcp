-- Enhanced schema for advanced ranking, vector embeddings, and interactive features

-- Vector embeddings table for semantic search
CREATE TABLE IF NOT EXISTS document_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    embedding BLOB NOT NULL, -- Store embeddings as binary data
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES document_pages(id) ON DELETE CASCADE
);

-- Conversation turns for context management
CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    context_tokens INTEGER NOT NULL DEFAULT 0,
    relevance_score REAL NOT NULL DEFAULT 1.0,
    model_used TEXT,
    metadata TEXT DEFAULT '{}', -- JSON metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- User interaction tracking for personalized ranking
CREATE TABLE IF NOT EXISTS user_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    interaction_type TEXT NOT NULL, -- 'view', 'search', 'copy', 'bookmark', 'rate'
    context TEXT, -- Search query, rating value, etc.
    duration_seconds INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES document_pages(id) ON DELETE CASCADE
);

-- Learning paths and tutorials
CREATE TABLE IF NOT EXISTS learning_paths (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    difficulty_level TEXT NOT NULL, -- 'beginner', 'intermediate', 'advanced'
    estimated_duration_minutes INTEGER,
    doc_type TEXT NOT NULL,
    created_by TEXT DEFAULT 'system',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Steps within learning paths
CREATE TABLE IF NOT EXISTS learning_path_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path_id TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    page_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    is_optional BOOLEAN DEFAULT FALSE,
    estimated_duration_minutes INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (path_id) REFERENCES learning_paths(id) ON DELETE CASCADE,
    FOREIGN KEY (page_id) REFERENCES document_pages(id) ON DELETE CASCADE,
    UNIQUE(path_id, step_order)
);

-- User progress through learning paths
CREATE TABLE IF NOT EXISTS user_learning_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    path_id TEXT NOT NULL,
    step_id INTEGER NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    completion_time DATETIME,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (path_id) REFERENCES learning_paths(id) ON DELETE CASCADE,
    FOREIGN KEY (step_id) REFERENCES learning_path_steps(id) ON DELETE CASCADE,
    UNIQUE(session_id, path_id, step_id)
);

-- Document relationships and recommendations
CREATE TABLE IF NOT EXISTS document_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_page_id TEXT NOT NULL,
    related_page_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL, -- 'prerequisite', 'follow_up', 'related', 'example'
    strength REAL NOT NULL DEFAULT 1.0, -- 0.0 to 1.0
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_page_id) REFERENCES document_pages(id) ON DELETE CASCADE,
    FOREIGN KEY (related_page_id) REFERENCES document_pages(id) ON DELETE CASCADE,
    UNIQUE(source_page_id, related_page_id, relationship_type)
);

-- Search analytics for ranking improvements
CREATE TABLE IF NOT EXISTS search_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    doc_type TEXT,
    results_count INTEGER NOT NULL,
    clicked_page_id TEXT,
    click_position INTEGER,
    search_duration_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (clicked_page_id) REFERENCES document_pages(id) ON DELETE SET NULL
);

-- Document quality metrics for ranking
CREATE TABLE IF NOT EXISTS document_quality_metrics (
    page_id TEXT PRIMARY KEY,
    freshness_score REAL DEFAULT 0.0, -- 0.0 to 1.0
    completeness_score REAL DEFAULT 0.0,
    accuracy_score REAL DEFAULT 1.0,
    user_rating_avg REAL DEFAULT 0.0,
    view_count INTEGER DEFAULT 0,
    bookmark_count INTEGER DEFAULT 0,
    last_calculated DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES document_pages(id) ON DELETE CASCADE
);

-- Adaptive content suggestions
CREATE TABLE IF NOT EXISTS content_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    suggested_page_id TEXT NOT NULL,
    suggestion_type TEXT NOT NULL, -- 'next_step', 'related', 'prerequisite', 'advanced'
    confidence_score REAL NOT NULL DEFAULT 0.0,
    reason TEXT,
    shown BOOLEAN DEFAULT FALSE,
    clicked BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (suggested_page_id) REFERENCES document_pages(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_document_embeddings_page_id ON document_embeddings(page_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_model ON document_embeddings(embedding_model);
CREATE INDEX IF NOT EXISTS idx_user_interactions_session ON user_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_page ON user_interactions(page_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_type ON user_interactions(interaction_type);
CREATE INDEX IF NOT EXISTS idx_learning_path_steps_path ON learning_path_steps(path_id);
CREATE INDEX IF NOT EXISTS idx_learning_path_steps_order ON learning_path_steps(path_id, step_order);
CREATE INDEX IF NOT EXISTS idx_user_progress_session ON user_learning_progress(session_id);
CREATE INDEX IF NOT EXISTS idx_user_progress_path ON user_learning_progress(path_id);
CREATE INDEX IF NOT EXISTS idx_document_relationships_source ON document_relationships(source_page_id);
CREATE INDEX IF NOT EXISTS idx_document_relationships_related ON document_relationships(related_page_id);
CREATE INDEX IF NOT EXISTS idx_search_analytics_session ON search_analytics(session_id);
CREATE INDEX IF NOT EXISTS idx_search_analytics_query ON search_analytics(query);
CREATE INDEX IF NOT EXISTS idx_content_suggestions_session ON content_suggestions(session_id);
CREATE INDEX IF NOT EXISTS idx_content_suggestions_type ON content_suggestions(suggestion_type);
