-- Create documentation sources table
CREATE TABLE IF NOT EXISTS documentation_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    last_updated DATETIME,
    version TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create document pages table
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES documentation_sources(id) ON DELETE CASCADE
);

-- Create indexes for better search performance
CREATE INDEX IF NOT EXISTS idx_document_pages_source_id ON document_pages(source_id);
CREATE INDEX IF NOT EXISTS idx_document_pages_path ON document_pages(path);
CREATE INDEX IF NOT EXISTS idx_document_pages_title ON document_pages(title);
CREATE INDEX IF NOT EXISTS idx_document_pages_content ON document_pages(content);

-- Create full-text search index for better content searching
CREATE VIRTUAL TABLE IF NOT EXISTS document_pages_fts USING fts5(
    title,
    content,
    markdown_content,
    content=document_pages,
    content_rowid=rowid
);

-- Trigger to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS document_pages_fts_insert AFTER INSERT ON document_pages BEGIN
    INSERT INTO document_pages_fts(rowid, title, content, markdown_content)
    VALUES (new.rowid, new.title, new.content, new.markdown_content);
END;

CREATE TRIGGER IF NOT EXISTS document_pages_fts_delete AFTER DELETE ON document_pages BEGIN
    INSERT INTO document_pages_fts(document_pages_fts, rowid, title, content, markdown_content)
    VALUES ('delete', old.rowid, old.title, old.content, old.markdown_content);
END;

CREATE TRIGGER IF NOT EXISTS document_pages_fts_update AFTER UPDATE ON document_pages BEGIN
    INSERT INTO document_pages_fts(document_pages_fts, rowid, title, content, markdown_content)
    VALUES ('delete', old.rowid, old.title, old.content, old.markdown_content);
    INSERT INTO document_pages_fts(rowid, title, content, markdown_content)
    VALUES (new.rowid, new.title, new.content, new.markdown_content);
END;
