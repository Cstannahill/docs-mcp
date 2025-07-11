use anyhow::{anyhow, Result};
use chrono::Utc;
use reqwest::Client;
use scraper::{Html, Selector};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Semaphore, Mutex};
use tokio::time::sleep;
use tracing::{info, warn, error};
use url::Url;

use crate::database::{Database, DocumentPage, DocumentationSource, DocType};

pub struct DocumentationFetcher {
    client: Client,
    db: Database,
    // Concurrency control - limit concurrent requests to be respectful
    semaphore: Arc<Semaphore>,
    // Batch size for concurrent operations
    batch_size: usize,
}

impl DocumentationFetcher {
    pub fn new(db: Database) -> Self {
        let client = Client::builder()
            .user_agent("docs-mcp-server/1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self { 
            client, 
            db,
            // Allow up to 10 concurrent requests to be respectful to servers
            semaphore: Arc::new(Semaphore::new(10)),
            batch_size: 50, // Process URLs in batches of 50
        }
    }

    pub async fn update_all_documentation(&self) -> Result<()> {
        let sources = self.get_documentation_sources().await?;
        
        for source in sources {
            info!("Updating documentation for {}", source.name);
            if let Err(e) = self.update_source_documentation(&source).await {
                error!("Failed to update {}: {}", source.name, e);
            }
        }

        Ok(())
    }

    /// Update documentation for a specific type
    pub async fn update_documentation_by_type(&self, doc_type: &DocType) -> Result<()> {
        let sources = self.get_documentation_sources().await?;
        let filtered_sources: Vec<_> = sources.into_iter()
            .filter(|source| &source.doc_type == doc_type)
            .collect();

        if filtered_sources.is_empty() {
            warn!("No sources found for doc type: {:?}", doc_type);
            return Ok(());
        }

        for source in filtered_sources {
            info!("Updating {} documentation: {}", doc_type.as_str(), source.name);
            if let Err(e) = self.update_source_documentation(&source).await {
                error!("Failed to update {}: {}", source.name, e);
            }
        }

        Ok(())
    }

    /// Update a specific source by ID
    pub async fn update_source_by_id(&self, source_id: &str) -> Result<()> {
        let sources = self.get_documentation_sources().await?;
        let source = sources.into_iter()
            .find(|s| s.id == source_id)
            .ok_or_else(|| anyhow!("Source not found: {}", source_id))?;

        info!("Updating specific source: {}", source.name);
        self.update_source_documentation(&source).await
    }

    async fn get_documentation_sources(&self) -> Result<Vec<DocumentationSource>> {
        // First, ensure all sources are in the database with current versions
        let mut default_sources = vec![
            DocumentationSource {
                id: "rust-std".to_string(),
                name: "Rust Standard Library".to_string(),
                base_url: "https://doc.rust-lang.org/std/".to_string(),
                doc_type: DocType::Rust,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "rust-book".to_string(),
                name: "The Rust Programming Language".to_string(),
                base_url: "https://doc.rust-lang.org/book/".to_string(),
                doc_type: DocType::Rust,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "tauri-docs".to_string(),
                name: "Tauri Documentation".to_string(),
                base_url: "https://v2.tauri.app/".to_string(),
                doc_type: DocType::Tauri,
                last_updated: None,
                version: Some("2.0".to_string()),
            },
            DocumentationSource {
                id: "react-docs".to_string(),
                name: "React Documentation".to_string(),
                base_url: "https://react.dev/".to_string(),
                doc_type: DocType::React,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "typescript-docs".to_string(),
                name: "TypeScript Handbook".to_string(),
                base_url: "https://www.typescriptlang.org/docs/".to_string(),
                doc_type: DocType::TypeScript,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "python-docs".to_string(),
                name: "Python Documentation".to_string(),
                base_url: "https://docs.python.org/3/".to_string(),
                doc_type: DocType::Python,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "tailwind-docs".to_string(),
                name: "Tailwind CSS Documentation".to_string(),
                base_url: "https://tailwindcss.com/docs/".to_string(),
                doc_type: DocType::Tailwind,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "shadcn-docs".to_string(),
                name: "shadcn/ui Documentation".to_string(),
                base_url: "https://ui.shadcn.com/docs/".to_string(),
                doc_type: DocType::Shadcn,
                last_updated: None,
                version: None,
            },
            // Package Manager Documentation Sources
            DocumentationSource {
                id: "cargo-docs".to_string(),
                name: "Cargo Documentation".to_string(),
                base_url: "https://doc.rust-lang.org/cargo/".to_string(),
                doc_type: DocType::Rust,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "npm-docs".to_string(),
                name: "npm Documentation".to_string(),
                base_url: "https://docs.npmjs.com/".to_string(),
                doc_type: DocType::TypeScript,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "pip-docs".to_string(),
                name: "pip Documentation".to_string(),
                base_url: "https://pip.pypa.io/en/stable/".to_string(),
                doc_type: DocType::Python,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "pypi-help".to_string(),
                name: "PyPI Help".to_string(),
                base_url: "https://pypi.org/help/".to_string(),
                doc_type: DocType::Python,
                last_updated: None,
                version: None,
            },
            DocumentationSource {
                id: "python-packaging".to_string(),
                name: "Python Packaging User Guide".to_string(),
                base_url: "https://packaging.python.org/".to_string(),
                doc_type: DocType::Python,
                last_updated: None,
                version: None,
            },
        ];

        // Add/update sources to database
        for source in &default_sources {
            self.db.add_source(source).await?;
        }

        // Return sources from database (to get any updates)
        self.db.get_sources().await
    }

    async fn update_source_documentation(&self, source: &DocumentationSource) -> Result<()> {
        info!("Starting update for {}", source.name);
        
        // Clear existing documents for this source
        self.db.clear_source_documents(&source.id).await?;

        match source.doc_type {
            DocType::Rust => self.fetch_rust_docs(source).await,
            DocType::React => self.fetch_react_docs(source).await,
            DocType::TypeScript => self.fetch_typescript_docs(source).await,
            DocType::Python => self.fetch_python_docs(source).await,
            DocType::Tauri => self.fetch_tauri_docs_recursive(source).await,
            DocType::Tailwind => self.fetch_tailwind_docs(source).await,
            DocType::Shadcn => self.fetch_shadcn_docs(source).await,
        }
    }

    pub async fn fetch_rust_docs(&self, source: &DocumentationSource) -> Result<()> {
        if source.id == "rust-std" {
            self.fetch_rust_std_docs(source).await
        } else if source.id == "rust-book" {
            self.fetch_rust_book_docs(source).await
        } else {
            Err(anyhow!("Unknown Rust documentation source: {}", source.id))
        }
    }

    pub async fn fetch_rust_std_docs(&self, source: &DocumentationSource) -> Result<()> {
        // Use recursive crawling to get comprehensive std library documentation
        self.fetch_rust_std_docs_recursive(source).await
    }

    pub async fn fetch_rust_std_docs_recursive(&self, source: &DocumentationSource) -> Result<()> {
        let mut visited = HashSet::new();
        let mut to_visit = VecDeque::new();
        
        // Start with comprehensive seed URLs for major std components
        let starting_urls = vec![
            source.base_url.clone(), // Main index
            format!("{}index.html", source.base_url),
            format!("{}all.html", source.base_url), // All items index
            format!("{}keyword.html", source.base_url), // Keywords
            // Core modules that branch into many submodules
            format!("{}collections/index.html", source.base_url),
            format!("{}io/index.html", source.base_url),
            format!("{}fs/index.html", source.base_url),
            format!("{}net/index.html", source.base_url),
            format!("{}thread/index.html", source.base_url),
            format!("{}sync/index.html", source.base_url),
            format!("{}process/index.html", source.base_url),
            format!("{}env/index.html", source.base_url),
            format!("{}path/index.html", source.base_url),
            format!("{}time/index.html", source.base_url),
            format!("{}fmt/index.html", source.base_url),
            format!("{}str/index.html", source.base_url),
            format!("{}string/index.html", source.base_url),
            format!("{}vec/index.html", source.base_url),
            format!("{}hash/index.html", source.base_url),
            format!("{}convert/index.html", source.base_url),
            format!("{}iter/index.html", source.base_url),
            format!("{}ops/index.html", source.base_url),
            format!("{}cmp/index.html", source.base_url),
            format!("{}mem/index.html", source.base_url),
            format!("{}ptr/index.html", source.base_url),
            format!("{}slice/index.html", source.base_url),
            format!("{}array/index.html", source.base_url),
            format!("{}option/index.html", source.base_url),
            format!("{}result/index.html", source.base_url),
            format!("{}error/index.html", source.base_url),
            format!("{}panic/index.html", source.base_url),
            format!("{}ffi/index.html", source.base_url),
            format!("{}os/index.html", source.base_url),
            format!("{}arch/index.html", source.base_url),
            format!("{}primitive.bool.html", source.base_url),
            format!("{}primitive.char.html", source.base_url),
            format!("{}primitive.str.html", source.base_url),
            format!("{}primitive.slice.html", source.base_url),
            format!("{}primitive.array.html", source.base_url),
            format!("{}primitive.tuple.html", source.base_url),
            format!("{}primitive.pointer.html", source.base_url),
            format!("{}primitive.reference.html", source.base_url),
            format!("{}primitive.fn.html", source.base_url),
            format!("{}primitive.never.html", source.base_url),
            format!("{}primitive.unit.html", source.base_url),
            // Important traits and types
            format!("{}trait.Clone.html", source.base_url),
            format!("{}trait.Copy.html", source.base_url),
            format!("{}trait.Debug.html", source.base_url),
            format!("{}trait.Default.html", source.base_url),
            format!("{}trait.Drop.html", source.base_url),
            format!("{}trait.Eq.html", source.base_url),
            format!("{}trait.PartialEq.html", source.base_url),
            format!("{}trait.Ord.html", source.base_url),
            format!("{}trait.PartialOrd.html", source.base_url),
            format!("{}trait.Send.html", source.base_url),
            format!("{}trait.Sync.html", source.base_url),
            format!("{}trait.Iterator.html", source.base_url),
            format!("{}trait.IntoIterator.html", source.base_url),
            format!("{}trait.From.html", source.base_url),
            format!("{}trait.Into.html", source.base_url),
        ];

        // Add all starting URLs to the queue with fragment normalization
        for url in starting_urls {
            if let Ok(parsed_url) = Url::parse(&url) {
                let mut normalized_url = parsed_url.clone();
                normalized_url.set_fragment(None);
                
                if visited.insert(normalized_url.to_string()) {
                    to_visit.push_back(url);
                }
            }
        }

        let mut batch_urls = Vec::new();
        let batch_size = 30; // Process in batches of 30 for Rust std

        // Breadth-first crawling with concurrent processing
        while let Some(current_url) = to_visit.pop_front() {
            batch_urls.push(current_url.clone());
            
            // Process in batches
            if batch_urls.len() >= batch_size || to_visit.is_empty() {
                // Fetch and store current batch
                if let Err(e) = self.fetch_and_store_pages_batch(source, batch_urls.clone()).await {
                    warn!("Failed to process batch for Rust std docs: {}", e);
                }
                
                // Find new links from the current batch
                for url in &batch_urls {
                    if let Ok(html) = self.fetch_page(url).await {
                        let document = Html::parse_document(&html);
                        let link_selector = Selector::parse("a[href]").unwrap();
                        
                        for element in document.select(&link_selector) {
                            if let Some(href) = element.value().attr("href") {
                                // Resolve relative URLs
                                if let Ok(current_base) = Url::parse(url) {
                                    if let Ok(full_url) = current_base.join(href) {
                                        let url_str = full_url.to_string();
                                        
                                        // Filter for relevant std documentation
                                        if self.is_relevant_rust_std_url(&url_str, &source.base_url) {
                                            let mut normalized_url = full_url.clone();
                                            normalized_url.set_fragment(None);
                                            
                                            if visited.insert(normalized_url.to_string()) {
                                                to_visit.push_back(url_str);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                batch_urls.clear();
                
                // Reasonable limit to prevent excessive crawling
                if visited.len() > 3000 {
                    info!("Reached crawling limit for Rust std docs");
                    break;
                }
            }
        }

        info!("Rust std documentation crawl completed. Visited {} unique URLs.", visited.len());
        Ok(())
    }

    fn is_relevant_rust_std_url(&self, url: &str, base_url: &str) -> bool {
        // Must be within the std documentation domain
        if !url.starts_with(base_url) {
            return false;
        }
        
        // Exclude print versions and source code
        if url.contains("/print.html") || url.contains("?print") || 
           url.contains("/source/") || url.contains("/src/") {
            return false;
        }
        
        // Exclude external links and non-documentation paths
        if url.contains("://github.com") || url.contains("://crates.io") ||
           url.contains("/settings.html") || url.contains("/help.html") {
            return false;
        }
        
        // Include std library content: modules, structs, traits, functions, primitives
        url.contains("/std/") || 
        url.ends_with(".html") && (
            url.contains("/index.html") ||
            url.contains("struct.") ||
            url.contains("trait.") ||
            url.contains("enum.") ||
            url.contains("fn.") ||
            url.contains("type.") ||
            url.contains("constant.") ||
            url.contains("macro.") ||
            url.contains("primitive.") ||
            url.contains("keyword.") ||
            url.contains("all.html")
        )
    }

    pub async fn fetch_rust_book_docs(&self, source: &DocumentationSource) -> Result<()> {
        // Use recursive crawling to get ALL pages including subsections
        self.fetch_rust_book_docs_recursive(source).await
    }

    pub async fn fetch_rust_book_docs_recursive(&self, source: &DocumentationSource) -> Result<()> {
        let mut visited = HashSet::new();
        let mut to_visit = VecDeque::new();
        
        // Start with the main index and table of contents
        let starting_urls = vec![
            source.base_url.clone(), // Main index
            format!("{}title-page.html", source.base_url),
            format!("{}foreword.html", source.base_url),
            format!("{}ch00-00-introduction.html", source.base_url),
            // Main chapters - but crawler will find all subsections
            format!("{}ch01-00-getting-started.html", source.base_url),
            format!("{}ch02-00-guessing-game-tutorial.html", source.base_url),
            format!("{}ch03-00-common-programming-concepts.html", source.base_url),
            format!("{}ch04-00-understanding-ownership.html", source.base_url),
            format!("{}ch05-00-structs.html", source.base_url),
            format!("{}ch06-00-enums.html", source.base_url),
            format!("{}ch07-00-managing-growing-projects-with-packages-crates-and-modules.html", source.base_url), // Fixed URL
            format!("{}ch08-00-common-collections.html", source.base_url),
            format!("{}ch09-00-error-handling.html", source.base_url),
            format!("{}ch10-00-generics.html", source.base_url),
            format!("{}ch11-00-testing.html", source.base_url),
            format!("{}ch12-00-an-io-project.html", source.base_url),
            format!("{}ch13-00-functional-features.html", source.base_url),
            format!("{}ch14-00-more-about-cargo.html", source.base_url),
            format!("{}ch15-00-smart-pointers.html", source.base_url),
            format!("{}ch16-00-concurrency.html", source.base_url),
            format!("{}ch17-00-async-and-await.html", source.base_url), // Updated chapter
            format!("{}ch18-00-oop.html", source.base_url),
            format!("{}ch19-00-patterns.html", source.base_url),
            format!("{}ch20-00-advanced-features.html", source.base_url),
            format!("{}ch21-00-final-project-a-web-server.html", source.base_url),
            // Appendices
            format!("{}appendix-00.html", source.base_url),
            
            // Cargo Documentation and Package Manager
            "https://doc.rust-lang.org/cargo/".to_string(),
            "https://doc.rust-lang.org/cargo/getting-started/".to_string(),
            "https://doc.rust-lang.org/cargo/guide/".to_string(),
            "https://doc.rust-lang.org/cargo/reference/".to_string(),
            "https://doc.rust-lang.org/cargo/commands/".to_string(),
            "https://doc.rust-lang.org/cargo/getting-started/installation.html".to_string(),
            "https://doc.rust-lang.org/cargo/getting-started/first-steps.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/creating-a-new-project.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/cargo-toml-vs-cargo-lock.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/working-on-an-existing-project.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/dependencies.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/project-layout.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/cargo-home.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/build-cache.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/continuous-integration.html".to_string(),
            "https://doc.rust-lang.org/cargo/guide/publishing.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/cargo-targets.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/workspaces.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/features.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/profiles.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/config.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/environment-variables.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/build-scripts.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/publishing.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/package-id-spec.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/source-replacement.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/external-tools.html".to_string(),
            "https://doc.rust-lang.org/cargo/reference/semver.html".to_string(),
            
            // Crates.io Registry Documentation  
            "https://doc.crates.io/".to_string(),
            "https://crates.io/policies".to_string(),
            "https://crates.io/help".to_string(),
        ];

        // Add all starting URLs to visit queue
        for url in starting_urls {
            if visited.insert(url.clone()) {
                to_visit.push_back(url);
            }
        }

        let base_url = Url::parse(&source.base_url)?;
        let base_domain = base_url.domain().unwrap_or("").to_string();
        let mut stored_count = 0;

        while let Some(current_url) = to_visit.pop_front() {
            info!("Crawling Rust Book page: {}", current_url);
            
            // Fetch and store the current page
            match self.fetch_and_store_page(source, &current_url).await {
                Ok(_) => {
                    stored_count += 1;
                    info!("Successfully stored Rust Book page #{}: {}", stored_count, current_url);
                }
                Err(e) => {
                    warn!("Failed to fetch {}: {}", current_url, e);
                    continue;
                }
            }

            // Try to get HTML content for link crawling
            match self.fetch_page(&current_url).await {
                Ok(html) => {
                    let document = Html::parse_document(&html);
                    let link_selector = Selector::parse("a[href]").unwrap();

                    for element in document.select(&link_selector) {
                        if let Some(href) = element.value().attr("href") {
                            // Skip anchor links and external links
                            if href.starts_with('#') || href.starts_with("mailto:") || href.starts_with("javascript:") {
                                continue;
                            }

                            // Resolve URL
                            let resolved_url = if href.starts_with("http://") || href.starts_with("https://") {
                                // Already absolute URL - check if it's Rust Book
                                href.to_string()
                            } else if href.starts_with('/') {
                                format!("{}{}", base_url.origin().ascii_serialization(), href)
                            } else {
                                // Relative URL - join with current base
                                base_url.join(href).map(|u| u.to_string()).unwrap_or_else(|_| format!("{}/{}", source.base_url.trim_end_matches('/'), href))
                            };

                            if let Ok(url) = Url::parse(&resolved_url) {
                                if url.domain() == Some(&base_domain) {
                                    let path = url.path();
                                    // Include all Rust Book content: chapters (ch##), appendices, intro, etc.
                                    if (path.contains("/book/") || path.starts_with("/book/")) &&
                                       (path.contains("ch") || path.contains("appendix") || 
                                        path.contains("foreword") || path.contains("introduction") ||
                                        path.contains("title-page") || path.ends_with("index.html") ||
                                        path.ends_with(".html")) &&
                                       !path.contains("print.html") && // Skip print version
                                       !path.contains("SUMMARY.md") { // Skip markdown files
                                        
                                        // Normalize URL by removing fragment to prevent infinite loops
                                        let mut normalized_url = url.clone();
                                        normalized_url.set_fragment(None);
                                        let normalized_url_str = normalized_url.to_string();
                                        
                                        if visited.insert(normalized_url_str.clone()) {
                                            to_visit.push_back(normalized_url_str);
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    warn!("Failed to fetch HTML from {} for crawling: {}", current_url, e);
                }
            }

            // Reasonable limit to prevent infinite loops
            if visited.len() > 500 {
                info!("Reached crawling limit for Rust Book docs");
                break;
            }
        }

        info!("Rust Book docs crawling completed. Visited {} URLs, stored {} pages.", visited.len(), stored_count);
        Ok(())
    }

    async fn fetch_tauri_docs(&self, source: &DocumentationSource) -> Result<()> {
        self.fetch_tauri_docs_recursive(source).await
    }

    /// Recursively crawl all internal documentation pages for Tauri V2
    pub async fn fetch_tauri_docs_recursive(&self, source: &DocumentationSource) -> Result<()> {
        let base_url = Url::parse(&source.base_url)?;
        let base_domain = base_url.domain().unwrap_or("").to_string();
        let mut visited = HashSet::new();
        
        // Start with the main sections we know exist
        let mut to_visit = vec![
            source.base_url.clone(),
            format!("{}start/", source.base_url),
            format!("{}concept/", source.base_url),
            format!("{}security/", source.base_url),
            format!("{}develop/", source.base_url),
            format!("{}distribute/", source.base_url),
            format!("{}learn/", source.base_url),
            format!("{}plugin/", source.base_url),
            format!("{}about/", source.base_url),
        ];

        while let Some(url) = to_visit.pop() {
            if !visited.insert(url.clone()) {
                continue;
            }
            
            info!("Crawling Tauri page: {}", url);
            
            // Fetch HTML for crawling
            match self.fetch_page(&url).await {
                Ok(html) => {
                    // Store the page
                    let _ = self.fetch_and_store_page(source, &url).await;
                    let document = Html::parse_document(&html);
                    let link_selector = Selector::parse("a[href]").unwrap();
                    
                    for element in document.select(&link_selector) {
                        if let Some(href) = element.value().attr("href") {
                            // Skip anchor links (fragments) - they point to same page
                            if href.starts_with('#') {
                                continue;
                            }
                            
                            let resolved_url = if href.starts_with("http://") || href.starts_with("https://") {
                                href.to_string()
                            } else if href.starts_with('/') {
                                format!("{}{}", base_url.origin().ascii_serialization(), href)
                            } else {
                                base_url.join(href).map(|u| u.to_string()).unwrap_or_else(|_| format!("{}/{}", source.base_url.trim_end_matches('/'), href))
                            };
                            
                            if let Ok(url_obj) = Url::parse(&resolved_url) {
                                if url_obj.domain() == Some(&base_domain) && !visited.contains(&resolved_url) {
                                    // Include all paths under the main Tauri V2 site, but exclude fragments
                                    let path = url_obj.path();
                                    if (path.starts_with("/start/") || path.starts_with("/concept/") || 
                                       path.starts_with("/security/") || path.starts_with("/develop/") ||
                                       path.starts_with("/distribute/") || path.starts_with("/learn/") ||
                                       path.starts_with("/plugin/") || path.starts_with("/about/") ||
                                       path == "/start" || path == "/concept" || path == "/security" ||
                                       path == "/develop" || path == "/distribute" || path == "/learn" ||
                                       path == "/plugin" || path == "/about" ||
                                       path.starts_with("/docs/") || path.starts_with("/guide/")) &&
                                       url_obj.fragment().is_none() {
                                        to_visit.push(resolved_url);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to fetch {}: {}", url, e);
                }
            }
        }
        Ok(())
    }

    fn is_relevant_tauri_url(&self, url: &str, base_url: &str) -> bool {
        // Must be within the Tauri v2 documentation domain
        if !url.starts_with("https://v2.tauri.app") {
            return false;
        }
        
        // Exclude assets, downloads, and external links
        if url.contains("/assets/") || url.contains(".css") || url.contains(".js") ||
           url.contains(".png") || url.contains(".jpg") || url.contains(".svg") ||
           url.contains(".pdf") || url.contains("/img/") || url.contains("/images/") ||
           url.contains("://github.com") || url.contains("://discord.gg") {
            return false;
        }
        
        // Include all Tauri documentation content
        true
    }

    async fn fetch_react_docs(&self, source: &DocumentationSource) -> Result<()> {
        // React docs main sections
        let sections = vec![
            "learn",
            "reference/react",
            "reference/react-dom", 
            "reference/rules",
            "community",
        ];

        for section in sections {
            let url = format!("{}{}", source.base_url, section);
            if let Err(e) = self.fetch_section_pages(source, &url).await {
                warn!("Failed to fetch section {}: {}", section, e);
            }
        }

        Ok(())
    }

    pub async fn fetch_typescript_docs(&self, source: &DocumentationSource) -> Result<()> {
        // Use recursive crawling for comprehensive TypeScript documentation
        self.fetch_typescript_docs_recursive(source).await
    }

    pub async fn fetch_typescript_docs_recursive(&self, source: &DocumentationSource) -> Result<()> {
        let mut visited = HashSet::new();
        let mut to_visit = VecDeque::new();
        
        // Start with comprehensive seed URLs for TypeScript documentation
        // Include multiple versions and comprehensive documentation sections
        let starting_urls = vec![
            source.base_url.clone(), // Main docs index
            
            // Current handbook and docs
            format!("{}handbook/", source.base_url),
            format!("{}reference/", source.base_url),
            format!("{}declaration-files/", source.base_url),
            format!("{}project-config/", source.base_url),
            
            // Handbook - Core Language Features
            format!("{}handbook/intro.html", source.base_url),
            format!("{}handbook/basic-types.html", source.base_url),
            format!("{}handbook/everyday-types.html", source.base_url),
            format!("{}handbook/narrowing.html", source.base_url),
            format!("{}handbook/more-on-functions.html", source.base_url),
            format!("{}handbook/object-types.html", source.base_url),
            format!("{}handbook/type-manipulation/", source.base_url),
            format!("{}handbook/type-manipulation/creating-types-from-types.html", source.base_url),
            format!("{}handbook/type-manipulation/generics.html", source.base_url),
            format!("{}handbook/type-manipulation/keyof-types.html", source.base_url),
            format!("{}handbook/type-manipulation/typeof-types.html", source.base_url),
            format!("{}handbook/type-manipulation/indexed-access-types.html", source.base_url),
            format!("{}handbook/type-manipulation/conditional-types.html", source.base_url),
            format!("{}handbook/type-manipulation/mapped-types.html", source.base_url),
            format!("{}handbook/type-manipulation/template-literal-types.html", source.base_url),
            format!("{}handbook/classes.html", source.base_url),
            format!("{}handbook/modules.html", source.base_url),
            
            // Advanced and Modern Features
            format!("{}handbook/2/", source.base_url),
            format!("{}handbook/variable-declarations.html", source.base_url),
            format!("{}handbook/interfaces.html", source.base_url),
            format!("{}handbook/functions.html", source.base_url),
            format!("{}handbook/literal-types.html", source.base_url),
            format!("{}handbook/unions-and-intersections.html", source.base_url),
            format!("{}handbook/enums.html", source.base_url),
            format!("{}handbook/generics.html", source.base_url),
            format!("{}handbook/type-inference.html", source.base_url),
            format!("{}handbook/type-compatibility.html", source.base_url),
            format!("{}handbook/advanced-types.html", source.base_url),
            format!("{}handbook/utility-types.html", source.base_url),
            format!("{}handbook/decorators.html", source.base_url),
            format!("{}handbook/declaration-merging.html", source.base_url),
            format!("{}handbook/symbols.html", source.base_url),
            format!("{}handbook/iterators-and-generators.html", source.base_url),
            format!("{}handbook/jsx.html", source.base_url),
            format!("{}handbook/mixins.html", source.base_url),
            format!("{}handbook/namespaces.html", source.base_url),
            format!("{}handbook/namespaces-and-modules.html", source.base_url),
            format!("{}handbook/module-resolution.html", source.base_url),
            format!("{}handbook/triple-slash-directives.html", source.base_url),
            format!("{}handbook/type-checking-javascript-files.html", source.base_url),
            
            // Declaration Files - comprehensive coverage
            format!("{}declaration-files/introduction.html", source.base_url),
            format!("{}declaration-files/library-structures.html", source.base_url),
            format!("{}declaration-files/by-example.html", source.base_url),
            format!("{}declaration-files/do-s-and-don-ts.html", source.base_url),
            format!("{}declaration-files/deep-dive.html", source.base_url),
            format!("{}declaration-files/templates/", source.base_url),
            format!("{}declaration-files/templates/module-d-ts.html", source.base_url),
            format!("{}declaration-files/templates/module-class-d-ts.html", source.base_url),
            format!("{}declaration-files/templates/module-function-d-ts.html", source.base_url),
            format!("{}declaration-files/templates/global-d-ts.html", source.base_url),
            format!("{}declaration-files/templates/global-modifying-module-d-ts.html", source.base_url),
            format!("{}declaration-files/publishing.html", source.base_url),
            format!("{}declaration-files/consumption.html", source.base_url),
            
            // Project Configuration
            format!("{}project-config/tsconfig-json.html", source.base_url),
            format!("{}project-config/compiler-options.html", source.base_url),
            format!("{}project-config/integrating-with-build-tools.html", source.base_url),
            format!("{}project-config/nightly-builds.html", source.base_url),
            format!("{}project-config/project-references.html", source.base_url),
            
            // Reference Documentation
            format!("{}reference/utility-types.html", source.base_url),
            format!("{}reference/decorators.html", source.base_url),
            format!("{}reference/declaration-merging.html", source.base_url),
            format!("{}reference/iterators-and-generators.html", source.base_url),
            format!("{}reference/jsx.html", source.base_url),
            format!("{}reference/mixins.html", source.base_url),
            format!("{}reference/modules.html", source.base_url),
            format!("{}reference/module-resolution.html", source.base_url),
            format!("{}reference/namespaces.html", source.base_url),
            format!("{}reference/symbols.html", source.base_url),
            format!("{}reference/triple-slash-directives.html", source.base_url),
            format!("{}reference/type-compatibility.html", source.base_url),
            format!("{}reference/variable-declarations.html", source.base_url),
            
            // Tutorials and Getting Started
            format!("{}handbook/typescript-in-5-minutes.html", source.base_url),
            format!("{}handbook/typescript-in-5-minutes-oop.html", source.base_url),
            format!("{}handbook/typescript-in-5-minutes-func.html", source.base_url),
            format!("{}docs/bootstrap", source.base_url),
            
            // Release Notes and Changelogs - Multiple Versions
            "https://www.typescriptlang.org/docs/handbook/release-notes/".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-0.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-1.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-2.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-3.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-4.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-5.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-6.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-9.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-8.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-7.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-6.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-5.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-4.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-3.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-2.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-1.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-0.html".to_string(),
            
            // Community and Migration Guides
            "https://www.typescriptlang.org/docs/handbook/migrating-from-javascript.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/asp-net-core.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/gulp.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/knockout.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/react.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/angular.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/babel-with-typescript.html".to_string(),
            
            // Configuration and Tooling
            "https://www.typescriptlang.org/tsconfig".to_string(),
            "https://www.typescriptlang.org/docs/handbook/compiler-options-in-msbuild.html".to_string(),
            "https://www.typescriptlang.org/docs/handbook/configuring-watch.html".to_string(),
            
            // NPM Registry and Package Manager Documentation
            "https://docs.npmjs.com/".to_string(),
            "https://docs.npmjs.com/about-npm".to_string(),
            "https://docs.npmjs.com/getting-started".to_string(),
            "https://docs.npmjs.com/creating-and-publishing-unscoped-public-packages".to_string(),
            "https://docs.npmjs.com/creating-and-publishing-scoped-public-packages".to_string(),
            "https://docs.npmjs.com/package-json".to_string(),
            "https://docs.npmjs.com/cli/v10/commands/npm".to_string(),
            "https://docs.npmjs.com/cli/v10/commands/npm-install".to_string(),
            "https://docs.npmjs.com/cli/v10/commands/npm-publish".to_string(),
            "https://docs.npmjs.com/cli/v10/commands/npm-version".to_string(),
            "https://docs.npmjs.com/cli/v10/commands/npm-run-script".to_string(),
            "https://docs.npmjs.com/cli/v10/configuring-npm/package-json".to_string(),
            "https://docs.npmjs.com/about-semantic-versioning".to_string(),
            "https://docs.npmjs.com/about-packages-and-modules".to_string(),
            "https://docs.npmjs.com/downloading-and-installing-packages-locally".to_string(),
            "https://docs.npmjs.com/downloading-and-installing-packages-globally".to_string(),
            "https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally".to_string(),
            "https://docs.npmjs.com/using-npm-packages-in-your-projects".to_string(),
            "https://docs.npmjs.com/specifying-dependencies-and-devdependencies-in-a-package-json-file".to_string(),
        ];

        // Add all starting URLs to the queue with fragment normalization
        for url in starting_urls {
            if let Ok(parsed_url) = Url::parse(&url) {
                let mut normalized_url = parsed_url.clone();
                normalized_url.set_fragment(None);
                
                if visited.insert(normalized_url.to_string()) {
                    to_visit.push_back(url);
                }
            }
        }

        let base_url = Url::parse(&source.base_url)?;
        let base_domain = base_url.domain().unwrap_or("").to_string();
        let mut batch_urls = Vec::new();
        let batch_size = 25; // Smaller batches for TypeScript to be respectful

        // Breadth-first crawling with concurrent processing
        while let Some(current_url) = to_visit.pop_front() {
            batch_urls.push(current_url.clone());
            
            // Process in batches
            if batch_urls.len() >= batch_size || to_visit.is_empty() {
                // Fetch and store current batch
                if let Err(e) = self.fetch_and_store_pages_batch(source, batch_urls.clone()).await {
                    warn!("Failed to process batch for TypeScript docs: {}", e);
                }
                
                // Find new links from the current batch
                for url in &batch_urls {
                    if let Ok(html) = self.fetch_page(url).await {
                        let document = Html::parse_document(&html);
                        let link_selector = Selector::parse("a[href]").unwrap();
                        
                        for element in document.select(&link_selector) {
                            if let Some(href) = element.value().attr("href") {
                                // Resolve relative URLs
                                if let Ok(current_base) = Url::parse(url) {
                                    if let Ok(full_url) = current_base.join(href) {
                                        let url_str = full_url.to_string();
                                        
                                        // Filter for relevant TypeScript documentation
                                        if self.is_relevant_typescript_url(&url_str, &source.base_url) {
                                            let mut normalized_url = full_url.clone();
                                            normalized_url.set_fragment(None);
                                            
                                            if visited.insert(normalized_url.to_string()) {
                                                to_visit.push_back(url_str);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                batch_urls.clear();
                
                // Increased limit for comprehensive TypeScript documentation coverage
                if visited.len() > 1500 {
                    info!("Reached crawling limit for TypeScript docs");
                    break;
                }
            }
        }

        info!("TypeScript documentation crawl completed. Visited {} unique URLs.", visited.len());
        Ok(())
    }

    fn is_relevant_typescript_url(&self, url: &str, base_url: &str) -> bool {
        // Must be within the TypeScript documentation domain or npm docs
        if !url.starts_with("https://www.typescriptlang.org") && !url.starts_with("https://docs.npmjs.com") {
            return false;
        }
        
        // Exclude playground, search, and external links
        if url.contains("/play") || url.contains("/search") || 
           url.contains("://github.com") || url.contains("://stackoverflow.com") ||
           url.contains("/assets/") || url.contains(".css") || url.contains(".js") ||
           url.contains(".png") || url.contains(".jpg") || url.contains(".svg") ||
           url.contains(".pdf") || url.contains("/img/") || url.contains("/images/") {
            return false;
        }
        
        // Include comprehensive TypeScript documentation content
        url.contains("/docs/") || 
        url.contains("/handbook/") ||
        url.contains("/reference/") ||
        url.contains("/declaration-files/") ||
        url.contains("/project-config/") ||
        url.contains("/type-manipulation/") ||
        url.contains("/release-notes/") ||
        url.contains("/templates/") ||
        url.contains("/download") ||
        url.contains("/community") ||
        url.contains("/tsconfig") ||
        url.ends_with("/docs") ||
        url.ends_with("/handbook") ||
        url.ends_with("/reference") ||
        url.ends_with("/declaration-files") ||
        url.ends_with("/project-config") ||
        url.ends_with("/release-notes") ||
        // Include npm documentation for JavaScript/TypeScript ecosystem
        (url.starts_with("https://docs.npmjs.com") && (
            url.contains("/cli/") || url.contains("/commands/") ||
            url.contains("/configuring-") || url.contains("/about-") ||
            url.contains("/creating-") || url.contains("/downloading-") ||
            url.contains("/getting-") || url.contains("/package-") ||
            url.contains("/resolving-") || url.contains("/specifying-") ||
            url.contains("/using-") || url.ends_with("/docs.npmjs.com/") ||
            url.ends_with(".html")
        )) ||
        // Include version-specific documentation
        (url.contains("typescript") && (
            url.contains("-4-") || url.contains("-5-") || 
            url.contains("v4.") || url.contains("v5.") ||
            url.contains("4.0") || url.contains("4.1") || url.contains("4.2") ||
            url.contains("4.3") || url.contains("4.4") || url.contains("4.5") ||
            url.contains("4.6") || url.contains("4.7") || url.contains("4.8") ||
            url.contains("4.9") || url.contains("5.0") || url.contains("5.1") ||
            url.contains("5.2") || url.contains("5.3") || url.contains("5.4") ||
            url.contains("5.5") || url.contains("5.6")
        )) ||
        (url.ends_with(".html") && (
            url.contains("handbook") ||
            url.contains("reference") ||
            url.contains("declaration") ||
            url.contains("project") ||
            url.contains("tutorial") ||
            url.contains("guide") ||
            url.contains("getting-started")
        ))
    }

    pub async fn fetch_python_docs(&self, source: &DocumentationSource) -> Result<()> {
        // Use recursive crawling for Python docs starting from main sections
        self.fetch_python_docs_recursive(source).await
    }

    pub async fn fetch_python_docs_recursive(&self, source: &DocumentationSource) -> Result<()> {
        let mut visited = HashSet::new();
        let mut to_visit = VecDeque::new();
        
        // Start with key documentation sections that are known to exist
        let starting_urls = vec![
            format!("{}tutorial/index.html", source.base_url),
            format!("{}library/index.html", source.base_url),
            format!("{}reference/index.html", source.base_url),
            format!("{}using/index.html", source.base_url),
            format!("{}howto/index.html", source.base_url),
            format!("{}faq/index.html", source.base_url),
            format!("{}extending/index.html", source.base_url),
            format!("{}c-api/index.html", source.base_url),
            format!("{}install/index.html", source.base_url),
            format!("{}distributing/index.html", source.base_url),
            format!("{}glossary.html", source.base_url),
            source.base_url.clone(), // Main index page
            
            // Pip and Package Management Documentation
            "https://pip.pypa.io/en/stable/".to_string(),
            "https://pip.pypa.io/en/stable/getting-started/".to_string(),
            "https://pip.pypa.io/en/stable/user_guide/".to_string(),
            "https://pip.pypa.io/en/stable/reference/".to_string(),
            "https://pip.pypa.io/en/stable/installation/".to_string(),
            "https://pip.pypa.io/en/stable/cli/".to_string(),
            "https://pip.pypa.io/en/stable/topics/".to_string(),
            "https://pip.pypa.io/en/stable/development/".to_string(),
            "https://pip.pypa.io/en/stable/news/".to_string(),
            "https://pip.pypa.io/en/stable/cli/pip/".to_string(),
            "https://pip.pypa.io/en/stable/cli/pip_install/".to_string(),
            "https://pip.pypa.io/en/stable/cli/pip_uninstall/".to_string(),
            "https://pip.pypa.io/en/stable/cli/pip_list/".to_string(),
            "https://pip.pypa.io/en/stable/cli/pip_show/".to_string(),
            "https://pip.pypa.io/en/stable/cli/pip_freeze/".to_string(),
            "https://pip.pypa.io/en/stable/user_guide/#requirements-files".to_string(),
            "https://pip.pypa.io/en/stable/topics/dependency-resolution/".to_string(),
            "https://pip.pypa.io/en/stable/topics/configuration/".to_string(),
            "https://pip.pypa.io/en/stable/topics/caching/".to_string(),
            "https://pip.pypa.io/en/stable/topics/local-project-installs/".to_string(),
            "https://pip.pypa.io/en/stable/topics/vcs-support/".to_string(),
            
            // PyPI and Python Package Index
            "https://pypi.org/help/".to_string(),
            "https://pypi.org/help/#publishing".to_string(),
            "https://pypi.org/help/#packages".to_string(),
            "https://pypi.org/help/#searches".to_string(),
            "https://pypi.org/help/#feedback".to_string(),
            
            // Python Packaging Guide
            "https://packaging.python.org/".to_string(),
            "https://packaging.python.org/en/latest/tutorials/".to_string(),
            "https://packaging.python.org/en/latest/guides/".to_string(),
            "https://packaging.python.org/en/latest/tutorials/packaging-projects/".to_string(),
            "https://packaging.python.org/en/latest/tutorials/installing-packages/".to_string(),
            "https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/".to_string(),
            "https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/".to_string(),
            "https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/".to_string(),
            "https://packaging.python.org/en/latest/specifications/".to_string(),
        ];

        // Add all starting URLs to visit queue
        for url in starting_urls {
            let mut normalized_url = Url::parse(&url).map_err(|e| anyhow::anyhow!("Invalid URL {}: {}", url, e))?;
            normalized_url.set_fragment(None); // Remove fragments to prevent infinite loops
            if visited.insert(normalized_url.to_string()) {
                to_visit.push_back(normalized_url.to_string());
            }
        }

        // Process URLs in concurrent batches
        while !to_visit.is_empty() {
            let batch_size = 25.min(to_visit.len());
            let batch: Vec<String> = (0..batch_size).map(|_| to_visit.pop_front().unwrap()).collect();
            
            info!("Processing batch of {} URLs for Python Documentation", batch.len());
            let results = self.fetch_pages_batch_with_content(batch).await;
            
            // Store the pages and process successful results to discover new URLs
            let mut stored_count = 0;
            for (url, html_opt) in &results {
                if html_opt.is_some() {
                    if let Err(e) = self.fetch_and_store_page(source, url).await {
                        warn!("Failed to store page {}: {}", url, e);
                    } else {
                        stored_count += 1;
                        info!("Stored page: {}", url);
                    }
                }
            }
            info!("Successfully processed and stored {}/{} pages for Python Documentation", stored_count, results.len());
            
            for (url, html_opt) in results {
                if let Some(html) = html_opt {
                    let document = Html::parse_document(&html);
                    let link_selector = Selector::parse("a[href]").unwrap();

                    for element in document.select(&link_selector) {
                        if let Some(href) = element.value().attr("href") {
                            if let Ok(resolved_url) = self.resolve_url(&url, href) {
                                if self.is_relevant_python_url(&resolved_url, &source.base_url) {
                                    let mut normalized_url = Url::parse(&resolved_url)?;
                                    normalized_url.set_fragment(None);
                                    let normalized = normalized_url.to_string();
                                    
                                    if visited.insert(normalized.clone()) {
                                        to_visit.push_back(normalized);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Limit crawling to prevent infinite loops
            if visited.len() > 2000 {
                info!("Reached crawling limit for Python docs");
                break;
            }

            // Rate limiting - small delay between batches
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        info!("Python documentation crawl completed. Visited {} unique URLs.", visited.len());
        Ok(())
    }

    fn is_relevant_python_url(&self, url: &str, base_url: &str) -> bool {
        // Must be within Python documentation domains or pip/packaging domains
        if !url.starts_with("https://docs.python.org") && 
           !url.starts_with("https://pip.pypa.io") && 
           !url.starts_with("https://packaging.python.org") &&
           !url.starts_with("https://pypi.org") {
            return false;
        }
        
        // Exclude assets and non-documentation paths
        if url.contains("/assets/") || url.contains(".css") || url.contains(".js") ||
           url.contains(".png") || url.contains(".jpg") || url.contains(".svg") ||
           url.contains(".pdf") || url.contains("/img/") || url.contains("/images/") ||
           url.contains("/_static/") || url.contains("/_sources/") {
            return false;
        }
        
        // Include Python 3.x documentation
        if url.starts_with("https://docs.python.org") {
            return url.contains("/3/") && !url.contains("/_") && !url.contains("/download");
        }
        
        // Include pip documentation
        if url.starts_with("https://pip.pypa.io") {
            return url.contains("/en/stable/") || url.contains("/en/latest/") || url.ends_with("/pip.pypa.io/");
        }
        
        // Include packaging documentation
        if url.starts_with("https://packaging.python.org") {
            return !url.contains("/_") && (
                url.contains("/tutorials/") || url.contains("/guides/") ||
                url.contains("/specifications/") || url.contains("/en/latest/") ||
                url.ends_with("/packaging.python.org/")
            );
        }
        
        // Include PyPI help pages
        if url.starts_with("https://pypi.org") {
            return url.contains("/help/") || url.ends_with("/help");
        }
        
        false
    }

    async fn fetch_tailwind_docs(&self, source: &DocumentationSource) -> Result<()> {
        // Tailwind CSS v4+ documentation sections
        let sections = vec![
            "installation",
            "editor-setup",
            "using-with-preprocessors",
            "optimizing-for-production",
            "browser-support",
            "upgrade-guide",
            // Core Concepts
            "utility-first",
            "hover-focus-and-other-states",
            "responsive-design",
            "dark-mode",
            "adding-custom-styles",
            "functions-and-directives",
            // Layout
            "aspect-ratio",
            "container",
            "columns",
            "break-after",
            "break-before",
            "break-inside",
            "box-decoration-break",
            "box-sizing",
            "display",
            "floats",
            "clear",
            "isolation",
            "object-fit",
            "object-position",
            "overflow",
            "overscroll-behavior",
            "position",
            "top-right-bottom-left",
            "visibility",
            "z-index",
            // Flexbox & Grid
            "flex-basis",
            "flex-direction",
            "flex-wrap",
            "flex",
            "flex-grow",
            "flex-shrink",
            "order",
            "grid-template-columns",
            "grid-column",
            "grid-template-rows",
            "grid-row",
            "grid-auto-flow",
            "grid-auto-columns",
            "grid-auto-rows",
            "gap",
            "justify-content",
            "justify-items",
            "justify-self",
            "align-content",
            "align-items",
            "align-self",
            "place-content",
            "place-items",
            "place-self",
            // Spacing
            "padding",
            "margin",
            "space",
            // Sizing
            "width",
            "min-width",
            "max-width",
            "height",
            "min-height",
            "max-height",
            "size",
            // Typography
            "font-family",
            "font-size",
            "font-smoothing",
            "font-style",
            "font-weight",
            "font-variant-numeric",
            "letter-spacing",
            "line-clamp",
            "line-height",
            "list-style-image",
            "list-style-position",
            "list-style-type",
            "text-align",
            "text-color",
            "text-decoration",
            "text-decoration-color",
            "text-decoration-style",
            "text-decoration-thickness",
            "text-underline-offset",
            "text-transform",
            "text-overflow",
            "text-wrap",
            "text-indent",
            "vertical-align",
            "whitespace",
            "word-break",
            "hyphens",
            "content",
            // Backgrounds
            "background-attachment",
            "background-clip",
            "background-color",
            "background-origin",
            "background-position",
            "background-repeat",
            "background-size",
            "background-image",
            "gradient-color-stops",
            // Borders
            "border-radius",
            "border-width",
            "border-color",
            "border-style",
            "divide-width",
            "divide-color",
            "divide-style",
            "outline-width",
            "outline-color",
            "outline-style",
            "outline-offset",
            "ring-width",
            "ring-color",
            "ring-opacity",
            "ring-offset-width",
            "ring-offset-color",
            // Effects
            "box-shadow",
            "box-shadow-color",
            "opacity",
            "mix-blend-mode",
            "background-blend-mode",
            // Filters
            "blur",
            "brightness",
            "contrast",
            "drop-shadow",
            "grayscale",
            "hue-rotate",
            "invert",
            "saturate",
            "sepia",
            "backdrop-blur",
            "backdrop-brightness",
            "backdrop-contrast",
            "backdrop-grayscale",
            "backdrop-hue-rotate",
            "backdrop-invert",
            "backdrop-opacity",
            "backdrop-saturate",
            "backdrop-sepia",
            // Tables
            "border-collapse",
            "border-spacing",
            "table-layout",
            "caption-side",
            // Transitions & Animation
            "transition-property",
            "transition-duration",
            "transition-timing-function",
            "transition-delay",
            "animation",
            // Transforms
            "scale",
            "rotate",
            "translate",
            "skew",
            "transform-origin",
            // Interactivity
            "accent-color",
            "appearance",
            "cursor",
            "caret-color",
            "pointer-events",
            "resize",
            "scroll-behavior",
            "scroll-margin",
            "scroll-padding",
            "scroll-snap-align",
            "scroll-snap-stop",
            "scroll-snap-type",
            "touch-action",
            "user-select",
            "will-change",
            // SVG
            "fill",
            "stroke",
            "stroke-width",
            // Accessibility
            "screen-readers",
            // Official Plugins
            "typography",
            "forms",
            "aspect-ratio",
            "container-queries",
        ];

        for section in sections {
            let url = format!("{}{}", source.base_url, section);
            if let Err(e) = self.fetch_and_store_page(source, &url).await {
                warn!("Failed to fetch Tailwind section {}: {}", section, e);
            }
        }

        Ok(())
    }

    async fn fetch_shadcn_docs(&self, source: &DocumentationSource) -> Result<()> {
        // shadcn/ui documentation sections
        let sections = vec![
            // Introduction
            "introduction",
            "installation",
            "installation/next",
            "installation/vite",
            "installation/remix",
            "installation/gatsby",
            "installation/astro",
            "installation/laravel",
            "installation/manual",
            // Components Documentation
            "components/accordion",
            "components/alert",
            "components/alert-dialog",
            "components/aspect-ratio",
            "components/avatar",
            "components/badge",
            "components/breadcrumb",
            "components/button",
            "components/calendar",
            "components/card",
            "components/carousel",
            "components/chart",
            "components/checkbox",
            "components/collapsible",
            "components/combobox",
            "components/command",
            "components/context-menu",
            "components/data-table",
            "components/date-picker",
            "components/dialog",
            "components/drawer",
            "components/dropdown-menu",
            "components/form",
            "components/hover-card",
            "components/input",
            "components/input-otp",
            "components/label",
            "components/menubar",
            "components/navigation-menu",
            "components/pagination",
            "components/popover",
            "components/progress",
            "components/radio-group",
            "components/resizable",
            "components/scroll-area",
            "components/select",
            "components/separator",
            "components/sheet",
            "components/skeleton",
            "components/slider",
            "components/sonner",
            "components/switch",
            "components/table",
            "components/tabs",
            "components/textarea",
            "components/toast",
            "components/toggle",
            "components/toggle-group",
            "components/tooltip",
            // Dark Mode
            "dark-mode",
            "dark-mode/next",
            "dark-mode/vite",
            "dark-mode/astro",
            // CLI
            "cli",
            // Theming
            "theming",
            // Typography
            "components/typography",
            // Customization
            "installation/manual",
            // Examples
            "examples/mail",
            "examples/dashboard",
            "examples/cards",
            "examples/tasks",
            "examples/playground",
            "examples/forms",
            "examples/music",
            "examples/authentication",
        ];

        for section in sections {
            let url = format!("{}{}", source.base_url, section);
            if let Err(e) = self.fetch_and_store_page(source, &url).await {
                warn!("Failed to fetch shadcn/ui section {}: {}", section, e);
            }
        }

        // Also fetch component examples and source code
        let component_pages = vec![
            "components",
            "examples",
            "blocks",
        ];

        for page in component_pages {
            let url = format!("{}{}", source.base_url, page);
            if let Err(e) = self.fetch_section_pages(source, &url).await {
                warn!("Failed to fetch shadcn/ui page {}: {}", page, e);
            }
        }

        Ok(())
    }

    async fn fetch_section_pages(&self, source: &DocumentationSource, section_url: &str) -> Result<()> {
        let html = self.fetch_page(section_url).await?;
        let document = Html::parse_document(&html);
        
        // Look for links within the same domain
        let link_selector = Selector::parse("a[href]").unwrap();
        let base_url = Url::parse(&source.base_url)?;
        let base_domain = base_url.domain().unwrap_or("").to_string();
        let mut visited = HashSet::new();
        
        // Collect URLs first to avoid borrowing issues
        let mut urls_to_fetch = Vec::new();
        for element in document.select(&link_selector) {
            if let Some(href) = element.value().attr("href") {
                // Properly resolve URLs - handle absolute vs relative links
                let resolved_url = if href.starts_with("http://") || href.starts_with("https://") {
                    // Already absolute URL
                    href.to_string()
                } else if href.starts_with('/') {
                    // Absolute path relative to domain
                    format!("{}{}", base_url.origin().ascii_serialization(), href)
                } else {
                    // Relative path
                    base_url.join(href).map(|u| u.to_string()).unwrap_or_else(|_| format!("{}/{}", source.base_url.trim_end_matches('/'), href))
                };
                
                if let Ok(url) = Url::parse(&resolved_url) {
                    if url.domain() == Some(&base_domain) && visited.insert(url.to_string()) {
                        urls_to_fetch.push(url.to_string());
                    }
                }
            }
        }
        
        // Now fetch the URLs
        for url in urls_to_fetch {
            if let Err(e) = self.fetch_and_store_page(source, &url).await {
                warn!("Failed to fetch {}: {}", url, e);
            }
        }

        Ok(())
    }

    async fn fetch_page(&self, url: &str) -> Result<String> {
        let response = self.client.get(url).send().await?;
        if response.status().is_success() {
            Ok(response.text().await?)
        } else {
            Err(anyhow!("HTTP error {}: {}", response.status(), url))
        }
    }

    async fn fetch_and_store_page(&self, source: &DocumentationSource, url: &str) -> Result<()> {
        let html = self.fetch_page(url).await?;
        let document = Html::parse_document(&html);
        
        // Extract title
        let title_selector = Selector::parse("title, h1").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .unwrap_or_else(|| "Untitled".to_string());

        // Extract main content
        let content_selectors = vec![
            "main",
            ".content",
            ".documentation",
            ".docs-content", 
            "article",
            ".main-content",
            "#content",
        ];

        let mut content = String::new();
        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    content = element.text().collect::<Vec<_>>().join(" ");
                    break;
                }
            }
        }

        // If no specific content area found, use body
        if content.is_empty() {
            if let Ok(selector) = Selector::parse("body") {
                if let Some(element) = document.select(&selector).next() {
                    content = element.text().collect::<Vec<_>>().join(" ");
                }
            }
        }

        // Convert HTML to markdown for better readability
        let markdown_content = html2md::parse_html(&html);

        // Extract path from URL
        let url_obj = Url::parse(url)?;
        let path = url_obj.path().to_string();

        let doc_page = DocumentPage {
            id: format!("{}:{}", source.id, path),
            source_id: source.id.clone(),
            title,
            url: url.to_string(),
            content: content.trim().to_string(),
            markdown_content,
            last_updated: Utc::now(),
            path,
            section: None,
        };

        self.db.add_document(&doc_page).await?;
        info!("Stored page: {}", doc_page.title);

        Ok(())
    }

    /// Check if source content has changed (for incremental updates)
    pub async fn has_source_changed(&self, source: &DocumentationSource) -> Result<bool> {
        // Check last-modified headers, version numbers, etc.
        let response = self.client
            .head(&source.base_url)
            .send()
            .await?;

        if let Some(last_modified) = response.headers().get("last-modified") {
            if let Ok(last_modified_str) = last_modified.to_str() {
                // Parse the date and compare with our last update
                if let Some(our_last_update) = source.last_updated {
                    // Simple heuristic - in a real implementation, you'd parse the HTTP date
                    return Ok(last_modified_str.len() > 0 && 
                             Utc::now() - our_last_update > chrono::Duration::hours(1));
                }
            }
        }

        // Default to assuming change if we can't determine
        Ok(true)
    }

    /// Get content fingerprint for change detection
    async fn get_content_fingerprint(&self, url: &str) -> Result<String> {
        let response = self.client.get(url).send().await?;
        let content = response.text().await?;
        
        // Create a simple hash of the content
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        Ok(format!("{:x}", hasher.finish()))
    }

    /// Fetch multiple URLs concurrently with rate limiting
    async fn fetch_urls_concurrently(&self, source: &DocumentationSource, urls: Vec<String>) -> Result<Vec<String>> {
        let mut handles = Vec::new();
        let successful_urls = Arc::new(Mutex::new(Vec::new()));
        
        for url in urls {
            let client = self.client.clone();
            let semaphore = Arc::clone(&self.semaphore);
            let successful_urls = Arc::clone(&successful_urls);
            let source_id = source.id.clone();
            
            let handle = tokio::spawn(async move {
                // Acquire semaphore permit to limit concurrency
                let _permit = semaphore.acquire().await.unwrap();
                
                // Small delay to be respectful to servers
                sleep(Duration::from_millis(100)).await;
                
                match client.get(&url).send().await {
                    Ok(response) if response.status().is_success() => {
                        match response.text().await {
                            Ok(html) => {
                                let mut urls = successful_urls.lock().await;
                                urls.push(url.clone());
                                Some((url, html))
                            }
                            Err(e) => {
                                warn!("Failed to read response from {}: {}", url, e);
                                None
                            }
                        }
                    }
                    Ok(response) => {
                        warn!("HTTP error {} for URL: {}", response.status(), url);
                        None
                    }
                    Err(e) => {
                        warn!("Failed to fetch {}: {}", url, e);
                        None
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all requests to complete
        let results = futures::future::join_all(handles).await;
        let mut all_content = Vec::new();
        
        for result in results {
            if let Ok(Some((url, html))) = result {
                all_content.push(html);
            }
        }
        
        let successful_urls = successful_urls.lock().await;
        info!("Successfully fetched {}/{} URLs for {}", successful_urls.len(), all_content.len(), source.name);
        
        Ok(all_content)
    }

    /// Process and store pages concurrently
    async fn process_and_store_pages_concurrently(&self, source: &DocumentationSource, url_html_pairs: Vec<(String, String)>) -> Result<()> {
        let mut handles = Vec::new();
        let db = self.db.clone();
        
        for (url, html) in url_html_pairs {
            let source_clone = source.clone();
            let db_clone = db.clone();
            
            let handle = tokio::spawn(async move {
                match Self::parse_and_create_doc_page(&source_clone, &url, &html) {
                    Ok(doc_page) => {
                        match db_clone.add_document(&doc_page).await {
                            Ok(_) => {
                                info!("Stored page: {}", doc_page.title);
                                Some(doc_page.title)
                            }
                            Err(e) => {
                                error!("Failed to store page {}: {}", url, e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to parse page {}: {}", url, e);
                        None
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all storage operations to complete
        let results = futures::future::join_all(handles).await;
        let successful_count = results.iter().filter(|r| r.is_ok() && r.as_ref().unwrap().is_some()).count();
        
        info!("Successfully processed and stored {}/{} pages for {}", successful_count, results.len(), source.name);
        Ok(())
    }

    /// Parse HTML content and create DocumentPage (static method for easy spawning)
    fn parse_and_create_doc_page(source: &DocumentationSource, url: &str, html: &str) -> Result<DocumentPage> {
        let document = Html::parse_document(html);
        
        // Extract title
        let title_selector = Selector::parse("title, h1").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .unwrap_or_else(|| "Untitled".to_string());

        // Extract main content
        let content_selectors = vec![
            "main",
            ".content",
            ".documentation",
            ".docs-content", 
            "article",
            ".main-content",
            "#content",
        ];

        let mut content = String::new();
        for selector_str in content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    content = element.text().collect::<Vec<_>>().join(" ");
                    break;
                }
            }
        }

        // If no specific content area found, use body
        if content.is_empty() {
            if let Ok(selector) = Selector::parse("body") {
                if let Some(element) = document.select(&selector).next() {
                    content = element.text().collect::<Vec<_>>().join(" ");
                }
            }
        }

        // Convert HTML to markdown for better readability
        let markdown_content = html2md::parse_html(html);

        // Extract path from URL
        let url_obj = Url::parse(url)?;
        let path = url_obj.path().to_string();

        Ok(DocumentPage {
            id: format!("{}:{}", source.id, path),
            source_id: source.id.clone(),
            title,
            url: url.to_string(),
            content: content.trim().to_string(),
            markdown_content,
            last_updated: Utc::now(),
            path,
            section: None,
        })
    }

    /// Concurrent version of fetch_and_store_page for batch operations
    async fn fetch_and_store_pages_batch(&self, source: &DocumentationSource, urls: Vec<String>) -> Result<()> {
        if urls.is_empty() {
            return Ok(());
        }

        info!("Processing batch of {} URLs for {}", urls.len(), source.name);
        
        // Fetch all URLs concurrently
        let mut url_html_pairs = Vec::new();
        let batch_size = self.batch_size.min(urls.len());
        
        for chunk in urls.chunks(batch_size) {
            let chunk_vec = chunk.to_vec();
            let mut handles = Vec::new();
            
            for url in chunk_vec {
                let client = self.client.clone();
                let semaphore = Arc::clone(&self.semaphore);
                let url_clone = url.clone();
                
                let handle = tokio::spawn(async move {
                    // Acquire semaphore permit to limit concurrency
                    let _permit = semaphore.acquire().await.unwrap();
                    
                    // Small delay to be respectful to servers
                    sleep(Duration::from_millis(50)).await;
                    
                    match client.get(&url_clone).send().await {
                        Ok(response) if response.status().is_success() => {
                            match response.text().await {
                                Ok(html) => Some((url_clone, html)),
                                Err(e) => {
                                    warn!("Failed to read response from {}: {}", url_clone, e);
                                    None
                                }
                            }
                        }
                        Ok(response) => {
                            warn!("HTTP error {} for URL: {}", response.status(), url_clone);
                            None
                        }
                        Err(e) => {
                            warn!("Request failed for {}: {}", url_clone, e);
                            None
                        }
                    }
                });
                handles.push(handle);
            }
            
            // Wait for this chunk to complete
            let chunk_results = futures::future::join_all(handles).await;
            for result in chunk_results {
                if let Ok(Some(pair)) = result {
                    url_html_pairs.push(pair);
                }
            }
        }
        
        // Process and store all fetched pages
        self.process_and_store_pages_concurrently(source, url_html_pairs).await?;
        
        Ok(())
    }

    /// Fetch URLs and return (URL, HTML) pairs for link discovery
    async fn fetch_pages_batch_with_content(&self, urls: Vec<String>) -> Vec<(String, Option<String>)> {
        let mut results = Vec::new();
        let batch_size = self.batch_size.min(urls.len());
        
        for chunk in urls.chunks(batch_size) {
            let chunk_vec = chunk.to_vec();
            let mut handles = Vec::new();
            
            for url in chunk_vec {
                let client = self.client.clone();
                let semaphore = Arc::clone(&self.semaphore);
                let url_clone = url.clone();
                
                let handle = tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.unwrap();
                    sleep(Duration::from_millis(50)).await;
                    
                    match client.get(&url_clone).send().await {
                        Ok(response) if response.status().is_success() => {
                            match response.text().await {
                                Ok(html) => (url_clone, Some(html)),
                                Err(e) => {
                                    warn!("Failed to read response from {}: {}", url_clone, e);
                                    (url_clone, None)
                                }
                            }
                        }
                        Ok(response) => {
                            warn!("HTTP error {} for URL: {}", response.status(), url_clone);
                            (url_clone, None)
                        }
                        Err(e) => {
                            warn!("Request failed for {}: {}", url_clone, e);
                            (url_clone, None)
                        }
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                if let Ok(result) = handle.await {
                    results.push(result);
                }
            }
            
            sleep(Duration::from_millis(100)).await;
        }
        
        results
    }

    /// Resolve a URL relative to a base URL
    fn resolve_url(&self, base_url: &str, href: &str) -> Result<String> {
        if href.starts_with("http://") || href.starts_with("https://") {
            Ok(href.to_string())
        } else if href.starts_with('#') || href.starts_with("mailto:") || href.starts_with("javascript:") {
            Err(anyhow::anyhow!("Skipping anchor/mailto/javascript URL"))
        } else {
            let base = Url::parse(base_url)?;
            let resolved = base.join(href)?;
            Ok(resolved.to_string())
        }
    }
}
