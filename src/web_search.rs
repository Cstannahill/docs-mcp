use anyhow::Result;
use serde::{Deserialize, Serialize};
use reqwest;
use scraper::{Html, Selector};
use url::Url;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tokio::time::{sleep, Duration};

#[derive(Clone)]
pub struct WebSearchEngine {
    client: reqwest::Client,
    rate_limit_delay: Duration,
    max_results: usize,
    timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub max_results: Option<usize>,
    pub search_type: SearchType,
    pub filters: SearchFilters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    General,
    Programming,
    Documentation,
    News,
    Academic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    pub site: Option<String>,
    pub file_type: Option<String>,
    pub date_range: Option<DateRange>,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub description: String,
    pub domain: String,
    pub relevance_score: f32,
    pub content_preview: Option<String>,
    pub last_updated: Option<DateTime<Utc>>,
    pub search_type: SearchType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchResponse {
    pub query: String,
    pub results: Vec<SearchResult>,
    pub total_found: usize,
    pub search_time_ms: u64,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebPageContent {
    pub url: String,
    pub title: String,
    pub content: String,
    pub text_content: String,
    pub links: Vec<Link>,
    pub meta_description: Option<String>,
    pub keywords: Vec<String>,
    pub last_modified: Option<DateTime<Utc>>,
    pub content_type: String,
    pub word_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub url: String,
    pub text: String,
    pub is_external: bool,
}

impl WebSearchEngine {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; DocsBot/1.0; Educational Documentation Assistant)")
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            rate_limit_delay: Duration::from_millis(1000), // 1 second between requests
            max_results: 20,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_rate_limit(mut self, delay_ms: u64) -> Self {
        self.rate_limit_delay = Duration::from_millis(delay_ms);
        self
    }

    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_results = max;
        self
    }

    /// Perform a web search using DuckDuckGo (privacy-focused, no API key required)
    pub async fn search(&self, request: SearchRequest) -> Result<WebSearchResponse> {
        let start_time = std::time::Instant::now();
        
        // Build search query with filters
        let search_query = self.build_search_query(&request);
        
        // Perform the search
        let results = self.search_duckduckgo(&search_query, request.max_results.unwrap_or(self.max_results)).await?;
        
        // Generate suggestions based on query
        let suggestions = self.generate_search_suggestions(&request.query);
        
        let search_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(WebSearchResponse {
            query: request.query,
            total_found: results.len(),
            results,
            search_time_ms,
            suggestions,
        })
    }

    /// Search specifically for programming and development resources
    pub async fn search_programming(&self, query: &str, language: Option<&str>) -> Result<WebSearchResponse> {
        let mut programming_query = format!("{} programming", query);
        
        if let Some(lang) = language {
            programming_query = format!("{} {} programming", query, lang);
        }

        // Add programming-specific sites
        programming_query.push_str(" site:stackoverflow.com OR site:github.com OR site:docs.rs OR site:developer.mozilla.org");

        let request = SearchRequest {
            query: programming_query,
            max_results: Some(15),
            search_type: SearchType::Programming,
            filters: SearchFilters {
                site: None,
                file_type: None,
                date_range: None,
                language: language.map(|s| s.to_string()),
            },
        };

        self.search(request).await
    }

    /// Search for documentation specifically
    pub async fn search_documentation(&self, query: &str, technology: Option<&str>) -> Result<WebSearchResponse> {
        let mut doc_query = format!("{} documentation", query);
        
        if let Some(tech) = technology {
            doc_query = format!("{} {} documentation", query, tech);
        }

        // Add documentation-specific sites
        doc_query.push_str(" site:docs.rs OR site:doc.rust-lang.org OR site:typescriptlang.org OR site:docs.python.org OR site:reactjs.org");

        let request = SearchRequest {
            query: doc_query,
            max_results: Some(10),
            search_type: SearchType::Documentation,
            filters: SearchFilters {
                site: None,
                file_type: None,
                date_range: None,
                language: None,
            },
        };

        self.search(request).await
    }

    /// Fetch and parse content from a webpage
    pub async fn fetch_page_content(&self, url: &str) -> Result<WebPageContent> {
        // Rate limiting
        sleep(self.rate_limit_delay).await;

        let response = self.client
            .get(url)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
        }

        let content = response.text().await?;
        let parsed = self.parse_html_content(url, &content)?;

        Ok(parsed)
    }

    /// Get page summary/preview
    pub async fn get_page_summary(&self, url: &str, max_words: usize) -> Result<String> {
        let page_content = self.fetch_page_content(url).await?;
        
        let words: Vec<&str> = page_content.text_content.split_whitespace().collect();
        let summary_words = words.into_iter().take(max_words).collect::<Vec<_>>().join(" ");
        
        Ok(summary_words)
    }

    /// Search for recent news/updates about a topic
    pub async fn search_news(&self, query: &str, days_back: u32) -> Result<WebSearchResponse> {
        let news_query = format!("{} news", query);
        
        let request = SearchRequest {
            query: news_query,
            max_results: Some(10),
            search_type: SearchType::News,
            filters: SearchFilters {
                site: None,
                file_type: None,
                date_range: Some(DateRange {
                    start: Utc::now() - chrono::Duration::days(days_back as i64),
                    end: Utc::now(),
                }),
                language: None,
            },
        };

        self.search(request).await
    }

    // Private helper methods

    async fn search_duckduckgo(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        sleep(self.rate_limit_delay).await;

        // DuckDuckGo HTML search (simplified approach)
        let search_url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding::encode(query));
        
        let response = self.client
            .get(&search_url)
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Search request failed: {}", response.status()));
        }

        let html = response.text().await?;
        let parsed_results = self.parse_duckduckgo_results(&html)?;

        Ok(parsed_results.into_iter().take(max_results).collect())
    }

    fn parse_duckduckgo_results(&self, html: &str) -> Result<Vec<SearchResult>> {
        let document = Html::parse_document(html);
        let result_selector = Selector::parse(".result").unwrap();
        let title_selector = Selector::parse(".result__title a").unwrap();
        let url_selector = Selector::parse(".result__url").unwrap();
        let snippet_selector = Selector::parse(".result__snippet").unwrap();

        let mut results = Vec::new();

        for result in document.select(&result_selector) {
            let title = result
                .select(&title_selector)
                .next()
                .map(|el| el.text().collect::<String>())
                .unwrap_or_default();

            let url = result
                .select(&title_selector)
                .next()
                .and_then(|el| el.value().attr("href"))
                .unwrap_or_default()
                .to_string();

            let description = result
                .select(&snippet_selector)
                .next()
                .map(|el| el.text().collect::<String>())
                .unwrap_or_default();

            if !title.is_empty() && !url.is_empty() {
                let domain = Url::parse(&url)
                    .map(|u| u.domain().unwrap_or("unknown").to_string())
                    .unwrap_or_else(|_| "unknown".to_string());

                results.push(SearchResult {
                    title,
                    url,
                    description,
                    domain,
                    relevance_score: 1.0, // Simplified scoring
                    content_preview: None,
                    last_updated: None,
                    search_type: SearchType::General,
                });
            }
        }

        Ok(results)
    }

    fn parse_html_content(&self, url: &str, html: &str) -> Result<WebPageContent> {
        let document = Html::parse_document(html);
        
        // Extract title
        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_else(|| "Untitled".to_string());

        // Extract meta description
        let meta_desc_selector = Selector::parse("meta[name='description']").unwrap();
        let meta_description = document
            .select(&meta_desc_selector)
            .next()
            .and_then(|el| el.value().attr("content"))
            .map(|s| s.to_string());

        // Extract main content (try to find article, main, or body content)
        let content_selectors = [
            "article", "main", ".content", "#content", 
            ".post", ".entry", "body"
        ];
        
        let mut text_content = String::new();
        for selector_str in &content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    text_content = element.text().collect::<Vec<_>>().join(" ");
                    break;
                }
            }
        }

        // If no specific content found, extract from body
        if text_content.is_empty() {
            let body_selector = Selector::parse("body").unwrap();
            text_content = document
                .select(&body_selector)
                .next()
                .map(|el| el.text().collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
        }

        // Extract links
        let link_selector = Selector::parse("a[href]").unwrap();
        let mut links = Vec::new();
        let base_url = Url::parse(url)?;

        for link_element in document.select(&link_selector) {
            if let Some(href) = link_element.value().attr("href") {
                let link_text = link_element.text().collect::<String>();
                let resolved_url = base_url.join(href).unwrap_or_else(|_| base_url.clone());
                let is_external = resolved_url.domain() != base_url.domain();

                links.push(Link {
                    url: resolved_url.to_string(),
                    text: link_text,
                    is_external,
                });
            }
        }

        // Extract keywords from content
        let keywords = self.extract_keywords(&text_content);
        let word_count = text_content.split_whitespace().count();

        Ok(WebPageContent {
            url: url.to_string(),
            title,
            content: html.to_string(),
            text_content,
            links,
            meta_description,
            keywords,
            last_modified: None, // Could be extracted from headers
            content_type: "text/html".to_string(),
            word_count,
        })
    }

    fn build_search_query(&self, request: &SearchRequest) -> String {
        let mut query = request.query.clone();

        // Add site filter
        if let Some(site) = &request.filters.site {
            query.push_str(&format!(" site:{}", site));
        }

        // Add file type filter
        if let Some(file_type) = &request.filters.file_type {
            query.push_str(&format!(" filetype:{}", file_type));
        }

        // Add search type specific terms
        match request.search_type {
            SearchType::Programming => {
                query.push_str(" programming code tutorial");
            }
            SearchType::Documentation => {
                query.push_str(" documentation docs guide");
            }
            SearchType::Academic => {
                query.push_str(" research paper academic");
            }
            SearchType::News => {
                query.push_str(" news update recent");
            }
            SearchType::General => {}
        }

        query
    }

    fn generate_search_suggestions(&self, query: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Add common programming-related suggestions
        if query.to_lowercase().contains("rust") {
            suggestions.extend(vec![
                "rust programming tutorial".to_string(),
                "rust async programming".to_string(),
                "rust cargo guide".to_string(),
            ]);
        }
        
        if query.to_lowercase().contains("python") {
            suggestions.extend(vec![
                "python tutorial beginners".to_string(),
                "python web development".to_string(),
                "python data science".to_string(),
            ]);
        }

        if query.to_lowercase().contains("javascript") || query.to_lowercase().contains("typescript") {
            suggestions.extend(vec![
                "javascript tutorial".to_string(),
                "typescript documentation".to_string(),
                "react javascript framework".to_string(),
            ]);
        }

        // Add generic suggestions
        suggestions.extend(vec![
            format!("{} tutorial", query),
            format!("{} documentation", query),
            format!("{} examples", query),
        ]);

        suggestions.into_iter().take(5).collect()
    }

    fn extract_keywords(&self, text: &str) -> Vec<String> {
        // Simple keyword extraction based on frequency and length
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        
        // Clean and split text
        let text_lower = text.to_lowercase();
        let words = text_lower
            .split_whitespace()
            .filter(|word| word.len() > 3 && word.chars().all(|c| c.is_alphabetic()))
            .map(|word| word.to_string());

        for word in words {
            *word_freq.entry(word).or_insert(0) += 1;
        }

        // Sort by frequency and take top keywords
        let mut keywords: Vec<(String, usize)> = word_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        
        keywords.into_iter()
            .take(10)
            .map(|(word, _)| word)
            .collect()
    }
}

impl Default for WebSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_engine_creation() {
        let engine = WebSearchEngine::new();
        assert_eq!(engine.max_results, 20);
    }

    #[test]
    fn test_keyword_extraction() {
        let engine = WebSearchEngine::new();
        let text = "rust programming language systems programming memory safety";
        let keywords = engine.extract_keywords(text);
        assert!(keywords.contains(&"programming".to_string()));
    }
}
