use docs_mcp_server::database::{Database, DocumentationSource, DocType};
use docs_mcp_server::fetcher::DocumentationFetcher;
use tracing_subscriber;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!("Starting standalone Tauri documentation fetcher...");

    // Use in-memory DB for fast testing
    let db = Database::new_in_memory().await?;
    let source = DocumentationSource {
        id: "tauri-docs".to_string(),
        name: "Tauri Documentation".to_string(),
        base_url: "https://v2.tauri.app/".to_string(),
        doc_type: DocType::Tauri,
        last_updated: None,
        version: Some("2.0".to_string()),
    };
    let fetcher = DocumentationFetcher::new(db.clone());
    
    // Add source to database first
    db.add_source(&source).await?;
    
    println!("Fetching Tauri docs from {}...", source.base_url);
    let result = fetcher.fetch_tauri_docs_recursive(&source).await;
    match result {
        Ok(_) => println!("Tauri docs fetch completed successfully."),
        Err(e) => println!("Tauri docs fetch failed: {}", e),
    }
    // Print summary
    let page_count = db.get_page_count_for_type(&DocType::Tauri).await.unwrap_or(0);
    println!("Total Tauri pages fetched: {}", page_count);
    Ok(())
}
