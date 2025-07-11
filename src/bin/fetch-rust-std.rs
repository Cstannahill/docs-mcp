use docs_mcp_server::database::{Database, DocumentationSource, DocType};
use docs_mcp_server::fetcher::DocumentationFetcher;
use tracing_subscriber;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!("Starting standalone Rust Standard Library documentation fetcher...");

    // Use in-memory DB for fast testing
    let db = Database::new_in_memory().await?;
    let source = DocumentationSource {
        id: "rust-std".to_string(),
        name: "Rust Standard Library".to_string(),
        base_url: "https://doc.rust-lang.org/std/".to_string(),
        doc_type: DocType::Rust,
        last_updated: None,
        version: None,
    };
    
    let fetcher = DocumentationFetcher::new(db.clone());
    
    // Add source to database first
    db.add_source(&source).await?;
    
    println!("Fetching Rust std docs from {}...", source.base_url);
    let result = fetcher.fetch_rust_docs(&source).await;
    match result {
        Ok(_) => println!("Rust std docs fetch completed successfully."),
        Err(e) => println!("Rust std docs fetch failed: {}", e),
    }
    // Print summary
    let page_count = db.get_page_count_for_type(&DocType::Rust).await.unwrap_or(0);
    println!("Total Rust std pages fetched: {}", page_count);

    Ok(())
}
