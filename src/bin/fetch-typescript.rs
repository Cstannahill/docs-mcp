use docs_mcp_server::database::{Database, DocumentationSource, DocType};
use docs_mcp_server::fetcher::DocumentationFetcher;
use tracing_subscriber;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!("Starting standalone TypeScript documentation fetcher with concurrent processing...");

    // Use persistent database to save the data
    let db = Database::new("docs.db").await?;
    let source = DocumentationSource {
        id: "typescript-docs".to_string(),
        name: "TypeScript Handbook".to_string(),
        base_url: "https://www.typescriptlang.org/docs/".to_string(),
        doc_type: DocType::TypeScript,
        last_updated: None,
        version: None,
    };
    
    let fetcher = DocumentationFetcher::new(db.clone());
    
    // Add source to database first
    db.add_source(&source).await?;
    
    println!("Fetching TypeScript docs from {} with concurrent processing...", source.base_url);
    let start = std::time::Instant::now();
    
    let result = fetcher.fetch_typescript_docs(&source).await;
    
    let elapsed = start.elapsed();
    println!("Fetching completed in {:?}", elapsed);
    
    match result {
        Ok(_) => println!("TypeScript docs fetch completed successfully."),
        Err(e) => println!("TypeScript docs fetch failed: {}", e),
    }
    
    // Print summary
    let page_count = db.get_page_count_for_type(&DocType::TypeScript).await.unwrap_or(0);
    println!("Total TypeScript pages fetched: {}", page_count);

    Ok(())
}
