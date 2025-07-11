use docs_mcp_server::database::{Database, DocumentationSource, DocType};
use docs_mcp_server::fetcher::DocumentationFetcher;
use tracing_subscriber;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!("Starting standalone Python documentation fetcher...");

    // Use in-memory DB for fast testing
    let db = Database::new_in_memory().await?;
    let source = DocumentationSource {
        id: "python-docs".to_string(),
        name: "Python Documentation".to_string(),
        base_url: "https://docs.python.org/3/".to_string(),
        doc_type: DocType::Python,
        last_updated: None,
        version: None,
    };
    let fetcher = DocumentationFetcher::new(db.clone());
    
    // Add source to database first
    db.add_source(&source).await?;
    
    println!("Fetching Python docs from {}...", source.base_url);
    let result = fetcher.fetch_python_docs(&source).await;
    match result {
        Ok(_) => println!("Python docs fetch completed successfully."),
        Err(e) => println!("Python docs fetch failed: {}", e),
    }
    // Print summary
    let page_count = db.get_page_count_for_type(&DocType::Python).await.unwrap_or(0);
    println!("Total Python pages fetched: {}", page_count);
    Ok(())
}
