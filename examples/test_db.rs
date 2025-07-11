use anyhow::Result;
use docs_mcp_server::database::Database;

#[tokio::main]
async fn main() -> Result<()> {
    // Test database connection and basic operations
    std::fs::create_dir_all("test_data").unwrap();
    let db = Database::new("sqlite://test_data/test.db").await?;
    
    println!("✓ Database connection successful");
    
    // Test getting sources (should be empty initially)
    let sources = db.get_sources().await?;
    println!("✓ Found {} documentation sources", sources.len());
    
    println!("✓ All database tests passed!");
    
    Ok(())
}
