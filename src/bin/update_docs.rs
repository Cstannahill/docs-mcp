use anyhow::Result;
use clap::{Arg, Command};
use tracing::{info, Level};
use tracing_subscriber;

use docs_mcp_server::database::Database;
use docs_mcp_server::fetcher::DocumentationFetcher;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let matches = Command::new("update-docs")
        .version("1.0.0")
        .about("Update documentation cache")
        .arg(
            Arg::new("database")
                .long("database")
                .value_name("PATH")
                .help("Path to SQLite database file")
                .default_value("docs.db")
        )
        .get_matches();

    let db_path = matches.get_one::<String>("database").unwrap();

    // Ensure database directory exists
    if let Some(parent) = std::path::PathBuf::from(db_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Initialize database
    let db_url = format!("sqlite://{}?mode=rwc", db_path);
    let db = Database::new(&db_url).await?;

    // Update all documentation
    let fetcher = DocumentationFetcher::new(db);
    info!("Starting documentation update...");
    fetcher.update_all_documentation().await?;
    info!("Documentation update completed!");

    Ok(())
}
