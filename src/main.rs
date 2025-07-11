mod ai_integration;
mod cache;
mod database;
mod fetcher;
mod server;
mod scheduler;

use anyhow::Result;
use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber;

use database::Database;
use server::McpServer;
use scheduler::Scheduler;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    let matches = Command::new("docs-mcp-server")
        .version("1.0.0")
        .about("MCP server providing access to latest documentation")
        .arg(
            Arg::new("database")
                .long("database")
                .value_name("PATH")
                .help("Path to SQLite database file")
                .default_value("docs.db")
        )
        .arg(
            Arg::new("update-now")
                .long("update-now")
                .action(clap::ArgAction::SetTrue)
                .help("Force update documentation before starting server")
        )
        .arg(
            Arg::new("openai-api-key")
                .long("openai-api-key")
                .value_name("KEY")
                .help("OpenAI API key for enhanced AI features (optional)")
        )
        .get_matches();

    let db_path = matches.get_one::<String>("database").unwrap();
    let update_now = matches.get_flag("update-now");
    let openai_api_key = matches.get_one::<String>("openai-api-key").cloned()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());

    // Ensure database directory exists
    if let Some(parent) = PathBuf::from(db_path).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Initialize database
    let db_url = format!("sqlite://{}?mode=rwc", db_path);
    let db = Database::new(&db_url).await?;

    // Initialize scheduler for daily updates
    let scheduler = Scheduler::new(db.clone());
    
    if update_now {
        info!("Performing initial documentation update...");
        scheduler.force_update().await?;
    }

    // Start daily update scheduler
    scheduler.start_daily_updates().await?;

    // Create and start MCP server
    let server = McpServer::new(&db_url, openai_api_key).await?;
    
    info!("Starting MCP server...");
    server.run().await?;

    Ok(())
}
