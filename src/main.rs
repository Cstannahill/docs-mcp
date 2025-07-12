use anyhow::Result;
use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber;

// Load environment variables from .env file
use dotenv::dotenv;

use docs_mcp_server::{
    database::Database,
    server::McpServer,
    http_server::HttpServer,
    scheduler::Scheduler,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenv().ok();
    
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
        .arg(
            Arg::new("http-server")
                .long("http-server")
                .action(clap::ArgAction::SetTrue)
                .help("Start HTTP server instead of MCP server for direct access")
        )
        .arg(
            Arg::new("port")
                .long("port")
                .value_name("PORT")
                .help("Port for HTTP server (default: 3000)")
                .default_value("3000")
        )
        .get_matches();

    let db_path = matches.get_one::<String>("database").unwrap();
    let update_now = matches.get_flag("update-now");
    let openai_api_key = matches.get_one::<String>("openai-api-key").cloned()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());
    let http_mode = matches.get_flag("http-server");
    let port = matches.get_one::<String>("port").unwrap().parse::<u16>()
        .map_err(|_| anyhow::anyhow!("Invalid port number"))?;

    // Ensure database directory exists
    if let Some(parent) = PathBuf::from(db_path).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Initialize database
    let db_url = format!("sqlite://{}?mode=rwc", db_path);
    let db = Database::new(&db_url).await?;

    // Initialize all database schemas during startup
    info!("Initializing database schemas...");
    
    // 1. Basic tables are already created in Database::new()
    // 2. Initialize model discovery tables if needed
    if let Err(e) = db.initialize_model_discovery().await {
        info!("Model discovery initialization skipped: {}", e);
    }
    
    // 3. Run advanced features migration
    if let Err(e) = db.run_advanced_features_migration().await {
        info!("Advanced features migration failed, continuing with basic functionality: {}", e);
    }
    
    info!("Database initialization complete");

    // Initialize scheduler for daily updates
    let scheduler = Scheduler::new(db.clone());
    
    if update_now {
        info!("Performing initial documentation update...");
        scheduler.force_update().await?;
    }

    // Start daily update scheduler
    scheduler.start_daily_updates().await?;

    if http_mode {
        // Start HTTP server for direct access
        info!("Starting HTTP server mode on port {}", port);
        let http_server = HttpServer::new(&db_url, openai_api_key).await?;
        http_server.start_server(port).await?;
    } else {
        // Create and start MCP server
        let server = McpServer::new(&db_url, openai_api_key).await.map_err(|e| anyhow::anyhow!("{}", e))?;
        
        info!("Starting MCP server...");
        server.run().await?;
    }

    Ok(())
}
