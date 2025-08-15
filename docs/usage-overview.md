# Project Usage Overview

This guide provides a detailed walkthrough of the `docs-mcp` project, describing the purpose and usage of each major section and module. Use this as a reference to understand the structure and functionality of the repository.

## Top-Level Files

- **Cargo.toml / Cargo.lock**: Rust package manifest and lockfile. Defines dependencies and project metadata.
- **package.json / tsconfig.json**: Node.js/TypeScript configuration files. Used for any JS/TS tooling or scripts.
- **README.md**: Project introduction, setup instructions, and general documentation.
- **LICENSE**: Project license information.

## docs/
Contains documentation and architecture guides:
- **AGENTIC_FLOW_ARCHITECTURE.md**: Describes the agentic flow architecture.
- **AGENTIC_IMPLEMENTATION_AUDIT.md**: Implementation audit details.
- **AGENTIC_IMPLEMENTATION_COMPLETE.md**: Complete implementation notes.
- **AGENTIC_IMPLEMENTATION_ROADMAP.md**: Roadmap for agentic implementation.
- **DOCUMENTATION_VERSIONING_SUMMARY.md**: Versioning strategy for documentation.
- **ENHANCEMENT_SUMMARY.md**: Summary of enhancements.
- **IMPLEMENTATION_ROADMAP.md**: General implementation roadmap.
- **MODEL_DISCOVERY_IMPLEMENTATION_SUMMARY.md**: Model discovery implementation details.
- **OLLAMA_COPILOT_ENHANCEMENTS.md**: Enhancements related to Ollama Copilot integration.
- **OLLAMA_INTEGRATION.md**: Details on integrating with Ollama.
- **QUICK_START_GUIDE.md**: Quick start instructions for new users.
- **WEB_SEARCH_FILE_OPS_SUMMARY.md**: Summary of web search and file operations.

## examples/
Contains example scripts and configuration files:
- **agentic_flow_demo.rs**: Rust demo for agentic flow.
- **code_analysis_example.rs**: Example of code analysis.
- **comprehensive_model_evaluation_demo.rs**: Comprehensive model evaluation demo.
- **copilot-config-production.json / copilot-config.json**: Configuration files for Copilot integration.
- **enhanced_features_demo_standalone.py / enhanced_features_demo.py**: Python demos for enhanced features.
- **enhanced-features-summary.sh**: Shell script summarizing enhanced features.
- **model_discovery_integration.rs**: Rust example for model discovery integration.
- **quick-enhanced-test.sh**: Quick test script for enhanced features.
- **run-server.sh**: Script to run the server.
- **test_db.rs**: Rust test for database functionality.
- **test-enhanced-features.py**: Python test for enhanced features.
- **test-enhanced-server.sh**: Shell test for enhanced server.
- **test-live-ai.py**: Python test for live AI features.
- **test-model-discovery.sh**: Shell test for model discovery.
- **test-phase2-ai.py**: Python test for phase 2 AI features.
- **test-real-agentic-flow.sh**: Shell test for real agentic flow.
- **test-server.sh**: Shell test for server.
- **test-version-system.sh**: Shell test for version system.

## migrations/
SQL migration scripts for database setup and updates:
- **001_initial.sql**: Initial database schema.
- **002_advanced_features.sql**: Adds advanced features to the schema.

## src/
Main Rust source code. Key modules:
- **ai_integration.rs**: Handles AI model integration.
- **cache.rs**: Implements caching logic.
- **chat_interface.rs**: Chat interface functionality.
- **database.rs**: Database access and management.
- **embeddings.rs**: Embedding model logic.
- **enhanced_search.rs**: Enhanced search features.
- **fetcher.rs**: Data fetching utilities.
- **file_manager.rs**: File management operations.
- **http_server.rs**: HTTP server implementation.
- **learning.rs**: Learning algorithms and logic.
- **lib.rs**: Library entry point.
- **main.rs**: Main binary entry point.
- **ranking.rs**: Ranking algorithms.
- **scheduler.rs**: Task scheduling logic.
- **server.rs**: Server orchestration.
- **web_search.rs**: Web search integration.

### src/agents/
Agent modules for specialized tasks:
- **code_analyzer.rs**: Code analysis agent.
- **context_manager.rs**: Context management agent.
- **coordinator.rs**: Agent coordination logic.
- **doc_generator.rs**: Documentation generation agent.
- **embedder.rs**: Embedding agent.
- **model_selector.rs**: Model selection agent.
- **performance_analyzer.rs**: Performance analysis agent.
- **security_auditor.rs**: Security auditing agent.
- **test_generator.rs**: Test generation agent.
- **mod.rs**: Module index for agents.

### src/bin/
Standalone binaries for specific tasks:
- **fetch-python.rs, fetch-rust-book.rs, fetch-rust-std.rs, fetch-tauri.rs, fetch-typescript.rs**: Fetch scripts for various languages/frameworks.
- **test-real-agentic-flow.rs**: Test binary for agentic flow.
- **update_docs.rs**: Documentation update utility.

### src/model_clients/
Model client modules:
- **mod.rs**: Module index.
- **ollama_client.rs**: Ollama model client integration.

### src/model_discovery/
Model discovery logic:
- **database.rs**: Model discovery database logic.
- **evaluator.rs**: Model evaluation logic.
- **mod.rs**: Module index.
- **ollama_provider.rs**: Ollama provider integration.
- **scheduler.rs**: Model discovery scheduler.

### src/orchestrator/
Orchestration logic:
- **mod.rs**: Module index.
- **model_router.rs**: Model routing logic.

## static/
Static assets for web server:
- **index.html**: Main HTML file for web interface.

## target/
Build output directory. Contains compiled binaries and intermediate files.

---

## Usage Guides by Section

### Documentation (`docs/`)
- Use the architecture and roadmap files to understand the agentic flow and implementation plans.
- Refer to enhancement and integration guides for details on advanced features and Ollama integration.
- The quick start guide is recommended for new users.

### Examples (`examples/`)
- Run shell scripts for quick tests and demonstrations: `zsh examples/run-server.sh`.
- Use Rust and Python examples to explore model evaluation, code analysis, and enhanced features.
- Configuration files (`copilot-config.json`) are used for Copilot integration.

### Migrations (`migrations/`)
- Apply SQL migration scripts to set up or update the database schema.
- Use with your preferred database tool: `psql`, `sqlite3`, etc.

### Source Code (`src/`)
- Main entry point: `src/main.rs`.
- Library entry: `src/lib.rs`.
- Each module is responsible for a specific aspect (AI, caching, chat, database, etc.).
- Agent modules in `src/agents/` provide specialized functionality for code analysis, context management, documentation, embedding, model selection, performance, security, and testing.
- Use binaries in `src/bin/` for standalone tasks (fetching, testing, updating docs).
- Model clients and discovery logic are in `src/model_clients/` and `src/model_discovery/`.
- Orchestration logic is in `src/orchestrator/`.

### Static Assets (`static/`)
- `static/index.html` is the main web interface entry point.

### Build Output (`target/`)
- Contains compiled binaries and build artifacts. Not typically edited directly.

---

## Getting Started
1. Review the `README.md` and `docs/QUICK_START_GUIDE.md`.
2. Set up the database using migration scripts in `migrations/`.
3. Build and run the server using Rust (`cargo build`, `cargo run`) or shell scripts in `examples/`.
4. Explore example scripts for demonstrations and tests.
5. Refer to documentation in `docs/` for architecture and feature details.

---

For more details on any module, refer to the corresponding source file or documentation in the `docs/` folder.
