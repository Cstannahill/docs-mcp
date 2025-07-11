# Documentation MCP Server

A Model Context Protocol (MCP) server that provides GitHub Copilot with access to the latest documentation for Rust, Tauri, React, TypeScript, Python, Tailwind CSS, and shadcn/ui. The server automatically fetches and updates documentation daily to ensure you always have access to the most current information.

## Features

- **Multi-language Documentation**: Supports Rust, Tauri, React, TypeScript, Python, Tailwind CSS, and shadcn/ui documentation
- **Automatic Updates**: Daily scheduled updates to keep documentation current
- **Full-text Search**: Advanced search capabilities across all documentation sources
- **MCP Integration**: Native integration with GitHub Copilot and other MCP-compatible tools
- **SQLite Storage**: Efficient local caching with SQLite database
- **Markdown Support**: Clean markdown formatting for better readability

## Documentation Sources

- **Rust**: Standard library documentation and The Rust Programming Language book
- **Tauri**: Complete Tauri framework documentation
- **React**: Official React documentation and guides
- **TypeScript**: TypeScript handbook and language reference
- **Python**: Official Python documentation including tutorials and library reference
- **Tailwind CSS**: Complete Tailwind CSS V4+ utility-first framework documentation
- **shadcn/ui**: Component library documentation and examples

## Installation

### Prerequisites

- Rust 1.70 or later
- SQLite 3

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd docs-mcp-server

# Build the project
cargo build --release

# The binary will be available at target/release/docs-mcp-server
```

## Usage

### Starting the MCP Server

```bash
# Start the server with default settings
./target/release/docs-mcp-server

# Start with custom database path
./target/release/docs-mcp-server --database /path/to/docs.db

# Force update documentation before starting
./target/release/docs-mcp-server --update-now
```

### Manual Documentation Update

```bash
# Update documentation manually
./target/release/update-docs

# Update with custom database path
./target/release/update-docs --database /path/to/docs.db
```

### GitHub Copilot Integration

To use this MCP server with GitHub Copilot, add it to your MCP configuration:

```json
{
  "mcpServers": {
    "docs-server": {
      "command": "/path/to/target/release/docs-mcp-server",
      "args": ["--database", "/path/to/docs.db"]
    }
  }
}
```

## Available Tools

### `search_documentation`

Search across all documentation sources for relevant information.

**Parameters:**

- `query` (required): Search query for documentation content
- `source` (optional): Limit search to specific documentation source (`rust`, `tauri`, `react`, `typescript`, `python`, `tailwind`, `shadcn`)
- `limit` (optional): Maximum number of results to return (default: 10)

**Example:**

```json
{
  "name": "search_documentation",
  "arguments": {
    "query": "async functions",
    "source": "rust",
    "limit": 5
  }
}
```

### `get_documentation_page`

Get a specific documentation page by path.

**Parameters:**

- `source_id` (required): Documentation source ID (e.g., 'rust-std', 'react-docs')
- `path` (required): Path to the specific documentation page

### `list_documentation_sources`

List all available documentation sources and their status.

## Available Resources

The server provides the following resources that can be accessed directly:

- `docs://rust/std` - Rust Standard Library Documentation
- `docs://rust/book` - The Rust Programming Language Book
- `docs://tauri` - Tauri Documentation
- `docs://react` - React Documentation
- `docs://typescript` - TypeScript Documentation
- `docs://python` - Python Documentation
- `docs://tailwind` - Tailwind CSS Documentation
- `docs://shadcn` - shadcn/ui Documentation

## Configuration

### Environment Variables

- `RUST_LOG`: Set logging level (default: `info`)

### Command Line Options

- `--database <PATH>`: Path to SQLite database file (default: `docs.db`)
- `--update-now`: Force update documentation before starting server

## Development

### Running in Development Mode

```bash
# Run with cargo
cargo run

# Run with auto-reload
cargo watch -x run
```

### Running Tests

```bash
cargo test
```

### Code Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy
```

## Database Schema

The server uses SQLite with the following main tables:

- `documentation_sources`: Stores information about documentation sources
- `document_pages`: Stores individual documentation pages
- `document_pages_fts`: Full-text search index for efficient searching

## Scheduling

The server automatically schedules daily updates at 2 AM UTC. You can also trigger manual updates using the `update-docs` binary or by starting the server with the `--update-now` flag.

## Error Handling

The server includes comprehensive error handling and logging. All errors are logged with appropriate context, and the server will continue running even if individual documentation updates fail.

## Performance

- **Caching**: All documentation is cached locally in SQLite for fast access
- **Incremental Updates**: Only updates changed content to minimize bandwidth
- **Full-text Search**: Uses SQLite FTS5 for efficient text searching
- **Connection Pooling**: Efficient database connection management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run `cargo test` and `cargo clippy`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Database Lock Errors**: Ensure only one instance of the server is running
2. **Network Timeouts**: Check internet connection and firewall settings
3. **Permission Errors**: Ensure write access to the database directory

### Logging

Enable detailed logging by setting the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug ./target/release/docs-mcp-server
```
