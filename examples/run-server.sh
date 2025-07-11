#!/bin/bash

# Example script to build and run the docs MCP server

set -e

echo "Building docs MCP server..."
cargo build --release

echo "Creating database directory..."
mkdir -p data

echo "Starting server with initial documentation update..."
./target/release/docs-mcp-server --database data/docs.db --update-now
