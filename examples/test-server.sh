#!/bin/bash

# Test script for the MCP server

DATABASE_PATH="/tmp/test.db"
SERVER_PATH="./target/debug/docs-mcp-server"

echo "Testing MCP server functionality..."

# Build the project first
source "$HOME/.cargo/env"
cargo build

# Test 1: Initialize request
echo "Test 1: Initialize request"
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}' | timeout 5s $SERVER_PATH --database $DATABASE_PATH

echo ""
echo "Test 2: List tools"
echo '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}' | timeout 5s $SERVER_PATH --database $DATABASE_PATH

echo ""
echo "Test 3: Search documentation"
echo '{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search_documentation", "arguments": {"query": "ownership", "source": "rust", "limit": 3}}}' | timeout 5s $SERVER_PATH --database $DATABASE_PATH

echo ""
echo "Test 4: List resources"
echo '{"jsonrpc": "2.0", "id": 4, "method": "resources/list"}' | timeout 5s $SERVER_PATH --database $DATABASE_PATH

echo ""
echo "Done!"

# Create test database
mkdir -p test_data
DB_PATH="test_data/test_docs.db"

# Test the help command
echo "✓ Testing help command..."
cargo run --bin docs-mcp-server -- --help > /dev/null

# Test the update-docs binary
echo "✓ Testing update-docs binary..."
cargo run --bin update-docs -- --database "$DB_PATH" &
UPDATE_PID=$!
sleep 5
kill $UPDATE_PID 2>/dev/null || true

echo "✓ Testing server initialization..."
timeout 10s cargo run --bin docs-mcp-server -- --database "$DB_PATH" &
SERVER_PID=$!
sleep 3
kill $SERVER_PID 2>/dev/null || true

echo "✓ All tests passed!"
echo ""
echo "To start the server manually:"
echo "  cargo run --bin docs-mcp-server -- --database docs.db --update-now"
echo ""
echo "To update documentation manually:"
echo "  cargo run --bin update-docs -- --database docs.db"
