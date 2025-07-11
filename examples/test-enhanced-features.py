#!/usr/bin/env python3
"""
Test script to demonstrate enhanced MCP documentation server features
"""
import subprocess
import json
import time
import sys
import os


def start_server():
    """Start the MCP server subprocess"""
    cmd = ["cargo", "run", "--bin", "docs-mcp-server", "--", "--database", "test.db"]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/workspaces/docs-mcp",
    )


def send_request(server, request):
    """Send a JSON-RPC request to the server"""
    request_json = json.dumps(request) + "\n"
    server.stdin.write(request_json)
    server.stdin.flush()

    # Read the response
    response_line = server.stdout.readline()
    if response_line:
        try:
            return json.loads(response_line.strip())
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response_line}")
            return None
    return None


def test_enhanced_search():
    """Test the enhanced search functionality"""
    print("🔍 Testing Enhanced Search Features")
    print("=" * 50)

    server = start_server()
    time.sleep(3)  # Give server time to start

    try:
        # Test 1: List available tools
        print("\n1. Testing available tools...")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = send_request(server, tools_request)
        if response and "result" in response:
            tools = response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")

        # Test 2: Enhanced search with content filtering
        print("\n2. Testing enhanced search with content filtering...")
        search_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_documentation",
                "arguments": {
                    "query": "useState",
                    "content_type": "tutorial",
                    "limit": 3,
                },
            },
        }

        response = send_request(server, search_request)
        if response and "result" in response:
            content = response["result"]["content"]
            print(f"✅ Search results: {len(content)} items")
            for i, result in enumerate(content[:3], 1):
                print(f"   {i}. {result.get('title', 'No title')}")
                if "snippet" in result:
                    snippet = (
                        result["snippet"][:100] + "..."
                        if len(result["snippet"]) > 100
                        else result["snippet"]
                    )
                    print(f"      Snippet: {snippet}")

        # Test 3: Cache statistics
        print("\n3. Testing cache statistics...")
        cache_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_cache_stats", "arguments": {}},
        }

        response = send_request(server, cache_request)
        if response and "result" in response:
            content = response["result"]["content"]
            print(f"✅ Cache stats: {content[:200]}...")

        # Test 4: Documentation sources
        print("\n4. Testing documentation sources list...")
        sources_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "list_documentation_sources", "arguments": {}},
        }

        response = send_request(server, sources_request)
        if response and "result" in response:
            content = response["result"]["content"]
            print(f"✅ Sources list: {content[:200]}...")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    test_enhanced_search()
