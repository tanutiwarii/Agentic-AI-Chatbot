#!/usr/bin/env python3
"""
Test script to verify the GitHub tools work correctly
"""

import os
from dotenv import load_dotenv
from tools.github_tools import CreateFileTool, PushToGitHubTool, TriggerWorkflowTool, CheckWorkflowStatusTool

def test_tools():
    """Test the GitHub tools"""
    load_dotenv()
    
    print("🧪 Testing GitHub Tools...")
    
    # Test environment variables
    required_vars = ['GITHUB_TOKEN', 'GITHUB_OWNER', 'GITHUB_REPO_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set up your .env file with the required variables")
        return False
    
    print("✅ Environment variables configured")
    
    # Test CreateFileTool
    print("\n📝 Testing CreateFileTool...")
    create_tool = CreateFileTool()
    test_content = "print('Hello, World!')"
    result = create_tool._run("test_file.py", test_content)
    print(f"Result: {result}")
    
    # Test CheckWorkflowStatusTool
    print("\n📊 Testing CheckWorkflowStatusTool...")
    status_tool = CheckWorkflowStatusTool()
    result = status_tool._run()
    print(f"Result: {result}")
    
    # Clean up test file
    if os.path.exists("test_file.py"):
        os.remove("test_file.py")
        print("🧹 Cleaned up test file")
    
    print("\n✅ Tool testing completed!")
    return True

if __name__ == "__main__":
    test_tools() 