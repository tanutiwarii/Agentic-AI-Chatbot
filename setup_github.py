#!/usr/bin/env python3
"""
Setup script to configure GitHub credentials
"""

import os
from dotenv import load_dotenv

def setup_github():
    """Setup GitHub credentials"""
    print("🔧 GitHub Setup")
    print("=" * 50)
    
    # Load existing .env file
    load_dotenv()
    
    # Get current values
    current_token = os.getenv("GITHUB_TOKEN", "")
    current_owner = os.getenv("GITHUB_OWNER", "")
    current_repo = os.getenv("GITHUB_REPO_NAME", "")
    
    print(f"Current GITHUB_TOKEN: {'✅ Set' if current_token and current_token != 'your_github_personal_access_token' else '❌ Not set'}")
    print(f"Current GITHUB_OWNER: {'✅ Set' if current_owner and current_owner != 'your_github_username' else '❌ Not set'}")
    print(f"Current GITHUB_REPO_NAME: {'✅ Set' if current_repo and current_repo != 'your_repository_name' else '❌ Not set'}")
    
    print("\n📝 To set up GitHub integration:")
    print("1. Go to GitHub Settings → Developer settings → Personal access tokens")
    print("2. Generate a new token with 'repo' and 'workflow' permissions")
    print("3. Update your .env file with:")
    print("   GITHUB_TOKEN=your_actual_token")
    print("   GITHUB_OWNER=your_github_username")
    print("   GITHUB_REPO_NAME=your_repository_name")
    
    print("\n🔗 Example .env file:")
    print("GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx")
    print("GITHUB_OWNER=yourusername")
    print("GITHUB_REPO_NAME=my-repo")
    print("GITHUB_BRANCH=main")
    print("OLLAMA_BASE_URL=http://localhost:11434")
    
    # Check if Git is initialized
    if os.path.exists(".git"):
        print("\n✅ Git repository is initialized")
    else:
        print("\n❌ Git repository not initialized")
        print("Run: git init")
    
    return True

if __name__ == "__main__":
    setup_github() 