# tools/github_tools.py

from crewai.tools import BaseTool
from git import Repo
import os
from dotenv import load_dotenv
import requests
from typing import Any
import json

load_dotenv()

class CreateFileTool(BaseTool):
    name: str = "CreateFile"
    description: str = "Create a file with specified content and filename"

    def _run(self, filename: str, content: str) -> str:
        try:
            with open(filename, 'w') as f:
                f.write(content)
            return f"✅ File '{filename}' created successfully with {len(content)} characters."
        except Exception as e:
            return f"❌ Error creating file: {str(e)}"


class PushToGitHubTool(BaseTool):
    name: str = "PushToGitHub"
    description: str = "Commit and push a file to the GitHub repository"

    def _run(self, filename: str, commit_msg: str = 'Add new file', repo_path: str = '.') -> str:
        try:
            # Get environment variables
            token = os.getenv("GITHUB_TOKEN")
            owner = os.getenv("GITHUB_OWNER")
            repo_name = os.getenv("GITHUB_REPO_NAME")
            
            if not all([token, owner, repo_name]):
                return "❌ Missing GitHub environment variables: GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO_NAME"
            
            # Initialize git repository
            repo = Repo(repo_path)
            
            # Check if file exists
            if not os.path.exists(filename):
                return f"❌ File '{filename}' does not exist"
            
            # Check if remote exists, if not create it
            try:
                origin = repo.remote(name='origin')
            except ValueError:
                # Remote doesn't exist, create it
                remote_url = f"https://github.com/{owner}/{repo_name}.git"
                origin = repo.create_remote('origin', remote_url)
            
            # Check if there are any changes to commit
            repo.git.add('.')  # Add all changes including the new file
            
            # Check if there are staged changes
            if not repo.index.diff('HEAD'):
                # No changes to commit
                return f"⚠️ No changes to commit for '{filename}'. File may already be committed or unchanged."
            
            # Commit changes
            repo.git.commit(m=commit_msg)
            
            # Push to remote with upstream branch
            origin.push(set_upstream=True, refspec=f"main:main")
            
            return f"🚀 Successfully pushed '{filename}' to GitHub with commit: '{commit_msg}'"
            
        except Exception as e:
            return f"❌ Error pushing to GitHub: {str(e)}"


class TriggerWorkflowTool(BaseTool):
    name: str = "TriggerWorkflow"
    description: str = "Trigger a GitHub Actions workflow"

    def _run(self, workflow_file: str = 'main.yml') -> str:
        try:
            # Get environment variables
            token = os.getenv("GITHUB_TOKEN")
            owner = os.getenv("GITHUB_OWNER")
            repo_name = os.getenv("GITHUB_REPO_NAME")
            branch = os.getenv("GITHUB_BRANCH", "main")
            
            if not all([token, owner, repo_name]):
                return "❌ Missing required environment variables: GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO_NAME"
            
            # First, check if workflows exist
            workflows_url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            workflows_response = requests.get(workflows_url, headers=headers)
            if workflows_response.status_code == 200:
                workflows = workflows_response.json()
                if workflows.get("total_count", 0) == 0:
                    return "⚠️ No workflows found in repository. Please add a workflow file first."
            
            # Try to trigger the workflow
            url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows/{workflow_file}/dispatches"
            data = {"ref": branch}
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 204:
                return f"✅ Workflow '{workflow_file}' triggered successfully on branch '{branch}'"
            elif response.status_code == 404:
                return f"❌ Workflow '{workflow_file}' not found. Please check the workflow file name."
            else:
                return f"❌ Failed to trigger workflow: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Error triggering workflow: {str(e)}"


class CheckWorkflowStatusTool(BaseTool):
    name: str = "CheckWorkflowStatus"
    description: str = "Check the status of the latest GitHub Actions workflow run"

    def _run(self) -> str:
        try:
            # Get environment variables
            token = os.getenv("GITHUB_TOKEN")
            owner = os.getenv("GITHUB_OWNER")
            repo_name = os.getenv("GITHUB_REPO_NAME")
            
            if not all([token, owner, repo_name]):
                return "❌ Missing required environment variables: GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO_NAME"
            
            url = f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                runs = response.json().get('workflow_runs', [])
                if runs:
                    latest_run = runs[0]
                    status = latest_run.get('status', 'unknown')
                    conclusion = latest_run.get('conclusion', 'unknown')
                    workflow_name = latest_run.get('name', 'Unknown Workflow')
                    
                    return f"📊 Latest workflow '{workflow_name}': Status={status}, Conclusion={conclusion}"
                else:
                    return "📊 No workflow runs found"
            else:
                return f"❌ Failed to get workflow status: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Error checking workflow status: {str(e)}"
