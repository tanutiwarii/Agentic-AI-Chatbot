import os
import queue
import streamlit as st
from langchain.agents import  Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_experimental.tools import PythonREPLTool
import speech_recognition as sr
import subprocess
import json
import pyttsx3
from langchain_community.llms import Ollama
import threading
from datetime import datetime
from dotenv import load_dotenv

# Import CrewAI and GitHub tools
from crewai import Agent, Crew, Task
from litellm import completion
from langchain.llms.base import LLM
from typing import Any, List, Optional

# Import GitHub tools
try:
    from tools.github_tools import CreateFileTool, PushToGitHubTool, TriggerWorkflowTool, CheckWorkflowStatusTool  # type: ignore
except ImportError:
    # Create dummy tools if not available
    class CreateFileTool:  # type: ignore
        def _run(self, filename, content):
            return f"File {filename} would be created with content: {content[:50]}..."
    
    class PushToGitHubTool:  # type: ignore
        def _run(self, filename, commit_message):
            return f"File {filename} would be pushed to GitHub with message: {commit_message}"
    
    class TriggerWorkflowTool:  # type: ignore
        def _run(self):
            return "GitHub Actions workflow would be triggered"
    
    class CheckWorkflowStatusTool:  # type: ignore
        def _run(self):
            return "Workflow status would be checked"

load_dotenv()

# Global TTS queue and thread for thread-safe speech
tts_queue = queue.Queue()
tts_thread = None
tts_running = False

# === Custom LLM Class for CrewAI ===
class CustomLLM(LLM):
    """Custom LLM class that works with CrewAI"""
    
    model_name: str = "ollama/llama3.2:latest"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self._get_best_model()
    
    def _get_best_model(self):
        # Test if Ollama is working properly using direct HTTP call
        try:
            import requests
            
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3.2:latest",
                "prompt": "test",
                "stream": False
            }
            
            response = requests.post(ollama_url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                print("✅ Ollama is working properly")
                return "ollama/llama3.2:latest"
        except Exception as e:
            print(f"Ollama test failed: {e}")
            # If Ollama fails, we'll still use it but with fallback handling
            pass
        return "ollama/llama3.2:latest"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Ensure prompt is not empty
            if not prompt or not prompt.strip():
                return "I understand your request and will proceed with the task."
            
            # Use direct HTTP call to Ollama to avoid litellm bugs
            import requests
            import json
            
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3.2:latest",
                "prompt": prompt.strip(),
                "stream": False
            }
            
            response = requests.post(ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                return result["response"]
            else:
                return "I understand your request and will proceed with the task."
                
        except Exception as e:
            print(f"LLM call failed: {e}")
            # Fallback response
            if "create" in prompt.lower() and "file" in prompt.lower():
                return "I will create the file as requested."
            elif "push" in prompt.lower() and "github" in prompt.lower():
                return "I will push the file to GitHub."
            elif "trigger" in prompt.lower() and "workflow" in prompt.lower():
                return "I will trigger the GitHub Actions workflow."
            else:
                return "I understand your request and will proceed with the task."

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._call(prompt, stop)

    def generate(self, prompts, stop=None, **kwargs):
        # Use the default LangChain generate logic
        return super().generate(prompts, stop=stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom"

# === Ansible Integration ===
def run_ansible_playbook(playbook):
    try:
        result = subprocess.run(
            ["ansible-playbook", "-i", "ansible/inventory.ini", playbook], 
            capture_output=True, 
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Ansible playbook failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    except FileNotFoundError:
        return "❌ Ansible not found. Please install Ansible first."
    except Exception as e:
        return f"Unexpected error running Ansible playbook: {str(e)}"

def run_ansible_trigger_workflow():
    return run_ansible_playbook("ansible/trigger_workflow.yml")

def run_ansible_check_latest_status():
    return run_ansible_playbook("ansible/check_latest_status.yml")

def run_ansible_download_logs():
    return run_ansible_playbook("ansible/download_logs.yml")

def run_ansible_remediate_failure():
    return run_ansible_playbook("ansible/remediate_failure.yml")

def run_ansible_retry_failed_job():
    return run_ansible_playbook("ansible/retry_failed_job.yml")

def run_ansible_create_repository(repo_name="new-repo", repo_description="Repository created via Ansible", repo_private=False):
    """Create a GitHub repository using Ansible with custom parameters"""
    try:
        # Set environment variables for the playbook
        env = os.environ.copy()
        env['REPO_NAME'] = repo_name
        env['REPO_DESCRIPTION'] = repo_description
        env['REPO_PRIVATE'] = str(repo_private).lower()
        
        result = subprocess.run(
            ["ansible-playbook", "-i", "ansible/inventory.ini", "ansible/create_repository.yml"], 
            capture_output=True, 
            text=True,
            env=env,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Ansible playbook failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    except FileNotFoundError:
        return "❌ Ansible not found. Please install Ansible first."
    except Exception as e:
        return f"Unexpected error running Ansible playbook: {str(e)}"

# === Complete Workflow Function ===
def execute_complete_workflow(filename, content, commit_message="Add new file"):
    """Execute the complete workflow: create file → push to GitHub → trigger workflow"""
    
    results = []
    
    # Step 1: Create File
    st.info("📝 Step 1: Creating file...")
    try:
        # Create the file directly using the tool
        create_tool = CreateFileTool()
        create_result = create_tool._run(filename, content)
        results.append(f"File Creation: {create_result}")
        
        if "❌" in create_result:
            st.error("File creation failed")
            return results
    except Exception as e:
        error_msg = f"File Creation failed: {str(e)}"
        results.append(error_msg)
        st.error(error_msg)
        return results
    
    # Step 2: Push to GitHub
    st.info("🚀 Step 2: Pushing to GitHub...")
    try:
        # Push to GitHub directly using the tool
        push_tool = PushToGitHubTool()
        push_result = push_tool._run(filename, commit_message)
        results.append(f"GitHub Push: {push_result}")
        
        if "❌" in push_result:
            st.error("GitHub push failed")
            return results
    except Exception as e:
        error_msg = f"GitHub Push failed: {str(e)}"
        results.append(error_msg)
        st.error(error_msg)
        return results
    
    # Step 3: Trigger Workflow
    st.info("⚡ Step 3: Triggering GitHub Actions workflow...")
    try:
        # Trigger workflow directly using the tool
        trigger_tool = TriggerWorkflowTool()
        trigger_result = trigger_tool._run()
        results.append(f"Workflow Trigger: {trigger_result}")
        
        if "❌" in trigger_result:
            st.warning("Workflow trigger failed")
    except Exception as e:
        error_msg = f"Workflow Trigger failed: {str(e)}"
        results.append(error_msg)
        st.error(error_msg)
    
    return results

# === Chat-Based Interface Functions ===
# Store last generated code in session state
def set_last_generated_code(code):
    st.session_state.last_generated_code = code

def get_last_generated_code():
    return getattr(st.session_state, 'last_generated_code', None)

# Store last created file for download and GitHub push
def set_last_created_file(filename, content):
    st.session_state.last_created_file = {'filename': filename, 'content': content}

def get_last_created_file():
    return getattr(st.session_state, 'last_created_file', None)

# Add handler for push to GitHub after file creation
def process_push_last_file_to_github():
    last_file = get_last_created_file()
    if not last_file:
        msg = "No file was created recently. Please create a file first."
        st.session_state.messages.append({"role": "assistant", "content": msg})
        with st.chat_message("assistant"):
            st.markdown(msg)
        return
    filename = last_file['filename']
    content = last_file['content']
    with st.spinner(f"Pushing {filename} to GitHub..."):
        try:
            push_tool = PushToGitHubTool()
            push_result = push_tool._run(filename, "Add file via assistant")
            msg = f"✅ File {filename} pushed to GitHub!\nGitHub Push: {push_result}"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
        except Exception as e:
            err = f"❌ Failed to push file: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.markdown(err)
    st.session_state.needs_save = True
    return

# Multi-step file creation state
FILE_CREATION_STEPS = [
    'ask_filename',
    'ask_content_source',
    'ask_generate_description',
    'create_file_confirm',
]

def reset_file_creation_state():
    st.session_state.file_creation = {
        'step': 'ask_filename',
        'filename': None,
        'content_source': None,  # 'existing', 'generate', 'user'
        'content': None,
        'generate_desc': None,
    }

def get_file_creation_state():
    if 'file_creation' not in st.session_state:
        reset_file_creation_state()
    return st.session_state.file_creation

# Always route input to file creation flow if active
def is_file_creation_active():
    return 'file_creation' in st.session_state and st.session_state.file_creation.get('step') not in (None, 'done')

# GitHub push state management
def reset_github_push_state():
    st.session_state.github_push = {
        'step': 'ask_source',
        'source': None,  # 'upload', 'paste', 'last_file'
        'filename': None,
        'content': None,
        'uploaded_file': None,
    }

def get_github_push_state():
    if 'github_push' not in st.session_state:
        reset_github_push_state()
    return st.session_state.github_push

def is_github_push_active():
    return 'github_push' in st.session_state and st.session_state.github_push.get('step') not in (None, 'done')

# Update process_chat_message to detect GitHub push intent
def process_chat_message(user_message):
    """Process user message and return appropriate response and action"""
    
    message_lower = user_message.lower()
    
    # Chained code-to-file-to-github command
    if (
        ("create file" in message_lower or "make file" in message_lower)
        and ("above code" in message_lower or "previous code" in message_lower or "last code" in message_lower)
        and ("push" in message_lower or "github" in message_lower)
    ):
        return {
            "action": "code_to_file_and_github",
            "response": None,
            "needs_input": False,
            "input_type": None
        }
    
    # Code generation commands
    if any(word in message_lower for word in ["code", "python", "function", "program", "script"]) and any(word in message_lower for word in ["sum", "add", "calculate", "compute", "generate", "write", "create"]):
        return {
            "action": "code_generation",
            "response": None,
            "needs_input": False,
            "input_type": None
        }
    
    # Multi-step file creation
    if any(word in message_lower for word in ["create file", "make file", "new file"]):
        reset_file_creation_state()
        return {
            "action": "file_creation_flow",
            "response": None,
            "needs_input": True,
            "input_type": "file_creation_flow"
        }
    
    # File creation commands
    elif any(word in message_lower for word in ["create", "make", "new"]) and any(word in message_lower for word in ["file", "python", "js", "html", "css", "json"]):
        return {
            "action": "create_file",
            "response": "I'll help you create a file! Please provide the filename and content.",
            "needs_input": True,
            "input_type": "file_creation"
        }
    
    # GitHub push commands (enhanced)
    elif any(word in message_lower for word in ["push", "commit", "upload"]) and any(word in message_lower for word in ["github", "repo", "repository"]):
        reset_github_push_state()
        return {
            "action": "github_push_flow",
            "response": None,
            "needs_input": True,
            "input_type": "github_push_flow"
        }
    
    # Workflow trigger commands
    elif any(word in message_lower for word in ["trigger", "run", "start"]) and any(word in message_lower for word in ["workflow", "action", "pipeline"]):
        return {
            "action": "trigger_workflow",
            "response": "I'll trigger the GitHub Actions workflow for you!",
            "needs_input": False,
            "input_type": None
        }
    
    # Status check commands
    elif any(word in message_lower for word in ["status", "check", "monitor"]) and any(word in message_lower for word in ["workflow", "action", "pipeline"]):
        return {
            "action": "check_status",
            "response": "I'll check the workflow status for you!",
            "needs_input": False,
            "input_type": None
        }
    
    # Complete workflow commands
    elif any(word in message_lower for word in ["complete", "full", "all"]) and any(word in message_lower for word in ["workflow", "process", "pipeline"]):
        return {
            "action": "complete_workflow",
            "response": "I'll execute the complete workflow! Please provide the filename, content, and commit message.",
            "needs_input": True,
            "input_type": "complete_workflow"
        }
    
    # Ansible workflow commands
    if "trigger" in message_lower and "workflow" in message_lower:
        return {
            "action": "ansible_trigger_workflow",
            "response": "I'll trigger the workflow using Ansible!",
            "needs_input": False,
            "input_type": None
        }
    if ("check" in message_lower or "status" in message_lower) and ("workflow" in message_lower or "run" in message_lower):
        return {
            "action": "ansible_check_latest_status",
            "response": "I'll check the latest workflow run status using Ansible!",
            "needs_input": False,
            "input_type": None
        }
    if ("download" in message_lower or "get" in message_lower) and ("log" in message_lower or "logs" in message_lower):
        return {
            "action": "ansible_download_logs",
            "response": "I'll download the latest workflow logs using Ansible!",
            "needs_input": False,
            "input_type": None
        }
    if ("remediate" in message_lower or "remediation" in message_lower or "fix" in message_lower) and ("failure" in message_lower or "fail" in message_lower):
        return {
            "action": "ansible_remediate_failure",
            "response": "I'll remediate the workflow failure using Ansible!",
            "needs_input": False,
            "input_type": None
        }
    if ("retry" in message_lower or "rerun" in message_lower) and ("job" in message_lower or "workflow" in message_lower or "run" in message_lower):
        return {
            "action": "ansible_retry_failed_job",
            "response": "I'll retry the failed workflow job using Ansible!",
            "needs_input": False,
            "input_type": None
        }
    
    # Repository creation commands
    if ("create" in message_lower or "make" in message_lower) and ("repo" in message_lower or "repository" in message_lower):
        # Extract repository details from the message
        import re
        
        # Look for repository name
        repo_match = re.search(r"(?:called|named|name)\s+([a-zA-Z0-9_-]+)", message_lower)
        repo_name = repo_match.group(1) if repo_match else "new-repo"
        
        # Look for description
        desc_match = re.search(r"(?:description|desc|about)\s+(.+?)(?:\s+(?:private|public)|$)", message_lower)
        repo_description = desc_match.group(1).strip() if desc_match else "Repository created via Ansible"
        
        # Check if private
        repo_private = "private" in message_lower
        
        return {
            "action": "ansible_create_repository",
            "response": f"I'll create a GitHub repository named '{repo_name}' using Ansible!",
            "needs_input": False,
            "input_type": None,
            "repo_name": repo_name,
            "repo_description": repo_description,
            "repo_private": repo_private
        }
    
    # README creation commands (robust) - PRIORITIZE THIS ABOVE GENERIC FILE CREATION
    if ("readme" in message_lower and "create" in message_lower) or ("readme.md" in message_lower and "create" in message_lower) or ("create" in message_lower and "readme" in message_lower) or ("make" in message_lower and "readme" in message_lower) or ("add" in message_lower and "readme" in message_lower):
        import re
        # Try to extract content after 'content:' or 'with content:'
        content_match = re.search(r"content: (.+)", user_message, re.IGNORECASE)
        if content_match:
            readme_content = content_match.group(1).strip()
        else:
            # Default template
            readme_content = "# Project Title\n\nA short description of the project.\n\n## Features\n- Feature 1\n- Feature 2\n\n## Installation\n\nInstructions here.\n\n## Usage\n\nHow to use this project.\n\n## License\n\nMIT"
        return {
            "action": "create_readme",
            "response": f"I'll create a README.md file for you!",
            "needs_input": False,
            "input_type": None,
            "filename": "README.md",
            "content": readme_content
        }

    # README content generation commands
    if ("generate" in message_lower and "readme" in message_lower):
        # Use the code generation logic to generate README content
        desc = user_message.replace("generate readme file content for", "").replace("generate readme content for", "").replace("generate readme for", "").strip()
        if not desc:
            desc = "a general project"
        # Specialized prompt for README
        code_prompt = f"""You are an expert at writing professional README.md files. Generate a complete, high-quality README.md for the following project (for a commercial website):\n\nProject description: {desc}\n\nThe README should include:\n- Project title\n- Description\n- Features\n- Installation instructions\n- Usage\n- License\n- Contact info (if appropriate)\n\nFormat the output as markdown, ready to be saved as README.md."""
        import requests
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2:latest",
            "prompt": code_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        try:
            response = requests.post(ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if "response" in result:
                readme_content = result["response"]
            else:
                readme_content = "# Project Title\n\nA short description of the project."
        except Exception as e:
            readme_content = f"# Project Title\n\nA short description of the project.\n\n*Error generating README: {str(e)}*"
        return {
            "action": "create_readme",
            "response": "Here's a generated README.md file for your project!",
            "needs_input": False,
            "input_type": None,
            "filename": "README.md",
            "content": readme_content
        }
    
    # Help commands
    elif any(word in message_lower for word in ["help", "what", "how", "commands"]):
        return {
            "action": "help",
            "response": """Here are the commands I understand:

📝 **File Operations:**
- "Create a Python file" or "Make a new file"
- "Create a JavaScript file"
- "Create an HTML file"

🚀 **GitHub Operations:**
- "Push to GitHub" or "Commit to repository"
- "Upload file to GitHub"
- "Create repository" or "Make repository"

⚡ **Workflow Operations (Ansible):**
- "Trigger workflow" or "Run GitHub Actions"
- "Check workflow status" or "Monitor pipeline"
- "Download logs" or "Get workflow logs"
- "Remediate failure" or "Fix workflow failure"
- "Retry failed job" or "Rerun workflow"

🔄 **Complete Workflow:**
- "Run complete workflow" or "Execute full pipeline"

💻 **Code Generation:**
- "Give me Python code for..." or "Write a function to..."
- "Create a program that..." or "Generate code for..."

💡 **Examples:**
- "Create a Python file called hello.py"
- "Push my changes to GitHub"
- "Create repository called my-project description: My awesome project private"
- "Trigger the workflow"
- "Check the workflow status"
- "Download logs"
- "Remediate failure"
- "Retry failed job"
- "Run the complete workflow"
- "Give me Python code for sum of n numbers"

What would you like to do?""",
            "needs_input": False,
            "input_type": None
        }
    
    # Push last created file to GitHub
    if ("push" in message_lower or "github" in message_lower) and ("this file" in message_lower or "last file" in message_lower or "recent file" in message_lower):
        return {
            "action": "push_last_file_to_github",
            "response": None,
            "needs_input": False,
            "input_type": None
        }
    
    # Default response - let the regular agent handle it
    else:
        return {
            "action": "regular_agent",
            "response": None,
            "needs_input": False,
            "input_type": None
        }

def extract_file_info_from_message(message):
    """Extract filename and content from user message"""
    # Simple extraction - can be enhanced with more sophisticated parsing
    words = message.split()
    filename = None
    content = None
    
    # Look for common file extensions
    for word in words:
        if any(ext in word.lower() for ext in ['.py', '.js', '.html', '.css', '.json', '.txt', '.md']):
            filename = word
            break
    
    # If no filename found, create a default one
    if not filename:
        if 'python' in message.lower():
            filename = 'script.py'
        elif 'javascript' in message.lower() or 'js' in message.lower():
            filename = 'script.js'
        elif 'html' in message.lower():
            filename = 'index.html'
        elif 'css' in message.lower():
            filename = 'style.css'
        elif 'json' in message.lower():
            filename = 'data.json'
        else:
            filename = 'file.txt'
    
    # Generate basic content based on file type
    if filename.endswith('.py'):
        content = '''def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()'''
    elif filename.endswith('.js'):
        content = '''function helloWorld() {
    console.log("Hello, World!");
}

helloWorld();'''
    elif filename.endswith('.html'):
        content = '''<!DOCTYPE html>
<html>
<head>
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>'''
    elif filename.endswith('.css'):
        content = '''body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}

h1 {
    color: #333;
}'''
    elif filename.endswith('.json'):
        content = '''{
    "name": "example",
    "version": "1.0.0",
    "description": "Example file"
}'''
    else:
        content = "# Custom file content"
    
    return filename, content

def tts_worker():
    """Worker thread for text-to-speech to avoid run loop conflicts"""
    global tts_running
    tts_running = True
    
    # Initialize TTS engine in the worker thread
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    
    while tts_running:
        try:
            # Get text from queue with timeout
            text = tts_queue.get(timeout=1)
            if text == "STOP":
                tts_queue.task_done()
                break
            
            print("🤖 Speaking:", text)
            engine.say(text)
            engine.runAndWait()
            tts_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS Error: {e}")
            continue
    
    print("TTS worker thread stopped")

def start_tts_worker():
    """Start the TTS worker thread"""
    global tts_thread, tts_running
    if tts_thread is None or not tts_thread.is_alive():
        tts_running = True
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

def stop_tts_worker():
    """Stop the TTS worker thread"""
    global tts_running
    tts_running = False
    if not tts_queue.empty():
        tts_queue.put("STOP")

# Conversation history file
HISTORY_FILE = "conversation_history.json"

def load_conversation_history():
    """Load conversation history from file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {"conversations": []}

def save_conversation_history(history):
    """Save conversation history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def generate_conversation_id(messages):
    """Generate a unique ID for a conversation based on its content"""
    if not messages:
        return ""
    # Use the first user message as the conversation identifier
    first_user_msg = next((msg[1] for msg in messages if msg[0] == "user"), "")
    return first_user_msg[:50]  # Use first 50 characters as ID

def find_existing_conversation(history, conversation_id):
    """Find if a conversation with the same ID already exists"""
    for i, conv in enumerate(history["conversations"]):
        if conv.get("conversation_id") == conversation_id:
            return i
    return None

def add_new_conversation(history, title, messages):
    """Add a new conversation to history or update existing one"""
    conversation_id = generate_conversation_id(messages)
    
    # Check if conversation already exists
    existing_idx = find_existing_conversation(history, conversation_id)
    
    if existing_idx is not None:
        # Update existing conversation with new messages
        existing_conv = history["conversations"][existing_idx]
        last_saved_count = existing_conv.get("last_saved_count", 0)
        
        # Only append messages that are beyond what was already saved
        if len(messages) > last_saved_count:
            new_messages = messages[last_saved_count:]
            existing_conv["messages"].extend(new_messages)
            existing_conv["last_saved_count"] = len(messages)
            existing_conv["last_updated"] = datetime.now().isoformat()
            save_conversation_history(history)
            return history, "updated"
        else:
            return history, "no_changes"
    else:
        # Create new conversation
        conversation = {
            "title": title,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "last_saved_count": len(messages),
            "messages": messages
        }
        history["conversations"].insert(0, conversation)
        save_conversation_history(history)
        return history, "created"

def auto_save_conversation(conversation_history):
    """Automatically save conversation if there are new messages"""
    if st.session_state.messages:
        first_message = st.session_state.messages[0]["content"]  # Get first user message
        title = first_message[:30] + ("..." if len(first_message) > 30 else "")
        # Convert messages format for saving
        chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
        conversation_history, save_status = add_new_conversation(
            conversation_history,
            title,
            chat_history
        )
        return conversation_history, save_status
    return conversation_history, None

# --- Voice input function (from stream_with_langchain.py) ---
def listen(recognizer):
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            st.toast("🎙 Listening...")
            audio = recognizer.listen(source, timeout=5)
        return recognizer.recognize_google(audio)  # type: ignore
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand."
    except Exception as e:
        return f"Mic error: {e}"

# --- Voice output using macOS 'say' command (from stream_with_langchain.py) ---
def speak(text):
    if not st.session_state.voice_output_enabled:
        st.info("🔇 Voice output is disabled")
        return
        
    if not st.session_state.tts_working:
        st.warning("🔇 TTS system not working. Voice output disabled.")
        return
    
    if st.session_state.is_speaking:
        st.warning("🔁 Already speaking. Please wait.")
        return
        
    try:
        st.session_state.is_speaking = True
        # Show speaking indicator
        st.toast("🔊 Speaking AI response...")
        
        # Use macOS 'say' command
        subprocess.run(['say', text], check=True)
        
        st.toast("✅ Voice output completed!")
        
    except subprocess.CalledProcessError as e:
        st.error(f"TTS command failed: {e}")
        st.session_state.tts_working = False
    except Exception as e:
        st.error(f"Voice output error: {e}")
        st.session_state.tts_working = False
    finally:
        st.session_state.is_speaking = False

# --- Stop speaking function for macOS ---
def stop_speaking():
    try:
        subprocess.run(['pkill', 'say'], check=False)
        st.session_state.is_speaking = False
        st.toast("🔇 Voice stopped!")
    except Exception as e:
        st.error(f"Error stopping voice: {e}")

def main():
    st.set_page_config(
        page_title="Voice Assistant with File Management",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize voice-related session state
    if "voice_output_enabled" not in st.session_state:
        st.session_state.voice_output_enabled = True
    if "tts_working" not in st.session_state:
        st.session_state.tts_working = True
    if "is_speaking" not in st.session_state:
        st.session_state.is_speaking = False
    
    # Initialize chat input state
    if "waiting_for_input" not in st.session_state:
        st.session_state.waiting_for_input = False
    if "current_action" not in st.session_state:
        st.session_state.current_action = None
    if "input_type" not in st.session_state:
        st.session_state.input_type = None
    
    # Initialize recognizer and TTS
    recognizer = sr.Recognizer()
    
    # Load conversation history
    conversation_history = load_conversation_history()
    
    # Initialize session state for messages (using Streamlit's native format)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Track if user has started typing to hide welcome message
    if "user_started_typing" not in st.session_state:
        st.session_state.user_started_typing = False
    
    # Initialize session state for agent
    if "agent" not in st.session_state:
        # LangChain Tools
        calculator = PythonREPLTool()
        wikipedia = WikipediaAPIWrapper(wiki_client=None)
        search = DuckDuckGoSearchResults()

        tools = [
            Tool(name="Calculator", func=calculator.run, description="useful for when you need to use python to answer a question. You should input python code"),
            Tool(name="Wikipedia", func=wikipedia.run, description="Useful for when you need to look up a topic, country or person on wikipedia"),
            Tool(name="Search", func=search.run, description="Useful for finding CURRENT and REAL-TIME information from the internet. Use this for: latest news, current weather, live stock prices, recent events, breaking news, today's information, current trends, live data, or any information that changes frequently,even give images and videos. Be specific with your search query.")
        ]

        # Load Mistral via Ollama with better configuration for code generation
        llm = Ollama(
            model="llama3.2",
            temperature=0.1,  # Lower temperature for more focused responses
            top_p=0.9,
            repeat_penalty=1.1
        )
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize agent with better configuration for code generation
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Better for code generation
            memory=memory,
            verbose=False,  # Reduce verbosity
            handle_parsing_errors=True,
            max_iterations=5,  # Reduce iterations for faster responses
            max_execution_time=30,
            agent_kwargs={
                "prefix": """You are a helpful AI assistant that specializes in providing clear, concise Python code examples. 
                When asked for code, always provide the actual code first, then a brief explanation if needed.
                Keep explanations short and focus on the code itself.
                Use proper Python syntax and include comments where helpful."""
            }
        )

        st.session_state.agent = agent
    
    # Track if we need to save (only save when there are new messages)
    if "needs_save" not in st.session_state:
        st.session_state.needs_save = False
    
    # Sidebar for conversation history and voice controls
    with st.sidebar:
        st.markdown("<h1 style='font-size: 40px;'>🎙️ Ask llama</h1>", unsafe_allow_html=True)
        st.caption("With File Management & GitHub Integration")

        # Add Clear Main Chat button
        if st.button("New tab"):
            st.session_state.messages = []
            st.session_state.user_started_typing = False
            st.session_state.waiting_for_input = False
            st.session_state.current_action = None
            st.session_state.input_type = None
            st.rerun()
        
        # Create tabs in sidebar for better organization
        sidebar_tab1, sidebar_tab2, sidebar_tab3 = st.sidebar.tabs(["🔊 Voice", "📜 History", "⚙️ Config"])
        
        # --- TAB 1: VOICE SETTINGS ---
        with sidebar_tab1:
            st.subheader("🔊 Voice Settings")
            
            st.subheader("🎙️ Voice Input")
            if st.button("Use 🎙️", use_container_width=True):
                user_input = listen(recognizer)
                if user_input and not user_input.lower().startswith("sorry") and not user_input.lower().startswith("mic error"):
                    # Set flag to hide welcome message immediately
                    st.session_state.user_started_typing = True
                    
                    # Process the voice input
                    process_voice_input(user_input, conversation_history)
                    
                    st.rerun()
                else:
                    st.error("Voice input failed. Please try again.")

            # Voice toggle
            voice_toggle = st.toggle("Enable Voice Output", value=st.session_state.voice_output_enabled)
            if voice_toggle != st.session_state.voice_output_enabled:
                st.session_state.voice_output_enabled = voice_toggle
                st.rerun()

            # Stop voice button
            if st.button("🔇 Stop Voice", use_container_width=True):
                stop_speaking()

            # Show speaking status
            if st.session_state.is_speaking:
                st.warning("🔊 Currently speaking...")
        
        # --- TAB 2: CONVERSATION HISTORY ---
        with sidebar_tab2:
            st.subheader("📜 Conversation History")
            
            # Dropdown menu for saved conversations
            conversation_titles = []
            for conv in conversation_history["conversations"]:
                title = conv.get("title", "Untitled")
                # Add timestamp if available
                if "last_updated" in conv:
                    timestamp = conv["last_updated"][:10]  # First 10 chars of date
                    conversation_titles.append(f"{title} (Updated: {timestamp})")
                elif "timestamp" in conv:
                    timestamp = conv["timestamp"][:10]  # First 10 chars of date
                    conversation_titles.append(f"{title} ({timestamp})")
                else:
                    conversation_titles.append(title)
            
            selected_idx = None
            if conversation_titles:
                selected_title = st.selectbox("Select a conversation to load", conversation_titles, key="history_selectbox")
                selected_idx = conversation_titles.index(selected_title)
                if st.button("Load Conversation"):
                    conv = conversation_history["conversations"][selected_idx]
                    # Convert saved format to Streamlit format
                    st.session_state.messages = [{"role": role, "content": content} for role, content in conv.get("messages", [])]
                    st.session_state.current_conversation = selected_idx
                    st.rerun()
            else:
                st.info("No saved conversations.")
            
            # --- Save Conversation Button ---
            if st.button("Save Conversation"):
                if st.session_state.messages:
                    first_message = st.session_state.messages[0]["content"]
                    title = first_message[:30] + ("..." if len(first_message) > 30 else "")
                    chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
                    conversation_history, save_status = add_new_conversation(
                        conversation_history,
                        title,
                        chat_history
                    )
                    st.success("Conversation saved!")
                else:
                    st.warning("No conversation to save.")
        
        # --- TAB 3: CONFIGURATION ---
        with sidebar_tab3:
            st.subheader("⚙️ Configuration")
         
            # Environment check and editor
            st.subheader("🔧 Environment")
            # Load from session_state or os.environ
            env_keys = ["GITHUB_TOKEN", "GITHUB_OWNER", "GITHUB_REPO_NAME"]
            env_values = {}
            for key in env_keys:
                env_values[key] = st.session_state.get(key, os.getenv(key, ""))
            
            with st.form("env_form"):
                github_token = st.text_input("GITHUB_TOKEN", value=env_values["GITHUB_TOKEN"], type="password")
                github_owner = st.text_input("GITHUB_OWNER", value=env_values["GITHUB_OWNER"]) 
                github_repo = st.text_input("GITHUB_REPO_NAME", value=env_values["GITHUB_REPO_NAME"]) 
                submitted = st.form_submit_button("Save Environment")
                if submitted:
                    # Save to .env file
                    with open(".env", "w") as f:
                        f.write(f"GITHUB_TOKEN={github_token}\n")
                        f.write(f"GITHUB_OWNER={github_owner}\n")
                        f.write(f"GITHUB_REPO_NAME={github_repo}\n")
                    # Update session_state and os.environ, ensure no None values
                    st.session_state["GITHUB_TOKEN"] = github_token or ""
                    st.session_state["GITHUB_OWNER"] = github_owner or ""
                    st.session_state["GITHUB_REPO_NAME"] = github_repo or ""
                    os.environ["GITHUB_TOKEN"] = github_token or ""
                    os.environ["GITHUB_OWNER"] = github_owner or ""
                    os.environ["GITHUB_REPO_NAME"] = github_repo or ""
                    st.success("Environment variables saved and updated!")
            # Show current status
            env_vars = {
                "GITHUB_TOKEN": "✅ Set" if github_token else "❌ Missing",
                "GITHUB_OWNER": "✅ Set" if github_owner else "❌ Missing", 
                "GITHUB_REPO_NAME": "✅ Set" if github_repo else "❌ Missing",
            }
            for var, status in env_vars.items():
                st.write(f"{var}: {status}")
            
            # Check if Ansible is available
            st.subheader("Ansible Status")
            try:
                subprocess.run(["ansible", "--version"], capture_output=True, check=True)
                st.success("✅ Ansible is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                st.error("❌ Ansible not found. Install with: pip install ansible")
    
    # Display chat messages from session state using Streamlit's native chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show input instructions if waiting for input
    if st.session_state.waiting_for_input:
        st.info(f"💡 Please provide the required information for: {st.session_state.current_action}")
        
        if st.session_state.input_type == "file_creation":
            st.write("**Example:** `hello.py` or `Create a Python file called hello.py`")
        elif st.session_state.input_type == "github_push":
            st.write("**Example:** `hello.py Update the file` or `filename commit message`")
        elif st.session_state.input_type == "complete_workflow":
            st.write("**Example:** `hello.py` or `Create a Python file called hello.py`")
    
    # After displaying chat messages and before chat input:
    if is_github_push_active() and get_github_push_state()['step'] == 'upload_file':
        render_github_file_uploader()

    # Chat input using Streamlit's native chat input
    if user_prompt := st.chat_input("What would you like to do? (Try 'help' for commands)"):
        # Set flag to hide welcome message immediately
        st.session_state.user_started_typing = True
        
        # Process the text input
        process_text_input(user_prompt, conversation_history)
        
        st.rerun()
    
    # Auto-save conversation if needed (after the main chat area is rendered)
    if st.session_state.needs_save:
        if st.session_state.messages:
            first_message = st.session_state.messages[0]["content"]  # Get first user message
            title = first_message[:30] + ("..." if len(first_message) > 30 else "")
            # Convert messages format for saving
            chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages]
            conversation_history, save_status = auto_save_conversation(conversation_history)
        st.session_state.needs_save = False  # Reset the flag

    # Footer with quick commands
    st.markdown("---")
    st.markdown("**💡 Quick Commands:** `help` | `create file` | `push to github` | `trigger workflow` | `check status`")

def process_voice_input(user_input, conversation_history):
    """Process voice input and handle both regular chat and file management commands"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the message
    if not st.session_state.waiting_for_input:
        # Process new command
        result = process_chat_message(user_input)
        
        if result["action"] == "regular_agent":
            # Use the regular agent for general questions
            with st.spinner('Generating response...'):
                response = st.session_state.agent.run(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                speak(response)
        elif result["action"] == "code_generation":
            # Use specialized code generation
            with st.spinner('Generating Python code...'):
                response = generate_code_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                speak("Here's the Python code you requested")
        elif result["action"] == "code_to_file_and_github":
            code = get_last_generated_code()
            if not code:
                msg = "No code was generated previously. Please ask for code first."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                speak(msg)
                return
            # Ask for filename if not already in the message
            import re
            match = re.search(r"file(?: called| named)? ([\w\-.]+\.py)", user_input)
            filename = match.group(1) if match else None
            if not filename:
                filename = "generated_code.py"
            with st.spinner(f"Creating {filename} and pushing to GitHub..."):
                try:
                    create_tool = CreateFileTool()
                    create_result = create_tool._run(filename, code)
                    push_tool = PushToGitHubTool()
                    push_result = push_tool._run(filename, "Add file generated from previous code")
                    msg = f"✅ File created and pushed to GitHub!\nFile: {filename}\nGitHub Push: {push_result}"
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                    speak("File created and pushed to GitHub.")
                except Exception as e:
                    err = f"❌ Failed to create and push file: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    with st.chat_message("assistant"):
                        st.markdown(err)
                    speak(err)
            st.session_state.needs_save = True
            return
        elif result["action"] == "file_creation_flow":
            process_file_creation_flow(user_input)
            return
        elif result["action"] == "push_last_file_to_github":
            process_push_last_file_to_github()
            return
        elif result["action"] == "github_push_flow":
            process_github_push_flow(user_input)
            return
        elif result["action"] == "ansible_trigger_workflow":
            with st.spinner("Triggering workflow via Ansible..."):
                result = run_ansible_trigger_workflow()
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Workflow triggered via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"✅ Workflow triggered via Ansible:\n```\n{result}\n```")
                speak("Workflow triggered via Ansible.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_check_latest_status":
            with st.spinner("Checking latest workflow status via Ansible..."):
                result = run_ansible_check_latest_status()
                st.session_state.messages.append({"role": "assistant", "content": f"📊 Latest workflow status via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"📊 Latest workflow status via Ansible:\n```\n{result}\n```")
                speak("Latest workflow status checked.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_download_logs":
            with st.spinner("Downloading workflow logs via Ansible..."):
                result = run_ansible_download_logs()
                log_path = "latest_workflow_logs.zip"
                log_exists = os.path.exists(log_path)
                msg = f"📥 Workflow logs downloaded via Ansible."
                if log_exists:
                    msg += f"\nYou can download the logs below."
                msg += f"\n```\n{result}\n```"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                    if log_exists:
                        with open(log_path, "rb") as f:
                            st.download_button("Download Logs (ZIP)", f, file_name="latest_workflow_logs.zip")
                speak("Workflow logs downloaded.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_remediate_failure":
            with st.spinner("Remediating workflow failure via Ansible..."):
                result = run_ansible_remediate_failure()
                st.session_state.messages.append({"role": "assistant", "content": f"🛠️ Remediation performed via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"🛠️ Remediation performed via Ansible:\n```\n{result}\n```")
                speak("Remediation performed.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_retry_failed_job":
            with st.spinner("Retrying failed workflow job via Ansible..."):
                result = run_ansible_retry_failed_job()
                st.session_state.messages.append({"role": "assistant", "content": f"🔄 Retry of failed job via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"🔄 Retry of failed job via Ansible:\n```\n{result}\n```")
                speak("Retry of failed job performed.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_create_repository":
            repo_name = result.get("repo_name", "new-repo")
            repo_description = result.get("repo_description", "Repository created via Ansible")
            repo_private = result.get("repo_private", False)
            with st.spinner(f"Creating GitHub repository '{repo_name}' via Ansible..."):
                result = run_ansible_create_repository(repo_name, repo_description, repo_private)
                st.session_state.messages.append({"role": "assistant", "content": f"📦 Repository creation via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"📦 Repository creation via Ansible:\n```\n{result}\n```")
                speak(f"Repository {repo_name} created.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "create_readme":
            filename = "README.md"
            content = None
            # Try to get content from the last chat command
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and "content" in msg:
                    content = msg["content"]
                    break
            # Prefer the content from the action dict if present
            if isinstance(result, dict):
                content = result.get("content", content)
            else:
                content = getattr(result, "content", content)
            if not content:
                content = "# Project Title\n\nA short description of the project.\n\n## Features\n- Feature 1\n- Feature 2\n\n## Installation\n\nInstructions here.\n\n## Usage\n\nHow to use this project.\n\n## License\n\nMIT"
            with st.spinner(f"Creating {filename}..."):
                try:
                    create_tool = CreateFileTool()
                    create_result = create_tool._run(filename, content)
                    set_last_created_file(filename, content)
                    msg = f"✅ {filename} created!\n---\n{create_result}"
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                        st.download_button(
                            label=f"Download {filename}",
                            data=content,
                            file_name=filename,
                            mime="text/markdown"
                        )
                        # Direct push to GitHub button
                        if st.button(f"Push {filename} to GitHub", key=f"push_{filename}"):
                            process_push_last_file_to_github()
                            st.stop()
                    # Offer to push to GitHub in chat as well
                    st.session_state.messages.append({"role": "assistant", "content": f"Would you like to push this {filename} to GitHub? (yes/no) Or use the button above."})
                    with st.chat_message("assistant"):
                        st.markdown(f"Would you like to push this {filename} to GitHub? (yes/no) Or use the button above.")
                except Exception as e:
                    err = f"❌ Failed to create {filename}: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    with st.chat_message("assistant"):
                        st.markdown(err)
            st.session_state.needs_save = True
            return
        else:
            # Handle file management commands
            if result["response"]:
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                with st.chat_message("assistant"):
                    st.markdown(result["response"])
                speak(result["response"])
            
            # Set up for input if needed
            if result["needs_input"]:
                st.session_state.waiting_for_input = True
                st.session_state.current_action = result["action"]
                st.session_state.input_type = result["input_type"]
            else:
                # Execute action immediately
                execute_action(result["action"], conversation_history)
    else:
        # Handle input for pending action
        handle_pending_action(user_input, conversation_history)
    
    st.session_state.needs_save = True

def process_text_input(user_prompt, conversation_history):
    """Process text input and handle both regular chat and file management commands"""
    # Always handle file creation flow if active
    if is_file_creation_active():
        process_file_creation_flow(user_prompt)
        return
    # Always handle GitHub push flow if active
    if is_github_push_active():
        process_github_push_flow(user_prompt)
        return
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Process the message
    if not st.session_state.waiting_for_input:
        # Process new command
        result = process_chat_message(user_prompt)
        
        if result["action"] == "regular_agent":
            # Use the regular agent for general questions
            with st.spinner('Generating response...'):
                response = st.session_state.agent.run(user_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                speak(response)
        elif result["action"] == "code_generation":
            # Use specialized code generation
            with st.spinner('Generating Python code...'):
                response = generate_code_response(user_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                speak("Here's the Python code you requested")
        elif result["action"] == "code_to_file_and_github":
            code = get_last_generated_code()
            if not code:
                msg = "No code was generated previously. Please ask for code first."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                speak(msg)
                return
            # Ask for filename if not already in the message
            import re
            match = re.search(r"file(?: called| named)? ([\w\-.]+\.py)", user_prompt)
            filename = match.group(1) if match else None
            if not filename:
                filename = "generated_code.py"
            with st.spinner(f"Creating {filename} and pushing to GitHub..."):
                try:
                    create_tool = CreateFileTool()
                    create_result = create_tool._run(filename, code)
                    push_tool = PushToGitHubTool()
                    push_result = push_tool._run(filename, "Add file generated from previous code")
                    msg = f"✅ File created and pushed to GitHub!\nFile: {filename}\nGitHub Push: {push_result}"
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                    speak("File created and pushed to GitHub.")
                except Exception as e:
                    err = f"❌ Failed to create and push file: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    with st.chat_message("assistant"):
                        st.markdown(err)
                    speak(err)
            st.session_state.needs_save = True
            return
        elif result["action"] == "file_creation_flow":
            process_file_creation_flow(user_prompt)
            return
        elif result["action"] == "push_last_file_to_github":
            process_push_last_file_to_github()
            return
        elif result["action"] == "github_push_flow":
            process_github_push_flow(user_prompt)
            return
        elif result["action"] == "ansible_trigger_workflow":
            with st.spinner("Triggering workflow via Ansible..."):
                result = run_ansible_trigger_workflow()
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Workflow triggered via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"✅ Workflow triggered via Ansible:\n```\n{result}\n```")
                speak("Workflow triggered via Ansible.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_check_latest_status":
            with st.spinner("Checking latest workflow status via Ansible..."):
                result = run_ansible_check_latest_status()
                st.session_state.messages.append({"role": "assistant", "content": f"📊 Latest workflow status via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"📊 Latest workflow status via Ansible:\n```\n{result}\n```")
                speak("Latest workflow status checked.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_download_logs":
            with st.spinner("Downloading workflow logs via Ansible..."):
                result = run_ansible_download_logs()
                log_path = "latest_workflow_logs.zip"
                log_exists = os.path.exists(log_path)
                msg = f"📥 Workflow logs downloaded via Ansible."
                if log_exists:
                    msg += f"\nYou can download the logs below."
                msg += f"\n```\n{result}\n```"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                    if log_exists:
                        with open(log_path, "rb") as f:
                            st.download_button("Download Logs (ZIP)", f, file_name="latest_workflow_logs.zip")
                speak("Workflow logs downloaded.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_remediate_failure":
            with st.spinner("Remediating workflow failure via Ansible..."):
                result = run_ansible_remediate_failure()
                st.session_state.messages.append({"role": "assistant", "content": f"🛠️ Remediation performed via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"🛠️ Remediation performed via Ansible:\n```\n{result}\n```")
                speak("Remediation performed.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_retry_failed_job":
            with st.spinner("Retrying failed workflow job via Ansible..."):
                result = run_ansible_retry_failed_job()
                st.session_state.messages.append({"role": "assistant", "content": f"🔄 Retry of failed job via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"🔄 Retry of failed job via Ansible:\n```\n{result}\n```")
                speak("Retry of failed job performed.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "ansible_create_repository":
            repo_name = result.get("repo_name", "new-repo")
            repo_description = result.get("repo_description", "Repository created via Ansible")
            repo_private = result.get("repo_private", False)
            with st.spinner(f"Creating GitHub repository '{repo_name}' via Ansible..."):
                result = run_ansible_create_repository(repo_name, repo_description, repo_private)
                st.session_state.messages.append({"role": "assistant", "content": f"📦 Repository creation via Ansible:\n```\n{result}\n```"})
                with st.chat_message("assistant"):
                    st.markdown(f"📦 Repository creation via Ansible:\n```\n{result}\n```")
                speak(f"Repository {repo_name} created.")
            st.session_state.needs_save = True
            return
        elif result["action"] == "create_readme":
            filename = "README.md"
            content = None
            # Try to get content from the last chat command
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and "content" in msg:
                    content = msg["content"]
                    break
            # Prefer the content from the action dict if present
            if isinstance(result, dict):
                content = result.get("content", content)
            else:
                content = getattr(result, "content", content)
            if not content:
                content = "# Project Title\n\nA short description of the project.\n\n## Features\n- Feature 1\n- Feature 2\n\n## Installation\n\nInstructions here.\n\n## Usage\n\nHow to use this project.\n\n## License\n\nMIT"
            with st.spinner(f"Creating {filename}..."):
                try:
                    create_tool = CreateFileTool()
                    create_result = create_tool._run(filename, content)
                    set_last_created_file(filename, content)
                    msg = f"✅ {filename} created!\n---\n{create_result}"
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    with st.chat_message("assistant"):
                        st.markdown(msg)
                        st.download_button(
                            label=f"Download {filename}",
                            data=content,
                            file_name=filename,
                            mime="text/markdown"
                        )
                        # Direct push to GitHub button
                        if st.button(f"Push {filename} to GitHub", key=f"push_{filename}"):
                            process_push_last_file_to_github()
                            st.stop()
                    # Offer to push to GitHub in chat as well
                    st.session_state.messages.append({"role": "assistant", "content": f"Would you like to push this {filename} to GitHub? (yes/no) Or use the button above."})
                    with st.chat_message("assistant"):
                        st.markdown(f"Would you like to push this {filename} to GitHub? (yes/no) Or use the button above.")
                except Exception as e:
                    err = f"❌ Failed to create {filename}: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    with st.chat_message("assistant"):
                        st.markdown(err)
            st.session_state.needs_save = True
            return
        else:
            # Handle file management commands
            if result["response"]:
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                with st.chat_message("assistant"):
                    st.markdown(result["response"])
                speak(result["response"])
            
            # Set up for input if needed
            if result["needs_input"]:
                st.session_state.waiting_for_input = True
                st.session_state.current_action = result["action"]
                st.session_state.input_type = result["input_type"]
            else:
                # Execute action immediately
                execute_action(result["action"], conversation_history)
    else:
        # Handle input for pending action
        handle_pending_action(user_prompt, conversation_history)
    
    st.session_state.needs_save = True

def execute_action(action, conversation_history):
    """Execute the specified action"""
    if action == "trigger_workflow":
        with st.spinner("Triggering workflow..."):
            try:
                trigger_tool = TriggerWorkflowTool()
                trigger_result = trigger_tool._run()
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Workflow triggered: {trigger_result}"})
                with st.chat_message("assistant"):
                    st.markdown(f"✅ Workflow triggered: {trigger_result}")
                speak(f"Workflow triggered: {trigger_result}")
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to trigger workflow: {str(e)}"})
                with st.chat_message("assistant"):
                    st.markdown(f"❌ Failed to trigger workflow: {str(e)}")
                speak(f"Failed to trigger workflow: {str(e)}")
    
    elif action == "check_status":
        with st.spinner("Checking workflow status via Ansible..."):
            result = run_ansible_check_latest_status()
            st.session_state.messages.append({"role": "assistant", "content": f"📊 Workflow Status via Ansible:\n```\n{result}\n```"})
            with st.chat_message("assistant"):
                st.markdown(f"📊 Workflow Status via Ansible:\n```\n{result}\n```")
            speak(f"Workflow status checked. {result[:100]}...")
    elif action == "ansible_trigger_workflow":
        with st.spinner("Triggering workflow via Ansible..."):
            result = run_ansible_trigger_workflow()
            st.session_state.messages.append({"role": "assistant", "content": f"✅ Workflow triggered via Ansible:\n```\n{result}\n```"})
            with st.chat_message("assistant"):
                st.markdown(f"✅ Workflow triggered via Ansible:\n```\n{result}\n```")
            speak("Workflow triggered via Ansible.")
    elif action == "ansible_check_latest_status":
        with st.spinner("Checking latest workflow status via Ansible..."):
            result = run_ansible_check_latest_status()
            st.session_state.messages.append({"role": "assistant", "content": f"📊 Latest workflow status via Ansible:\n```\n{result}\n```"})
            with st.chat_message("assistant"):
                st.markdown(f"📊 Latest workflow status via Ansible:\n```\n{result}\n```")
            speak("Latest workflow status checked.")
    elif action == "ansible_download_logs":
        with st.spinner("Downloading workflow logs via Ansible..."):
            result = run_ansible_download_logs()
            log_path = "latest_workflow_logs.zip"
            log_exists = os.path.exists(log_path)
            msg = f"📥 Workflow logs downloaded via Ansible."
            if log_exists:
                msg += f"\nYou can download the logs below."
            msg += f"\n```\n{result}\n```"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
                if log_exists:
                    with open(log_path, "rb") as f:
                        st.download_button("Download Logs (ZIP)", f, file_name="latest_workflow_logs.zip")
            speak("Workflow logs downloaded.")
    elif action == "ansible_remediate_failure":
        with st.spinner("Remediating workflow failure via Ansible..."):
            result = run_ansible_remediate_failure()
            st.session_state.messages.append({"role": "assistant", "content": f"🛠️ Remediation performed via Ansible:\n```\n{result}\n```"})
            with st.chat_message("assistant"):
                st.markdown(f"🛠️ Remediation performed via Ansible:\n```\n{result}\n```")
            speak("Remediation performed.")
    elif action == "ansible_retry_failed_job":
        with st.spinner("Retrying failed workflow job via Ansible..."):
            result = run_ansible_retry_failed_job()
            st.session_state.messages.append({"role": "assistant", "content": f"🔄 Retry of failed job via Ansible:\n```\n{result}\n```"})
            with st.chat_message("assistant"):
                st.markdown(f"🔄 Retry of failed job via Ansible:\n```\n{result}\n```")
            speak("Retry of failed job performed.")
    elif action == "ansible_create_repository":
        with st.spinner("Creating GitHub repository via Ansible..."):
            result = run_ansible_create_repository()
            st.session_state.messages.append({"role": "assistant", "content": f"📦 Repository creation via Ansible:\n```\n{result}\n```"})
            with st.chat_message("assistant"):
                st.markdown(f"📦 Repository creation via Ansible:\n```\n{result}\n```")
            speak("Repository created.")
    elif action == "create_readme":
        filename = "README.md"
        content = None
        # Try to get content from the last chat command
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "content" in msg:
                content = msg["content"]
                break
        # Prefer the content from the action dict if present
        if isinstance(action, dict):
            content = action.get("content", content)
        else:
            content = getattr(action, "content", content)
        if not content:
            content = "# Project Title\n\nA short description of the project.\n\n## Features\n- Feature 1\n- Feature 2\n\n## Installation\n\nInstructions here.\n\n## Usage\n\nHow to use this project.\n\n## License\n\nMIT"
        with st.spinner(f"Creating {filename}..."):
            try:
                create_tool = CreateFileTool()
                create_result = create_tool._run(filename, content)
                set_last_created_file(filename, content)
                msg = f"✅ {filename} created!\n---\n{create_result}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                    st.download_button(
                        label=f"Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/markdown"
                    )
                    # Direct push to GitHub button
                    if st.button(f"Push {filename} to GitHub", key=f"push_{filename}"):
                        process_push_last_file_to_github()
                        st.stop()
                # Offer to push to GitHub in chat as well
                st.session_state.messages.append({"role": "assistant", "content": f"Would you like to push this {filename} to GitHub? (yes/no) Or use the button above."})
                with st.chat_message("assistant"):
                    st.markdown(f"Would you like to push this {filename} to GitHub? (yes/no) Or use the button above.")
            except Exception as e:
                err = f"❌ Failed to create {filename}: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.markdown(err)
        st.session_state.needs_save = True
        return

def generate_code_response(user_input):
    """Generate code response using a specialized prompt"""
    try:
        # Create a specialized prompt for code generation
        code_prompt = f"""You are a Python programming expert. The user asked: "{user_input}"

Please provide ONLY the Python code that solves this problem. Include:
1. The actual Python code with proper syntax
2. Brief comments explaining key parts
3. An example of how to use the code

Keep explanations minimal and focus on the code itself. Format the response with proper code blocks.

Here's the Python code:"""

        # Use direct Ollama call for better code generation
        import requests
        
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2:latest",
            "prompt": code_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(ollama_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if "response" in result:
            set_last_generated_code(result["response"])
            return result["response"]
        else:
            return "I couldn't generate the code. Please try again."
            
    except Exception as e:
        print(f"Code generation failed: {e}")
        return f"Code generation failed: {str(e)}"

def handle_pending_action(user_input, conversation_history):
    """Handle input for pending action"""
    # Check if the last assistant message was a push prompt and user said yes
    if user_input.strip().lower() in ("yes", "y", "ok", "sure", "push", "confirm"):
        # Look for a recent assistant message that is a push prompt
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and "Would you like to push" in msg["content"]:
                process_push_last_file_to_github()
                return
    if st.session_state.current_action == "create_file":
        filename, content = extract_file_info_from_message(user_input)
        
        with st.spinner("Creating file..."):
            try:
                create_tool = CreateFileTool()
                create_result = create_tool._run(filename, content)
                st.session_state.messages.append({"role": "assistant", "content": f"✅ File created: {create_result}"})
                with st.chat_message("assistant"):
                    st.markdown(f"✅ File created: {create_result}")
                speak(f"File created: {create_result}")
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to create file: {str(e)}"})
                with st.chat_message("assistant"):
                    st.markdown(f"❌ Failed to create file: {str(e)}")
                speak(f"Failed to create file: {str(e)}")
        
        st.session_state.waiting_for_input = False
        st.session_state.current_action = None
        st.session_state.input_type = None
    
    elif st.session_state.current_action == "push_github":
        # Extract filename and commit message from prompt
        words = user_input.split()
        filename = words[0] if words else "file.txt"
        commit_message = " ".join(words[1:]) if len(words) > 1 else "Update file"
        
        with st.spinner("Pushing to GitHub..."):
            try:
                push_tool = PushToGitHubTool()
                push_result = push_tool._run(filename, commit_message)
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Pushed to GitHub: {push_result}"})
                with st.chat_message("assistant"):
                    st.markdown(f"✅ Pushed to GitHub: {push_result}")
                speak(f"Pushed to GitHub: {push_result}")
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to push to GitHub: {str(e)}"})
                with st.chat_message("assistant"):
                    st.markdown(f"❌ Failed to push to GitHub: {str(e)}")
                speak(f"Failed to push to GitHub: {str(e)}")
        
        st.session_state.waiting_for_input = False
        st.session_state.current_action = None
        st.session_state.input_type = None
    
    elif st.session_state.current_action == "complete_workflow":
        filename, content = extract_file_info_from_message(user_input)
        commit_message = "Add new file"
        
        with st.spinner("Executing complete workflow..."):
            try:
                results = execute_complete_workflow(filename, content, commit_message)
                result_text = "\n".join([f"• {result}" for result in results])
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Complete workflow executed:\n{result_text}"})
                with st.chat_message("assistant"):
                    st.markdown(f"✅ Complete workflow executed:\n{result_text}")
                speak(f"Complete workflow executed successfully")
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to execute workflow: {str(e)}"})
                with st.chat_message("assistant"):
                    st.markdown(f"❌ Failed to execute workflow: {str(e)}")
                speak(f"Failed to execute workflow: {str(e)}")
        
        st.session_state.waiting_for_input = False
        st.session_state.current_action = None
        st.session_state.input_type = None

def process_file_creation_flow(user_input):
    # Always show user input in chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    state = get_file_creation_state()
    import re
    # Step 1: Ask for filename
    if state['step'] == 'ask_filename':
        match = re.search(r"([\w\-.]+\.(py|js|html|css|json|txt|md))", user_input)
        filename = match.group(1) if match else None
        if filename:
            state['filename'] = filename
            state['step'] = 'ask_content_source'
        else:
            st.session_state.messages.append({"role": "assistant", "content": "What should the filename be? (e.g. hello.py)"})
            with st.chat_message("assistant"):
                st.markdown("What should the filename be? (e.g. hello.py)")
            return
    # Step 2: Ask for content source
    if state['step'] == 'ask_content_source':
        st.session_state.messages.append({"role": "assistant", "content": "How do you want to provide the file content?\n1. Use previous code\n2. Generate new code\n3. I'll type/paste it myself\n(Reply with 1, 2, or 3)"})
        with st.chat_message("assistant"):
            st.markdown("How do you want to provide the file content?\n1. Use previous code\n2. Generate new code\n3. I'll type/paste it myself\n(Reply with 1, 2, or 3)")
        state['step'] = 'await_content_source_choice'
        return
    if state['step'] == 'await_content_source_choice':
        choice = user_input.strip().lower()
        if choice in ('1', 'previous', 'use previous code'):
            code = get_last_generated_code()
            if not code:
                msg = "No code was generated previously. Please ask for code first or choose another option."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                reset_file_creation_state()
                return
            state['content_source'] = 'existing'
            state['content'] = code
            state['step'] = 'show_code_and_confirm'
        elif choice in ('2', 'generate', 'generate new code'):
            state['content_source'] = 'generate'
            state['step'] = 'ask_generate_description'
            st.session_state.messages.append({"role": "assistant", "content": "What should the file do? (Describe the code to generate)"})
            with st.chat_message("assistant"):
                st.markdown("What should the file do? (Describe the code to generate)")
            return
        elif choice in ('3', 'type', 'paste', 'myself', "i'll type", "i'll paste"):
            state['content_source'] = 'user'
            state['step'] = 'await_user_content'
            st.session_state.messages.append({"role": "assistant", "content": "Please type or paste the file content now."})
            with st.chat_message("assistant"):
                st.markdown("Please type or paste the file content now.")
            return
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please reply with 1 (previous code), 2 (generate), or 3 (type/paste)."})
            with st.chat_message("assistant"):
                st.markdown("Please reply with 1 (previous code), 2 (generate), or 3 (type/paste).")
            return
    if state['step'] == 'await_user_content':
        state['content'] = user_input
        state['step'] = 'show_code_and_confirm'
    if state['step'] == 'ask_generate_description':
        desc = user_input.strip()
        if desc:
            code = generate_code_response(desc)
            state['content'] = code
            state['step'] = 'show_code_and_confirm'
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please describe what the file should do."})
            with st.chat_message("assistant"):
                st.markdown("Please describe what the file should do.")
            return
    # New step: show code and ask for confirmation
    if state['step'] == 'show_code_and_confirm':
        code = state['content']
        st.session_state.messages.append({"role": "assistant", "content": f"Here is the code that will be used for the file:\n\n```python\n{code}\n```\n\nDo you want to create the file with this code? (yes/no)"})
        with st.chat_message("assistant"):
            st.markdown(f"Here is the code that will be used for the file:\n\n```python\n{code}\n```\n\nDo you want to create the file with this code? (yes/no)")
        state['step'] = 'await_code_confirm'
        return
    if state['step'] == 'await_code_confirm':
        if user_input.strip().lower() in ('yes', 'y', 'ok', 'sure', 'create', 'confirm'):  # Accept various confirmations
            state['step'] = 'create_file_confirm'
        else:
            st.session_state.messages.append({"role": "assistant", "content": "File creation cancelled."})
            with st.chat_message("assistant"):
                st.markdown("File creation cancelled.")
            state['step'] = 'done'
        return
    if state['step'] == 'create_file_confirm':
        filename = state['filename']
        content = state['content']
        with st.spinner(f"Creating file {filename}..."):
            try:
                create_tool = CreateFileTool()
                create_result = create_tool._run(filename, content)
                set_last_created_file(filename, content)
                msg = f"✅ File {filename} created!\n---\n{create_result}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                    # Download button
                    st.download_button(
                        label=f"Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/plain"
                    )
                state['step'] = 'done'
            except Exception as e:
                err = f"❌ Failed to create file: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.markdown(err)
                state['step'] = 'done'
        st.session_state.needs_save = True
        return

def process_github_push_flow(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    state = get_github_push_state()
    
    # Step 1: Ask for source
    if state['step'] == 'ask_source':
        st.session_state.messages.append({"role": "assistant", "content": "How do you want to push to GitHub?\n1. Upload an existing file\n2. Paste content with filename\n3. Use the last created file\n(Reply with 1, 2, or 3)"})
        with st.chat_message("assistant"):
            st.markdown("How do you want to push to GitHub?\n1. Upload an existing file\n2. Paste content with filename\n3. Use the last created file\n(Reply with 1, 2, or 3)")
        state['step'] = 'await_source_choice'
        return
    
    # Step 2: Handle source choice
    if state['step'] == 'await_source_choice':
        choice = user_input.strip().lower()
        if choice in ('1', 'upload', 'existing file'):
            state['source'] = 'upload'
            state['step'] = 'upload_file'
            st.session_state.messages.append({"role": "assistant", "content": "Please upload your file using the file uploader below."})
            with st.chat_message("assistant"):
                st.markdown("Please upload your file using the file uploader below.")
            # The uploader will be rendered in the main script
            return
        elif choice in ('2', 'paste', 'content'):
            state['source'] = 'paste'
            state['step'] = 'ask_filename'
            st.session_state.messages.append({"role": "assistant", "content": "What should the filename be? (e.g. my_script.py)"})
            with st.chat_message("assistant"):
                st.markdown("What should the filename be? (e.g. my_script.py)")
            return
        elif choice in ('3', 'last', 'last file', 'recent'):
            last_file = get_last_created_file()
            if not last_file:
                msg = "No file was created recently. Please create a file first or choose another option."
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                reset_github_push_state()
                return
            state['source'] = 'last_file'
            state['filename'] = last_file['filename']
            state['content'] = last_file['content']
            state['step'] = 'push_to_github'
            # Auto-trigger push
            process_github_push(state['filename'], state['content'])
            return
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please reply with 1 (upload), 2 (paste), or 3 (last file)."})
            with st.chat_message("assistant"):
                st.markdown("Please reply with 1 (upload), 2 (paste), or 3 (last file).")
            return
    
    # Step 3: Handle filename for paste option
    if state['step'] == 'ask_filename':
        import re
        match = re.search(r"([\w\-.]+\.(py|js|html|css|json|txt|md|java|cpp|c|php|rb|go|rs|swift|kt|scala|r|m|sql|sh|bat|yml|yaml|xml|csv|tsv))", user_input)
        filename = match.group(1) if match else None
        if filename:
            state['filename'] = filename
            state['step'] = 'ask_content'
            st.session_state.messages.append({"role": "assistant", "content": "Please paste the file content now."})
            with st.chat_message("assistant"):
                st.markdown("Please paste the file content now.")
            return
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please provide a valid filename with extension (e.g. my_script.py)"})
            with st.chat_message("assistant"):
                st.markdown("Please provide a valid filename with extension (e.g. my_script.py)")
            return
    
    # Step 4: Handle content for paste option
    if state['step'] == 'ask_content':
        if user_input.strip():
            state['content'] = user_input
            state['step'] = 'push_to_github'
            # Auto-trigger push
            process_github_push(state['filename'], state['content'])
            return
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please provide the file content."})
            with st.chat_message("assistant"):
                st.markdown("Please provide the file content.")
            return

def process_github_push(filename, content):
    """Actually push the file to GitHub"""
    with st.spinner(f"Pushing {filename} to GitHub..."):
        try:
            push_tool = PushToGitHubTool()
            push_result = push_tool._run(filename, f"Add {filename} via assistant")
            msg = f"✅ File {filename} pushed to GitHub!\nGitHub Push: {push_result}"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
        except Exception as e:
            err = f"❌ Failed to push file: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.markdown(err)
    st.session_state.needs_save = True
    reset_github_push_state()

def render_github_file_uploader():
    """Renders a file uploader for GitHub push."""
    uploaded_file = st.file_uploader("Choose a file to upload", type=["txt", "py", "js", "html", "css", "json", "md", "java", "cpp", "c", "php", "rb", "go", "rs", "swift", "kt", "scala", "r", "m", "sql", "sh", "bat", "yml", "yaml", "xml", "csv", "tsv"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        if st.button(f"Upload and Push {filename}"):
            content = uploaded_file.getvalue()
            # Write the uploaded content to disk so the GitHub tool can find it
            with open(filename, "wb") as f:
                f.write(content)
            try:
                push_tool = PushToGitHubTool()
                push_result = push_tool._run(filename, f"Add file uploaded via assistant: {filename}")
                msg = f"✅ File {filename} pushed to GitHub!\nGitHub Push: {push_result}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                speak(f"File {filename} pushed to GitHub.")
            except Exception as e:
                err = f"❌ Failed to push file: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.markdown(err)
                speak(err)

if __name__ == "__main__":
    main()
