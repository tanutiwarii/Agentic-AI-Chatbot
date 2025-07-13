# streamlit_app.py (Voice Assistant with Streamlit UI)

import os
import subprocess
import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Crew, Task
from litellm import completion
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from typing import Any, List, Optional

from tools.github_tools import CreateFileTool, PushToGitHubTool, TriggerWorkflowTool, CheckWorkflowStatusTool

load_dotenv()

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

# === LangChain LLM & Memory ===
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize LLM
llm = CustomLLM()

# === CrewAI Agents ===
file_creator = Agent(
    role="FileCreator",
    goal="Generate files based on user input with proper content.",
    backstory="You are an expert at creating well-structured code files based on user requirements.",
    tools=[CreateFileTool()],
    verbose=True,
    llm=llm
)

github_committer = Agent(
    role="GitHubCommitter",
    goal="Commit and push files to GitHub repository.",
    backstory="You are a Git expert who handles version control operations safely and efficiently.",
    tools=[PushToGitHubTool()],
    verbose=True,
    llm=llm
)

workflow_manager = Agent(
    role="WorkflowManager",
    goal="Trigger and monitor GitHub Actions workflows.",
    backstory="You manage CI/CD pipelines and ensure workflows run successfully.",
    tools=[TriggerWorkflowTool(), CheckWorkflowStatusTool()],
    verbose=True,
    llm=llm
)

# === Ansible Integration ===
def run_ansible_check():
    try:
        result = subprocess.run(
            ["ansible-playbook", "ansible/check_workflow.yml"], 
            capture_output=True, 
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Ansible check failed: {e.stderr}"
    except FileNotFoundError:
        return "❌ Ansible not found. Please install Ansible first."

# === Complete Workflow Function ===
def execute_complete_workflow(filename, content, commit_message="Add new file"):
    """Execute the complete workflow: create file → push to GitHub → trigger workflow"""
    
    results = []
    
    # Step 1: Create File
    st.info("📝 Step 1: Creating file...")
    try:
        # Create the file directly using the tool
        from tools.github_tools import CreateFileTool
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
        from tools.github_tools import PushToGitHubTool
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
        from tools.github_tools import TriggerWorkflowTool
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
def process_chat_message(user_message):
    """Process user message and return appropriate response and action"""
    
    message_lower = user_message.lower()
    
    # File creation commands
    if any(word in message_lower for word in ["create", "make", "new"]) and any(word in message_lower for word in ["file", "python", "js", "html", "css", "json"]):
        return {
            "action": "create_file",
            "response": "I'll help you create a file! Please provide the filename and content.",
            "needs_input": True,
            "input_type": "file_creation"
        }
    
    # GitHub push commands
    elif any(word in message_lower for word in ["push", "commit", "upload"]) and any(word in message_lower for word in ["github", "repo", "repository"]):
        return {
            "action": "push_github",
            "response": "I'll help you push files to GitHub! Please provide the filename and commit message.",
            "needs_input": True,
            "input_type": "github_push"
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

⚡ **Workflow Operations:**
- "Trigger workflow" or "Run GitHub Actions"
- "Check workflow status" or "Monitor pipeline"

🔄 **Complete Workflow:**
- "Run complete workflow" or "Execute full pipeline"

💡 **Examples:**
- "Create a Python file called hello.py"
- "Push my changes to GitHub"
- "Trigger the workflow"
- "Check the workflow status"
- "Run the complete workflow"

What would you like to do?""",
            "needs_input": False,
            "input_type": None
        }
    
    # Default response
    else:
        return {
            "action": "unknown",
            "response": "I'm not sure what you'd like to do. Try saying 'help' to see available commands, or ask me to create a file, push to GitHub, trigger a workflow, or check status.",
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

# === Streamlit Chat Interface ===
st.set_page_config(page_title="Agentic File Manager - Chat", layout="wide")
st.title("💬 Agentic File Manager - Chat Interface")
st.caption("Powered by CrewAI, Ansible, and GitHub Actions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize input state
if "waiting_for_input" not in st.session_state:
    st.session_state.waiting_for_input = False
if "current_action" not in st.session_state:
    st.session_state.current_action = None
if "input_type" not in st.session_state:
    st.session_state.input_type = None

# Sidebar for configuration and status
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Make sure to set up your .env file with GitHub credentials")
    
    # LLM Status
    st.subheader("🤖 LLM Status")
    try:
        model_name = llm.model_name
        if model_name == "ollama/llama3.2:latest":
            st.success(f"✅ LLM configured: {model_name}")
        else:
            st.warning("⚠️ LLM not configured as expected.")
    except Exception as e:
        st.error(f"❌ LLM Error: {str(e)}")
    
    # Environment check
    st.subheader("🔧 Environment")
    env_vars = {
        "GITHUB_TOKEN": "✅ Set" if os.getenv("GITHUB_TOKEN") else "❌ Missing",
        "GITHUB_OWNER": "✅ Set" if os.getenv("GITHUB_OWNER") else "❌ Missing", 
        "GITHUB_REPO_NAME": "✅ Set" if os.getenv("GITHUB_REPO_NAME") else "❌ Missing",
    }
    
    for var, status in env_vars.items():
        st.write(f"{var}: {status}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.waiting_for_input = False
        st.session_state.current_action = None
        st.session_state.input_type = None
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to do? (Try 'help' for commands)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process the message
    if not st.session_state.waiting_for_input:
        # Process new command
        result = process_chat_message(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(result["response"])
        
        # Set up for input if needed
        if result["needs_input"]:
            st.session_state.waiting_for_input = True
            st.session_state.current_action = result["action"]
            st.session_state.input_type = result["input_type"]
        else:
            # Execute action immediately
            if result["action"] == "trigger_workflow":
                with st.spinner("Triggering workflow..."):
                    try:
                        from tools.github_tools import TriggerWorkflowTool
                        trigger_tool = TriggerWorkflowTool()
                        trigger_result = trigger_tool._run()
                        st.session_state.messages.append({"role": "assistant", "content": f"✅ Workflow triggered: {trigger_result}"})
                        with st.chat_message("assistant"):
                            st.markdown(f"✅ Workflow triggered: {trigger_result}")
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to trigger workflow: {str(e)}"})
                        with st.chat_message("assistant"):
                            st.markdown(f"❌ Failed to trigger workflow: {str(e)}")
            
            elif result["action"] == "check_status":
                with st.spinner("Checking workflow status..."):
                    try:
                        status_result = run_ansible_check()
                        st.session_state.messages.append({"role": "assistant", "content": f"📊 Workflow Status:\n```\n{status_result}\n```"})
                        with st.chat_message("assistant"):
                            st.markdown(f"📊 Workflow Status:\n```\n{status_result}\n```")
                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to check status: {str(e)}"})
                        with st.chat_message("assistant"):
                            st.markdown(f"❌ Failed to check status: {str(e)}")
    else:
        # Handle input for pending action
        if st.session_state.current_action == "create_file":
            filename, content = extract_file_info_from_message(prompt)
            
            with st.spinner("Creating file..."):
                try:
                    from tools.github_tools import CreateFileTool
                    create_tool = CreateFileTool()
                    create_result = create_tool._run(filename, content)
                    st.session_state.messages.append({"role": "assistant", "content": f"✅ File created: {create_result}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"✅ File created: {create_result}")
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to create file: {str(e)}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"❌ Failed to create file: {str(e)}")
            
            st.session_state.waiting_for_input = False
            st.session_state.current_action = None
            st.session_state.input_type = None
        
        elif st.session_state.current_action == "push_github":
            # Extract filename and commit message from prompt
            words = prompt.split()
            filename = words[0] if words else "file.txt"
            commit_message = " ".join(words[1:]) if len(words) > 1 else "Update file"
            
            with st.spinner("Pushing to GitHub..."):
                try:
                    from tools.github_tools import PushToGitHubTool
                    push_tool = PushToGitHubTool()
                    push_result = push_tool._run(filename, commit_message)
                    st.session_state.messages.append({"role": "assistant", "content": f"✅ Pushed to GitHub: {push_result}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"✅ Pushed to GitHub: {push_result}")
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to push to GitHub: {str(e)}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"❌ Failed to push to GitHub: {str(e)}")
            
            st.session_state.waiting_for_input = False
            st.session_state.current_action = None
            st.session_state.input_type = None
        
        elif st.session_state.current_action == "complete_workflow":
            filename, content = extract_file_info_from_message(prompt)
            commit_message = "Add new file"
            
            with st.spinner("Executing complete workflow..."):
                try:
                    results = execute_complete_workflow(filename, content, commit_message)
                    result_text = "\n".join([f"• {result}" for result in results])
                    st.session_state.messages.append({"role": "assistant", "content": f"✅ Complete workflow executed:\n{result_text}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"✅ Complete workflow executed:\n{result_text}")
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Failed to execute workflow: {str(e)}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"❌ Failed to execute workflow: {str(e)}")
            
            st.session_state.waiting_for_input = False
            st.session_state.current_action = None
            st.session_state.input_type = None

# Show input instructions if waiting for input
if st.session_state.waiting_for_input:
    st.info(f"💡 Please provide the required information for: {st.session_state.current_action}")
    
    if st.session_state.input_type == "file_creation":
        st.write("**Example:** `hello.py` or `Create a Python file called hello.py`")
    elif st.session_state.input_type == "github_push":
        st.write("**Example:** `hello.py Update the file` or `filename commit message`")
    elif st.session_state.input_type == "complete_workflow":
        st.write("**Example:** `hello.py` or `Create a Python file called hello.py`")

# Footer with quick commands
st.markdown("---")
st.markdown("**💡 Quick Commands:** `help` | `create file` | `push to github` | `trigger workflow` | `check status`")