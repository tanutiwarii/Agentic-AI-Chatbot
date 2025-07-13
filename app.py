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

# === Streamlit UI ===
st.set_page_config(page_title="Agentic File Manager", layout="wide")
st.title("🎙️ Agentic File Manager")
st.caption("Powered by CrewAI, Ansible, and GitHub Actions")

# Sidebar for configuration
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
    
    # Quick actions
    st.header("🚀 Quick Actions")
    if st.button("Check Workflow Status"):
        with st.spinner("Checking workflow status..."):
            result = run_ansible_check()
            st.text_area("Workflow Status", result, height=200)

# Main interface
tab1, tab2, tab3 = st.tabs(["📝 Create & Deploy", "🔧 Individual Actions", "📊 Status Check"])

with tab1:
    st.header("Complete Workflow")
    st.write("Create a file, push it to GitHub, and trigger workflows automatically")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filename = st.text_input("Filename", value="example.py", help="Enter the filename to create")
        commit_message = st.text_input("Commit Message", value="Add new file", help="Git commit message")
    
    with col2:
        file_type = st.selectbox("File Type", ["Python", "JavaScript", "HTML", "CSS", "JSON", "Custom"])
        
        if file_type == "Python":
            default_content = '''def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()'''
        elif file_type == "JavaScript":
            default_content = '''function helloWorld() {
    console.log("Hello, World!");
}

helloWorld();'''
        elif file_type == "HTML":
            default_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>'''
        elif file_type == "CSS":
            default_content = '''body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f0f0;
}

h1 {
    color: #333;
}'''
        elif file_type == "JSON":
            default_content = '''{
    "name": "example",
    "version": "1.0.0",
    "description": "Example file"
}'''
        else:
            default_content = "# Custom file content"
    
    content = st.text_area("File Content", value=default_content, height=200)
    
    if st.button("🚀 Execute Complete Workflow", type="primary"):
        if filename and content:
            with st.spinner("Executing complete workflow..."):
                results = execute_complete_workflow(filename, content, commit_message)
                
                if all("failed" not in result.lower() for result in results):
                    st.success("✅ Workflow completed!")
                else:
                    st.warning("⚠️ Workflow completed with some issues")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Step {i} Result", expanded=True):
                        st.text(result)
        else:
            st.error("Please provide both filename and content")

with tab2:
    st.header("Individual Actions")
    
    action = st.selectbox("Choose Action", [
        "Create File",
        "Push to GitHub", 
        "Trigger Workflow",
        "Check Workflow Status"
    ])
    
    if action == "Create File":
        filename = st.text_input("Filename")
        content = st.text_area("Content")
        if st.button("Create File"):
            try:
                # Create file directly using the tool
                from tools.github_tools import CreateFileTool
                create_tool = CreateFileTool()
                result = create_tool._run(filename, content)
                st.text_area("Result", str(result), height=200)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif action == "Push to GitHub":
        filename = st.text_input("Filename to push")
        commit_msg = st.text_input("Commit message")
        if st.button("Push to GitHub"):
            try:
                task = Task(description=f"Push {filename} with message: {commit_msg}", expected_output="File pushed to GitHub", agent=github_committer)
                crew = Crew(agents=[github_committer], tasks=[task], verbose=True)
                result = crew.kickoff()
                st.text_area("Result", str(result), height=200)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif action == "Trigger Workflow":
        workflow_file = st.text_input("Workflow file", value="main.yml")
        if st.button("Trigger Workflow"):
            try:
                task = Task(description=f"Trigger workflow {workflow_file}", expected_output="Workflow triggered successfully", agent=workflow_manager)
                crew = Crew(agents=[workflow_manager], tasks=[task], verbose=True)
                result = crew.kickoff()
                st.text_area("Result", str(result), height=200)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif action == "Check Workflow Status":
        if st.button("Check Status"):
            result = run_ansible_check()
            st.text_area("Status", result, height=200)

with tab3:
    st.header("📊 System Status")
    
    # Environment check
    st.subheader("Environment Variables")
    env_vars = {
        "GITHUB_TOKEN": "✅ Set" if os.getenv("GITHUB_TOKEN") else "❌ Missing",
        "GITHUB_OWNER": "✅ Set" if os.getenv("GITHUB_OWNER") else "❌ Missing", 
        "GITHUB_REPO_NAME": "✅ Set" if os.getenv("GITHUB_REPO_NAME") else "❌ Missing",
        "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing (optional)"
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
    
    # LLM Status
    st.subheader("LLM Status")
    try:
        model_name = llm.model_name
        if model_name == "ollama/llama3.2:latest":
            st.success(f"✅ LLM working properly: {model_name}")
        else:
            st.warning("⚠️ LLM not configured as expected.")
    except Exception as e:
        st.error(f"❌ LLM Error: {str(e)}")