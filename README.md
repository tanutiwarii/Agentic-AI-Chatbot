# Agentic File Manager

A powerful file management system that combines CrewAI agents, GitHub integration, and Ansible automation to create, deploy, and manage files with automated workflows.

## Features

- 🤖 **CrewAI Agents**: Intelligent agents for file creation, GitHub operations, and workflow management
- 📁 **File Management**: Create files with various templates (Python, JavaScript, HTML, CSS, JSON)
- 🚀 **GitHub Integration**: Automatic commit and push to GitHub repositories
- ⚡ **Workflow Automation**: Trigger and monitor GitHub Actions workflows
- 🔧 **Ansible Integration**: Infrastructure automation and workflow status checking
- 🎨 **Streamlit UI**: Beautiful and intuitive web interface

## Prerequisites

- Python 3.8+
- Git
- Ansible
- GitHub Personal Access Token
- LLM (choose one):
  - OpenAI API key (recommended)
  - Ollama with llama3.2 model (local LLM)
  - Fallback mock LLM (limited functionality)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd new
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```bash
   cp env_example.txt .env
   ```
   
   Edit `.env` with your GitHub credentials and LLM configuration:
   ```env
   # GitHub Configuration
   GITHUB_TOKEN=your_github_personal_access_token
   GITHUB_OWNER=your_github_username
   GITHUB_REPO_NAME=your_repository_name
   GITHUB_BRANCH=main
   
   # LLM Configuration (choose one)
   # Option 1: OpenAI (recommended)
   OPENAI_API_KEY=your_openai_api_key
   
   # Option 2: Ollama (local LLM)
   OLLAMA_BASE_URL=http://localhost:11434
   ```

4. **Install Ansible** (if not already installed):
   ```bash
   pip install ansible
   ```

## GitHub Setup

1. **Create a Personal Access Token**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate a new token with `repo` and `workflow` permissions
   - Copy the token to your `.env` file

2. **Set up your repository**:
   - Create a GitHub repository
   - Update `GITHUB_OWNER` and `GITHUB_REPO_NAME` in your `.env` file
   - Ensure your repository has GitHub Actions enabled

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Complete Workflow

1. **Create & Deploy Tab**: 
   - Enter a filename
   - Choose a file type or write custom content
   - Click "Execute Complete Workflow"
   - The system will automatically:
     - Create the file
     - Push it to GitHub
     - Trigger GitHub Actions workflow

2. **Individual Actions Tab**:
   - Perform individual operations:
     - Create files
     - Push to GitHub
     - Trigger workflows
     - Check workflow status

3. **Status Check Tab**:
   - Monitor environment variables
   - Check Ansible availability
   - View system status

### Example Workflow

1. **Create a Python file**:
   - Filename: `hello.py`
   - Content: 
     ```python
     def hello_world():
         print("Hello, World!")
     
     if __name__ == "__main__":
         hello_world()
     ```

2. **Execute workflow**:
   - The system creates `hello.py`
   - Commits and pushes to GitHub
   - Triggers any configured GitHub Actions

## Project Structure

```
new/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables template
├── README.md             # This file
├── tools/
│   ├── __init__.py
│   └── github_tools.py   # GitHub integration tools
└── ansible/
    └── check_workflow.yml # Ansible playbook for workflow checking
```

## CrewAI Agents

### FileCreator Agent
- **Role**: Creates files with proper content
- **Tools**: CreateFileTool
- **Goal**: Generate well-structured code files

### GitHubCommitter Agent
- **Role**: Manages Git operations
- **Tools**: PushToGitHubTool
- **Goal**: Safely commit and push files to GitHub

### WorkflowManager Agent
- **Role**: Manages CI/CD pipelines
- **Tools**: TriggerWorkflowTool, CheckWorkflowStatusTool
- **Goal**: Trigger and monitor GitHub Actions workflows

## Ansible Integration

The Ansible playbook (`ansible/check_workflow.yml`) provides:
- Environment variable validation
- GitHub Actions status checking
- Workflow run monitoring
- Error handling and reporting

## Troubleshooting

### Common Issues

1. **GitHub Token Error**:
   - Ensure your token has the correct permissions
   - Check that the token is valid and not expired

2. **Ansible Not Found**:
   ```bash
   pip install ansible
   ```

3. **Environment Variables Missing**:
   - Copy `env_example.txt` to `.env`
   - Fill in all required variables

4. **Git Repository Issues**:
   - Ensure your local directory is a Git repository
   - Check that you have a remote origin configured

### Debug Mode

Enable verbose logging by setting `verbose=True` in the agent configurations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. 