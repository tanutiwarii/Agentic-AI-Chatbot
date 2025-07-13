#!/usr/bin/env python3
"""
Test script to verify CrewAI integration works properly
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task
from app import llm
from tools.github_tools import CreateFileTool

def test_crewai_integration():
    """Test CrewAI integration"""
    load_dotenv()
    
    print("🧪 Testing CrewAI Integration...")
    
    # Create a simple agent
    file_creator = Agent(
        role="FileCreator",
        goal="Generate files based on user input with proper content.",
        backstory="You are an expert at creating well-structured code files based on user requirements.",
        tools=[CreateFileTool()],
        verbose=True,
        llm=llm
    )
    
    # Create a task
    task = Task(
        description="Create a file named test_crewai.py with content: print('Hello from CrewAI!')",
        expected_output="File test_crewai.py created successfully",
        agent=file_creator
    )
    
    # Create a crew
    crew = Crew(agents=[file_creator], tasks=[task], verbose=True)
    
    print("🚀 Executing CrewAI task...")
    
    try:
        result = crew.kickoff()
        print(f"✅ CrewAI Result: {result}")
        
        # Check if file was created
        if os.path.exists("test_crewai.py"):
            print("✅ File was created successfully!")
            with open("test_crewai.py", "r") as f:
                content = f.read()
            print(f"📄 File content: {content}")
        else:
            print("❌ File was not created")
            
    except Exception as e:
        print(f"❌ CrewAI Error: {str(e)}")
        return False
    
    # Clean up
    if os.path.exists("test_crewai.py"):
        os.remove("test_crewai.py")
        print("🧹 Cleaned up test file")
    
    print("✅ CrewAI integration test completed!")
    return True

if __name__ == "__main__":
    test_crewai_integration() 