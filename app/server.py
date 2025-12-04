"""
Paper2Code AgentOS Server

This module sets up and exposes the Paper2Code agent team via the Agno AgentOS.
"""

import os
from agno.os import AgentOS
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv

from app.main import create_paper2code_agent

# Load environment variables from .env file
load_dotenv()

# Get GitHub token from environment
github_token = os.getenv("GITHUB_TOKEN")

# Create the main Paper2Code agent instance
# We expose the 'team' attribute of the agent to AgentOS
paper2code_agent_instance = create_paper2code_agent(github_token=github_token)

# Set up the AgentOS
agent_os = AgentOS(
    agents=[paper2code_agent_instance.team],
    db=SqliteDb(db_file="tmp/paper2code_agentos.db"),
)

# Get the FastAPI app
app = agent_os.get_app()
