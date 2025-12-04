"""
Paper2Code - AI-Powered Scientific Paper to Code Implementation Agent

This package provides the main components for converting scientific papers
into production-ready code implementations.
"""

__version__ = "0.1.0"
__author__ = "Paper2Code Team"
__email__ = "team@paper2code.ai"

# Main exports
from .main import Paper2CodeAgent, create_paper2code_agent, main
from .pipeline import PaperProcessingPipeline, create_pipeline, PipelineConfig
from .prompt_manager import PromptManager, prompt_manager

__all__ = [
    "Paper2CodeAgent",
    "create_paper2code_agent", 
    "main",
    "PaperProcessingPipeline",
    "create_pipeline",
    "PipelineConfig",
    "PromptManager",
    "prompt_manager"
]