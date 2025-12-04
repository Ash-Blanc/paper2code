"""
Paper Analysis Agent

This agent extracts and structures information from scientific papers,
supporting multiple input formats (PDF, arXiv URLs, DOI identifiers).
"""

import os
import re
import hashlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from agno import agent
from agno.models.openrouter import OpenRouter
from agno.tools.file import FileTools
from agno.tools.web import WebTools
from agno.tools.search import DuckDuckGoSearchTools

from ..models.paper import Paper, PaperMetadata, Author, Algorithm, Experiment

logger = logging.getLogger(__name__)


@dataclass
class PaperAnalysisResult:
    """Result from paper analysis"""
    paper: Paper
    confidence_score: float
    processing_time: float
    input_type: str
    raw_content: str


class PaperAnalysisAgent:
    """Agent for analyzing and extracting information from scientific papers"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.llm = OpenRouter(model=model)
        self.file_tools = FileTools()
        self.web_tools = WebTools()
        self.search_tools = DuckDuckGoSearchTools()
        
        # Create agent with structured output
        self.agent = agent(
            name="paper_analyzer",
            model=self.llm,
            tools=[self.file_tools, self.web_tools, self.search_tools],
            output_schema=Paper,
            description="Extract and structure information from scientific papers"
        )
    
    def detect_input_type(self, input_data: str) -> str:
        """Detect the type of input provided"""
        if input_data.lower().startswith(('http://arxiv.org/', 'https://arxiv.org/')):
            return 'arxiv_url'
        elif input_data.lower().startswith(('http://', 'https://')):
            # Check if it's a DOI resolver or other URL
            if 'doi.org' in input_data.lower():
                return 'doi_url'
            else:
                return 'url'
        elif input_data.lower().startswith('10.'):  # DOI format
            return 'doi'
        elif input_data.lower().endswith('.pdf'):
            return 'pdf'
        else:
            # Try to determine if it's a DOI
            if re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', input_data, re.IGNORECASE):
                return 'doi'
            else:
                return 'unknown'
    
    def extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            # Use file tools to read PDF
            result = self.file_tools.read(file_path=pdf_path)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def fetch_arxiv_content(self, arxiv_url: str) -> str:
        """Fetch content from arXiv URL"""
        try:
            # Extract arXiv ID from URL
            arxiv_id = arxiv_url.split('/')[-1]
            
            # Use web tools to fetch content
            result = self.web_tools.fetch(url=f"https://arxiv.org/abs/{arxiv_id}")
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            logger.error(f"Error fetching arXiv content: {e}")
            raise
    
    def fetch_doi_content(self, doi: str) -> str:
        """Fetch content from DOI"""
        try:
            # Use DOI resolver to get content
            result = self.web_tools.fetch(url=f"https://doi.org/{doi}")
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            logger.error(f"Error fetching DOI content: {e}")
            raise
    
    def generate_paper_hash(self, input_data: str, input_type: str) -> str:
        """Generate consistent hash for paper input"""
        content = f"{input_type}:{input_data}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def analyze_paper_content(self, content: str) -> Paper:
        """Analyze paper content and extract structured information"""
        try:
            # Use the agent to extract paper information
            result = self.agent.run(
                prompt=f"""
                Analyze the following scientific paper content and extract structured information:
                
                Content:
                {content[:8000]}  # Limit context length
                
                Please extract:
                1. Paper metadata (title, authors, publication date, abstract)
                2. Key algorithms and methods described
                3. Experimental setups and datasets
                4. Main findings and contributions
                5. Research domain and category
                
                Return the information in structured format.
                """
            )
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing paper content: {e}")
            raise
    
    def run(self, input_data: str, input_type: Optional[str] = None) -> PaperAnalysisResult:
        """Main method to analyze a paper"""
        start_time = datetime.now()
        
        # Detect input type if not provided
        if input_type is None:
            input_type = self.detect_input_type(input_data)
        
        logger.info(f"Analyzing paper with input type: {input_type}")
        
        # Extract content based on input type
        raw_content = ""
        
        try:
            if input_type == 'pdf':
                if not os.path.exists(input_data):
                    raise FileNotFoundError(f"PDF file not found: {input_data}")
                raw_content = self.extract_pdf_content(input_data)
            elif input_type == 'arxiv_url':
                raw_content = self.fetch_arxiv_content(input_data)
            elif input_type in ['doi', 'doi_url']:
                raw_content = self.fetch_doi_content(input_data)
            elif input_type == 'url':
                raw_content = self.web_tools.fetch(url=input_data).content
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            # Analyze the content
            paper = self.analyze_paper_content(raw_content)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence score (simplified)
            confidence_score = self._calculate_confidence_score(paper, raw_content)
            
            return PaperAnalysisResult(
                paper=paper,
                confidence_score=confidence_score,
                processing_time=processing_time,
                input_type=input_type,
                raw_content=raw_content
            )
            
        except Exception as e:
            logger.error(f"Error in paper analysis: {e}")
            raise
    
    def _calculate_confidence_score(self, paper: Paper, content: str) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.0
        
        # Check for required fields
        if paper.metadata.title:
            score += 0.2
        if paper.metadata.authors:
            score += 0.2
        if paper.metadata.abstract:
            score += 0.2
        if paper.algorithms:
            score += 0.2
        if paper.experiments:
            score += 0.2
        
        # Adjust based on content length
        if len(content) > 1000:
            score += 0.1
        
        return min(1.0, score)