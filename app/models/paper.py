"""
Data models for scientific papers.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class PaperMetadata(BaseModel):
    title: str
    authors: List[Author]
    abstract: str
    published_date: Optional[datetime] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    keywords: List[str] = []
    categories: List[str] = []


class Algorithm(BaseModel):
    name: str
    description: str
    pseudocode: Optional[str] = None
    complexity: Optional[str] = None
    parameters: List[Dict[str, Any]] = []


class Experiment(BaseModel):
    name: str
    description: str
    datasets: List[str]
    metrics: List[str]
    baseline_results: Optional[Dict[str, Any]] = None


class Paper(BaseModel):
    metadata: PaperMetadata
    algorithms: List[Algorithm] = []
    experiments: List[Experiment] = []
    full_text: Optional[str] = None
    pdf_path: Optional[str] = None
    source_url: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }