"""
Data models for repository configurations.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RepositoryVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"


class RepositoryConfig(BaseModel):
    name: str
    description: str
    visibility: RepositoryVisibility = RepositoryVisibility.PUBLIC
    license: str = "MIT"
    gitignore_template: Optional[str] = None
    ci_cd_enabled: bool = True
    auto_documentation: bool = True
    
    class Config:
        use_enum_values = True


class Repository(BaseModel):
    """GitHub repository model"""
    
    name: str
    full_name: str
    description: Optional[str] = None
    html_url: str
    clone_url: str
    language: Optional[str] = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    url: Optional[str] = None
    
    class Config:
        use_enum_values = True