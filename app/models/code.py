"""
Data models for code implementations.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    JULIA = "julia"


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    SKLEARN = "sklearn"
    HUGGINGFACE = "huggingface"
    LANGCHAIN = "langchain"


class Dependency(BaseModel):
    name: str
    version: str
    pip_url: Optional[str] = None
    description: Optional[str] = None


class CodeFile(BaseModel):
    name: str
    path: str
    content: str
    language: Language
    description: Optional[str] = None
    imports: List[str] = []
    functions: List[str] = []


class CodeImplementation(BaseModel):
    main_language: Language
    framework: Optional[Framework] = None
    dependencies: List[Dependency] = []
    files: List[CodeFile] = []
    tests: List[CodeFile] = []
    documentation: Dict[str, str] = {}
    setup_instructions: str = ""
    example_usage: Optional[str] = None