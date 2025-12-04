"""
GitHub tools for searching and analyzing repositories.
"""
from typing import List, Optional, Dict, Any, AsyncGenerator
import logging
import aiohttp
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)


class GitHubSearch:
    """Tool for searching GitHub repositories."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Paper2Code-Agent",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    async def search_repositories(self, query: str, language: Optional[str] = None, 
                                min_stars: int = 10) -> List[Dict[str, Any]]:
        """Search GitHub repositories."""
        try:
            search_query = f"{query} stars:>{min_stars}"
            if language:
                search_query += f" language:{language}"
            
            url = "https://api.github.com/search/repositories"
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": 10,
            }
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    repositories = []
                    for repo in data.get("items", []):
                        repositories.append({
                            "name": repo["name"],
                            "full_name": repo["full_name"],
                            "description": repo["description"],
                            "html_url": repo["html_url"],
                            "clone_url": repo["clone_url"],
                            "stargazers_count": repo["stargazers_count"],
                            "language": repo["language"],
                            "updated_at": repo["updated_at"],
                            "implementation_score": self._calculate_implementation_score(repo),
                        })
                    
                    return repositories
                    
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {e}")
            raise
    
    def _calculate_implementation_score(self, repo: Dict[str, Any]) -> float:
        """Calculate implementation quality score."""
        score = 0.0
        
        # Stars (0-0.3)
        stars = repo.get("stargazers_count", 0)
        score += min(stars / 1000, 0.3)
        
        # Language match (0-0.2)
        if repo.get("language"):
            score += 0.2
        
        # Recent updates (0-0.2)
        # This would require parsing the updated_at date
        score += 0.1
        
        # Has README (0-0.1)
        if repo.get("description"):
            score += 0.1
        
        # Has issues (0-0.1)
        # This would require checking issues
        score += 0.1
        
        return min(score, 1.0)


class GitHubTools:
    """GitHub tools for repository analysis."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Paper2Code-Agent",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    async def get_repository(self, full_name: str) -> Dict[str, Any]:
        """Get repository details."""
        try:
            url = f"https://api.github.com/repos/{full_name}"
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except Exception as e:
            logger.error(f"Error getting repository {full_name}: {e}")
            raise
    
    async def analyze_repository_structure(self, full_name: str) -> Dict[str, Any]:
        """Analyze repository structure."""
        try:
            # Get repository contents
            contents = await self._get_repository_contents(full_name)
            
            structure = {
                "main_language": self._detect_main_language(contents),
                "framework": self._detect_framework(contents),
                "files": contents,
                "tests": self._find_test_files(contents),
                "documentation": self._find_documentation(contents),
                "setup_instructions": self._extract_setup_instructions(contents),
                "example_usage": self._extract_example_usage(contents),
            }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing repository structure: {e}")
            raise
    
    async def _get_repository_contents(self, full_name: str, path: str = "") -> List[Dict[str, Any]]:
        """Get repository contents."""
        try:
            url = f"https://api.github.com/repos/{full_name}/contents/{path}"
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    contents = await response.json()
                    
                    # Convert to list if it's a single file
                    if isinstance(contents, dict):
                        return [contents]
                    
                    return contents
                    
        except Exception as e:
            logger.error(f"Error getting repository contents: {e}")
            raise
    
    def _detect_main_language(self, contents: List[Dict[str, Any]]) -> str:
        """Detect main programming language."""
        language_count = {}
        
        for item in contents:
            if item.get("type") == "file":
                ext = item.get("name", "").split(".")[-1].lower()
                language_count[ext] = language_count.get(ext, 0) + 1
        
        if language_count:
            return max(language_count.items(), key=lambda x: x[1])[0]
        
        return "unknown"
    
    def _detect_framework(self, contents: List[Dict[str, Any]]) -> Optional[str]:
        """Detect framework used in the repository."""
        # Check for common framework indicators
        framework_indicators = {
            "pytorch": ["requirements.txt", "setup.py", "pytorch", "torch"],
            "tensorflow": ["tensorflow", "tf", "tf.keras"],
            "sklearn": ["sklearn", "scikit-learn"],
            "huggingface": ["transformers", "huggingface"],
            "langchain": ["langchain"],
        }
        
        file_names = [item.get("name", "").lower() for item in contents if item.get("type") == "file"]
        
        for framework, indicators in framework_indicators.items():
            if any(indicator in " ".join(file_names) for indicator in indicators):
                return framework
        
        return None
    
    def _find_test_files(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find test files in the repository."""
        test_files = []
        
        for item in contents:
            if item.get("type") == "file":
                name = item.get("name", "").lower()
                if any(test_indicator in name for test_indicator in ["test_", "spec_", "_test", "_spec"]):
                    test_files.append(item)
        
        return test_files
    
    def _find_documentation(self, contents: List[Dict[str, Any]]) -> Dict[str, str]:
        """Find documentation files."""
        documentation = {}
        
        for item in contents:
            if item.get("type") == "file":
                name = item.get("name", "").lower()
                if name in ["readme.md", "readme.rst", "readme.txt"]:
                    documentation["readme"] = item.get("download_url", "")
                elif name in ["api.md", "api.rst", "api.txt"]:
                    documentation["api"] = item.get("download_url", "")
                elif name in ["changelog.md", "changelog.rst", "changelog.txt"]:
                    documentation["changelog"] = item.get("download_url", "")
        
        return documentation
    
    def _extract_setup_instructions(self, contents: List[Dict[str, Any]]) -> str:
        """Extract setup instructions from README."""
        # This would typically involve reading the README file
        # For now, return a placeholder
        return "Setup instructions would be extracted from README.md"
    
    def _extract_example_usage(self, contents: List[Dict[str, Any]]) -> Optional[str]:
        """Extract example usage from repository."""
        # This would involve looking for example files or notebooks
        return None
    
    async def extract_dependencies(self, full_name: str) -> List[Dict[str, Any]]:
        """Extract dependencies from repository."""
        try:
            dependencies = []
            
            # Check for requirements.txt
            requirements = await self._get_file_content(full_name, "requirements.txt")
            if requirements:
                deps = self._parse_requirements(requirements)
                dependencies.extend(deps)
            
            # Check for setup.py
            setup_py = await self._get_file_content(full_name, "setup.py")
            if setup_py:
                deps = self._parse_setup_py(setup_py)
                dependencies.extend(deps)
            
            # Check for pyproject.toml
            pyproject = await self._get_file_content(full_name, "pyproject.toml")
            if pyproject:
                deps = self._parse_pyproject_toml(pyproject)
                dependencies.extend(deps)
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Error extracting dependencies: {e}")
            raise
    
    async def _get_file_content(self, full_name: str, filename: str) -> Optional[str]:
        """Get file content from repository."""
        try:
            url = f"https://api.github.com/repos/{full_name}/contents/{filename}"
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        import base64
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        return content
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return None
    
    def _parse_requirements(self, content: str) -> List[Dict[str, Any]]:
        """Parse requirements.txt content."""
        dependencies = []
        
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Simple parsing - in production, use a proper requirements parser
                if ">=" in line:
                    name, version = line.split(">=", 1)
                    dependencies.append({
                        "name": name.strip(),
                        "version": version.strip(),
                        "pip_url": f"pip install {name.strip()}>={version.strip()}",
                    })
                elif "==" in line:
                    name, version = line.split("==", 1)
                    dependencies.append({
                        "name": name.strip(),
                        "version": version.strip(),
                        "pip_url": f"pip install {name.strip()}=={version.strip()}",
                    })
                else:
                    dependencies.append({
                        "name": line.strip(),
                        "version": "latest",
                        "pip_url": f"pip install {line.strip()}",
                    })
        
        return dependencies
    
    def _parse_setup_py(self, content: str) -> List[Dict[str, Any]]:
        """Parse setup.py content."""
        # This would require proper Python parsing
        return []
    
    def _parse_pyproject_toml(self, content: str) -> List[Dict[str, Any]]:
        """Parse pyproject.toml content."""
        # This would require TOML parsing
        return []