"""
Research Agent

This agent finds similar papers and existing implementations on GitHub,
analyzes patterns, and extracts best practices for code generation.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import arxiv
from duckduckgo_search import DDGS

from agno import agent
from agno.models.openrouter import OpenRouter
from app.tools.github_tools import GitHubTools

from ..models.paper import Paper
from ..models.repository import RepositoryConfig, Repository

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Result from research analysis"""
    similar_papers: List[Dict[str, Any]]
    github_repositories: List[Repository]
    common_patterns: Dict[str, Any]
    best_practices: Dict[str, Any]
    research_confidence: float
    processing_time: float


class ResearchAgent:
    """Agent for researching similar papers and existing implementations"""
    
    def __init__(self, github_token: str, model: str = "gpt-4o"):
        self.model = model
        self.llm = OpenRouter(model=model)
        self.github_tools = GitHubTools(github_token=github_token)
        
        # Create agent with GitHub tools
        self.agent = agent(
            name="research_analyzer",
            model=self.llm,
            tools=[self.github_tools],
            description="Research similar papers and existing GitHub implementations"
        )
    
    def search_similar_papers(self, paper: Paper) -> List[Dict[str, Any]]:
        """Search for similar papers based on title, authors, and keywords"""
        try:
            # Extract search terms from paper
            search_terms = self._extract_search_terms(paper)
            
            # Search for similar papers using web search
            similar_papers = []
            
            # Search by title
            title_results = self._search_by_title(paper.metadata.title)
            similar_papers.extend(title_results)
            
            # Search by keywords
            if search_terms:
                keyword_results = self._search_by_keywords(search_terms)
                similar_papers.extend(keyword_results)
            
            # Remove duplicates and rank by relevance
            unique_papers = self._deduplicate_and_rank(similar_papers)
            
            return unique_papers[:10]  # Return top 10 results
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            return []
    
    def search_github_repositories(self, paper: Paper) -> List[Repository]:
        """Search GitHub for implementations related to the paper"""
        try:
            # Extract search terms for GitHub
            search_terms = self._extract_github_search_terms(paper)
            
            repositories = []
            
            # Search by paper title
            title_repos = self.github_tools.search_repositories(
                query=f"{paper.metadata.title} implementation",
                limit=5
            )
            repositories.extend(title_repos)
            
            # Search by algorithms/methods
            for algorithm in paper.algorithms:
                algo_repos = self.github_tools.search_repositories(
                    query=f"{algorithm.name} implementation",
                    limit=3
                )
                repositories.extend(algo_repos)
            
            # Search by research domain
            domain_repos = self.github_tools.search_repositories(
                query=f"{paper.metadata.domain} code",
                limit=5
            )
            repositories.extend(domain_repos)
            
            # Remove duplicates and analyze repositories
            unique_repos = self._deduplicate_repositories(repositories)
            analyzed_repos = self._analyze_repositories(unique_repos)
            
            return analyzed_repos[:15]  # Return top 15 repositories
            
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {e}")
            return []
    
    def extract_common_patterns(self, repositories: List[Repository]) -> Dict[str, Any]:
        """Extract common patterns from existing implementations"""
        try:
            if not repositories:
                return {}
            
            # Analyze repository structures
            structures = [repo.structure for repo in repositories if repo.structure]
            
            # Analyze programming languages
            languages = {}
            for repo in repositories:
                if repo.language:
                    languages[repo.language] = languages.get(repo.language, 0) + 1
            
            # Analyze dependencies
            dependencies = {}
            for repo in repositories:
                if repo.dependencies:
                    for dep in repo.dependencies:
                        dependencies[dep.name] = dependencies.get(dep.name, 0) + 1
            
            # Analyze common file patterns
            file_patterns = {}
            for structure in structures:
                for file_path in structure.keys():
                    pattern = self._categorize_file_pattern(file_path)
                    file_patterns[pattern] = file_patterns.get(pattern, 0) + 1
            
            return {
                'programming_languages': languages,
                'dependencies': dependencies,
                'file_patterns': file_patterns,
                'repository_structures': structures,
                'total_repositories_analyzed': len(repositories)
            }
            
        except Exception as e:
            logger.error(f"Error extracting common patterns: {e}")
            return {}
    
    def extract_best_practices(self, paper: Paper, repositories: List[Repository]) -> Dict[str, Any]:
        """Extract best practices from existing implementations"""
        try:
            if not repositories:
                return {}
            
            # Use agent to analyze best practices
            practices_text = self._generate_best_practices_text(paper, repositories)
            
            # Parse best practices
            best_practices = self.agent.run(
                prompt=f"""
                Analyze the following information about existing implementations and extract best practices:
                
                Paper: {paper.metadata.title}
                Abstract: {paper.metadata.abstract}
                
                Repositories analyzed: {len(repositories)}
                
                Implementation details:
                {practices_text}
                
                Extract best practices for:
                1. Code structure and organization
                2. Testing approaches
                3. Documentation standards
                4. Performance optimization
                5. Error handling
                6. Dependencies management
                
                Return structured best practices.
                """
            )
            
            return best_practices
            
        except Exception as e:
            logger.error(f"Error extracting best practices: {e}")
            return {}
    
    def _extract_search_terms(self, paper: Paper) -> List[str]:
        """Extract search terms from paper"""
        terms = []
        
        # Add title words
        if paper.metadata.title:
            terms.extend(paper.metadata.title.lower().split())
        
        # Add author names
        if paper.metadata.authors:
            for author in paper.metadata.authors:
                terms.extend(author.name.lower().split())
        
        # Add keywords from abstract
        if paper.metadata.abstract:
            # Extract meaningful keywords (simplified)
            words = paper.metadata.abstract.lower().split()
            meaningful_words = [word for word in words if len(word) > 3]
            terms.extend(meaningful_words)
        
        # Add algorithm names
        for algorithm in paper.algorithms:
            terms.extend(algorithm.name.lower().split())
        
        return list(set(terms))  # Remove duplicates
    
    def _extract_github_search_terms(self, paper: Paper) -> List[str]:
        """Extract GitHub-specific search terms"""
        terms = self._extract_search_terms(paper)
        
        # Add GitHub-specific terms
        terms.extend(['implementation', 'code', 'source', 'github'])
        
        # Add domain-specific terms
        if paper.metadata.domain:
            terms.append(paper.metadata.domain.lower())
        
        return list(set(terms))
    
    def _search_by_title(self, title: str) -> List[Dict[str, Any]]:
        """Search papers by title using arXiv and web search"""
        papers = []
        try:
            # Search arXiv
            search = arxiv.Search(
                query=title,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'url': result.entry_id,
                    'published': result.published.isoformat(),
                    'source': 'arxiv',
                    'relevance': 0.9
                })
        except Exception as e:
            logger.warning(f"arXiv search failed for title '{title}': {e}")

        # Fallback to DuckDuckGo web search
        try:
            with DDGS() as ddgs:
                ddgs_results = list(ddgs.text(title, max_results=5))
                for r in ddgs_results:
                    papers.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'abstract': r.get('body', ''),
                        'source': 'web',
                        'relevance': 0.6
                    })
        except Exception as e:
            logger.warning(f"Web search failed for title '{title}': {e}")

        return papers
    
    def _search_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search papers by keywords using arXiv and web search"""
        papers = []
        query = ' '.join(keywords[:10])  # Limit to avoid overly long queries
        try:
            # arXiv search
            search = arxiv.Search(
                query=query,
                max_results=8,
                sort_by=arxiv.SortCriterion.Relevance
            )
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'url': result.entry_id,
                    'published': result.published.isoformat(),
                    'source': 'arxiv',
                    'relevance': 0.7
                })
        except Exception as e:
            logger.warning(f"arXiv keyword search failed: {e}")

        # DuckDuckGo web search fallback
        try:
            with DDGS() as ddgs:
                ddgs_results = list(ddgs.text(query, max_results=8))
                for r in ddgs_results:
                    papers.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'abstract': r.get('body', ''),
                        'source': 'web',
                        'relevance': 0.5
                    })
        except Exception as e:
            logger.warning(f"Web keyword search failed: {e}")

        return papers
    
    def _deduplicate_and_rank(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank papers by relevance"""
        # Simple deduplication based on title
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        # Sort by relevance (simplified)
        unique_papers.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        return unique_papers
    
    def _deduplicate_repositories(self, repositories: List[Repository]) -> List[Repository]:
        """Remove duplicate repositories"""
        seen_urls = set()
        unique_repos = []
        
        for repo in repositories:
            if repo.url not in seen_urls:
                seen_urls.add(repo.url)
                unique_repos.append(repo)
        
        return unique_repos
    
    def _analyze_repositories(self, repositories: List[Repository]) -> List[Repository]:
        """Analyze repositories and extract additional information"""
        analyzed_repos = []
        
        for repo in repositories:
            try:
                # Analyze repository structure
                structure = self.github_tools.analyze_repository_structure(repo.url)
                
                # Extract dependencies
                dependencies = self.github_tools.extract_dependencies(repo.url)
                
                # Update repository with analyzed information
                repo.structure = structure
                repo.dependencies = dependencies
                
                analyzed_repos.append(repo)
                
            except Exception as e:
                logger.warning(f"Error analyzing repository {repo.url}: {e}")
                analyzed_repos.append(repo)  # Keep original repo if analysis fails
        
        return analyzed_repos
    
    def _categorize_file_pattern(self, file_path: str) -> str:
        """Categorize file path into pattern type"""
        if file_path.endswith('.py'):
            return 'python_file'
        elif file_path.endswith('.js') or file_path.endswith('.ts'):
            return 'javascript_file'
        elif file_path.endswith('.md'):
            return 'markdown_file'
        elif file_path.endswith('.json'):
            return 'json_file'
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return 'yaml_file'
        elif 'test' in file_path.lower():
            return 'test_file'
        elif 'doc' in file_path.lower():
            return 'documentation_file'
        else:
            return 'other_file'
    
    def _generate_best_practices_text(self, paper: Paper, repositories: List[Repository]) -> str:
        """Generate text for best practices analysis"""
        text = f"Found {len(repositories)} repositories:\n\n"
        
        for i, repo in enumerate(repositories[:5]):  # Limit to top 5
            text += f"Repository {i+1}:\n"
            text += f"  URL: {repo.url}\n"
            text += f"  Language: {repo.language}\n"
            text += f"  Stars: {repo.stars}\n"
            text += f"  Description: {repo.description}\n"
            
            if repo.dependencies:
                text += f"  Dependencies: {', '.join([dep.name for dep in repo.dependencies])}\n"
            
            text += "\n"
        
        return text
    
    def run(self, paper: Paper) -> ResearchResult:
        """Main method to research paper and existing implementations"""
        start_time = datetime.now()
        
        logger.info(f"Researching similar papers and implementations for: {paper.metadata.title}")
        
        try:
            # Search similar papers
            similar_papers = self.search_similar_papers(paper)
            
            # Search GitHub repositories
            github_repositories = self.search_github_repositories(paper)
            
            # Extract common patterns
            common_patterns = self.extract_common_patterns(github_repositories)
            
            # Extract best practices
            best_practices = self.extract_best_practices(paper, github_repositories)
            
            # Calculate research confidence
            research_confidence = self._calculate_research_confidence(
                similar_papers, github_repositories
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ResearchResult(
                similar_papers=similar_papers,
                github_repositories=github_repositories,
                common_patterns=common_patterns,
                best_practices=best_practices,
                research_confidence=research_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in research analysis: {e}")
            raise
    
    def _calculate_research_confidence(self, similar_papers: List, repositories: List) -> float:
        """Calculate confidence score for research results"""
        score = 0.0
        
        # Papers found
        if similar_papers:
            score += 0.3
        
        # Repositories found
        if repositories:
            score += 0.4
            # Bonus for high-quality repositories
            high_quality_repos = [r for r in repositories if r.stars > 10]
            if high_quality_repos:
                score += 0.2
        
        # Content quality
        if len(similar_papers) + len(repositories) > 5:
            score += 0.1
        
        return min(1.0, score)