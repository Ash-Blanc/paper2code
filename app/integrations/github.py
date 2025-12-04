"""
GitHub Integration for Paper2Code Agent System

This module provides GitHub integration for creating and managing repositories
generated from scientific papers.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import json

from github import Github, GithubException
from github.Repository import Repository
from github.ContentFile import ContentFile

from app.models.paper import PaperMetadata
from app.models.code import CodeImplementation, CodeFile
from app.models.repository import RepositoryConfig, RepositoryVisibility

logger = logging.getLogger(__name__)


class GitHubRepositoryResult:
    """Result of GitHub repository creation operation"""
    
    def __init__(self, success: bool, repository_url: str = None, 
                 repository_name: str = None, clone_url: str = None,
                 errors: List[str] = None, warnings: List[str] = None):
        self.success = success
        self.repository_url = repository_url
        self.repository_name = repository_name
        self.clone_url = clone_url
        self.errors = errors or []
        self.warnings = warnings or []


class GitHubIntegration:
    """GitHub integration for repository creation and management"""
    
    def __init__(self, token: str = None, organization: str = "paper2code-repos"):
        """
        Initialize GitHub integration
        
        Args:
            token: GitHub personal access token
            organization: Organization name for repositories
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.organization = organization
        self.github = None
        self._initialize_github()
    
    def _initialize_github(self):
        """Initialize GitHub client"""
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable or pass token parameter.")
        
        try:
            self.github = Github(self.token)
            # Verify authentication
            user = self.github.get_user()
            logger.info(f"Authenticated as: {user.login}")
            
            # Verify organization access
            try:
                org = self.github.get_organization(self.organization)
                logger.info(f"Organization access verified: {org.login}")
            except GithubException:
                # If organization doesn't exist, use user account
                logger.info(f"Organization '{self.organization}' not found, using user account")
                self.organization = user.login
                
        except GithubException as e:
            logger.error(f"GitHub authentication failed: {e}")
            raise
    
    def create_repository(self, paper_metadata: PaperMetadata, 
                         code_implementation: CodeImplementation,
                         config: RepositoryConfig = None) -> GitHubRepositoryResult:
        """
        Create GitHub repository for paper implementation
        
        Args:
            paper_metadata: Paper metadata
            code_implementation: Generated code implementation
            config: Repository configuration
            
        Returns:
            GitHubRepositoryResult with creation results
        """
        if config is None:
            config = RepositoryConfig()
        
        logger.info(f"Creating GitHub repository for paper: {paper_metadata.title}")
        
        try:
            # Generate repository name
            repo_name = self._generate_repo_name(paper_metadata)
            
            # Create repository
            repository = self._create_github_repository(repo_name, config)
            
            # Create basic repository structure
            self._create_repository_structure(repository, paper_metadata, code_implementation)
            
            # Generate and push README
            self._generate_and_push_readme(repository, paper_metadata, code_implementation)
            
            # Generate basic .gitignore
            self._generate_gitignore(repository, code_implementation)
            
            # Create basic issue template
            self._create_issue_template(repository)
            
            logger.info(f"Repository created successfully: {repository.html_url}")
            
            return GitHubRepositoryResult(
                success=True,
                repository_url=repository.html_url,
                repository_name=repository.name,
                clone_url=repository.clone_url,
                errors=[],
                warnings=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return GitHubRepositoryResult(
                success=False,
                errors=[str(e)],
                warnings=[]
            )
    
    def _generate_repo_name(self, paper_metadata: PaperMetadata) -> str:
        """Generate repository name from paper metadata"""
        # Clean title for repository name
        clean_title = self._clean_string(paper_metadata.title)
        
        # Add year if available
        year_suffix = f"-{paper_metadata.publication_year}" if paper_metadata.publication_year else ""
        
        # Generate final name
        repo_name = f"{clean_title}{year_suffix}".lower()
        
        # Ensure it's not too long
        if len(repo_name) > 100:
            repo_name = repo_name[:100]
        
        # Remove trailing hyphens
        repo_name = repo_name.rstrip('-')
        
        return repo_name
    
    def _clean_string(self, text: str) -> str:
        """Clean string for repository name"""
        # Remove special characters and replace with hyphens
        import re
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text.strip('-')
    
    def _create_github_repository(self, repo_name: str, config: RepositoryConfig) -> Repository:
        """Create GitHub repository"""
        try:
            # Get user or organization
            if self.organization == self.github.get_user().login:
                # Create in user account
                user = self.github.get_user()
                repository = user.create_repo(
                    name=repo_name,
                    description=config.description or f"Code implementation for: {repo_name}",
                    private=config.visibility == RepositoryVisibility.PRIVATE,
                    has_issues=config.has_issues,
                    has_wiki=config.has_wiki,
                    has_projects=config.has_projects,
                    auto_init=True,
                    gitignore_template=config.gitignore_template
                )
            else:
                # Create in organization
                org = self.github.get_organization(self.organization)
                repository = org.create_repo(
                    name=repo_name,
                    description=config.description or f"Code implementation for: {repo_name}",
                    private=config.visibility == RepositoryVisibility.PRIVATE,
                    has_issues=config.has_issues,
                    has_wiki=config.has_wiki,
                    has_projects=config.has_projects,
                    auto_init=True,
                    gitignore_template=config.gitignore_template
                )
            
            logger.info(f"Repository created: {repository.full_name}")
            return repository
            
        except GithubException as e:
            logger.error(f"Failed to create GitHub repository: {e}")
            raise
    
    def _create_repository_structure(self, repository: Repository, 
                                   paper_metadata: PaperMetadata,
                                   code_implementation: CodeImplementation):
        """Create basic repository structure"""
        try:
            # Create src directory
            self._create_directory(repository, "src", "Source code")
            
            # Create tests directory
            self._create_directory(repository, "tests", "Test files")
            
            # Create docs directory
            self._create_directory(repository, "docs", "Documentation")
            
            # Create examples directory
            self._create_directory(repository, "examples", "Usage examples")
            
            # Create paper analysis files
            self._create_paper_analysis_files(repository, paper_metadata)
            
            logger.info("Repository structure created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise
    
    def _create_directory(self, repository: Repository, dir_name: str, description: str):
        """Create directory with README file"""
        # Create README file for directory
        readme_content = f"# {dir_name.title()}\n\n{description}"
        
        try:
            repository.create_file(
                path=f"{dir_name}/README.md",
                message=f"Create {dir_name} directory",
                content=readme_content,
                branch="main"
            )
        except GithubException as e:
            logger.warning(f"Failed to create {dir_name}/README.md: {e}")
    
    def _create_paper_analysis_files(self, repository: Repository, 
                                   paper_metadata: PaperMetadata):
        """Create paper analysis files"""
        try:
            # Paper summary
            summary_content = f"""# Paper Summary

## Title
{paper_metadata.title}

## Authors
{', '.join([author.name for author in paper_metadata.authors])}

## Publication
{paper_metadata.journal or 'Unknown'} ({paper_metadata.publication_year or 'Unknown'})

## Abstract
{paper_metadata.abstract}

## DOI
{paper_metadata.doi or 'Not available'}

## URL
{paper_metadata.url or 'Not available'}

## Research Domain
{paper_metadata.domain or 'Unknown'}
"""
            
            repository.create_file(
                path="docs/paper_summary.md",
                message="Add paper summary",
                content=summary_content,
                branch="main"
            )
            
            # Research questions
            if hasattr(paper_metadata, 'research_questions') and paper_metadata.research_questions:
                questions_content = "# Research Questions\n\n" + "\n".join([f"1. {q}" for q in paper_metadata.research_questions])
                
                repository.create_file(
                    path="docs/research_questions.md",
                    message="Add research questions",
                    content=questions_content,
                    branch="main"
                )
            
            logger.info("Paper analysis files created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create paper analysis files: {e}")
    
    def _generate_and_push_readme(self, repository: Repository,
                                paper_metadata: PaperMetadata,
                                code_implementation: CodeImplementation):
        """Generate and push README file"""
        try:
            # Generate README content
            readme_content = self._generate_readme_content(paper_metadata, code_implementation)
            
            # Create README file
            repository.create_file(
                path="README.md",
                message="Add README",
                content=readme_content,
                branch="main"
            )
            
            logger.info("README file created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create README: {e}")
    
    def _generate_readme_content(self, paper_metadata: PaperMetadata,
                              code_implementation: CodeImplementation) -> str:
        """Generate README content"""
        # Get code files
        code_files = code_implementation.generated_files if hasattr(code_implementation, 'generated_files') else []
        
        # Group files by type
        main_files = [f for f in code_files if f.file_type in ['main', 'algorithm']]
        test_files = [f for f in code_files if f.file_type == 'test']
        example_files = [f for f in code_files if f.file_type == 'example']
        
        # Generate README
        readme_content = f"""# {paper_metadata.title}

> Code implementation for: "{paper_metadata.title}"

## Paper Information

- **Title**: {paper_metadata.title}
- **Authors**: {', '.join([author.name for author in paper_metadata.authors])}
- **Publication**: {paper_metadata.journal or 'Unknown'} ({paper_metadata.publication_year or 'Unknown'})
- **DOI**: {paper_metadata.doi or 'Not available'}
- **URL**: {paper_metadata.url or 'Not available'}
- **Research Domain**: {paper_metadata.domain or 'Unknown'}

## About

This repository contains the code implementation of the research paper "{paper_metadata.title}". The implementation is generated automatically by the Paper2Code agent system.

## Implementation Details

### Programming Language
- **Primary Language**: {code_implementation.language_used if hasattr(code_implementation, 'language_used') else 'Python'}
- **Framework**: {code_implementation.framework_used if hasattr(code_implementation, 'framework_used') else 'Not specified'}

### Code Structure
"""
        
        # Add code structure information
        if main_files:
            readme_content += f"""
#### Main Implementation
{', '.join([f.name for f in main_files])}
"""
        
        if test_files:
            readme_content += f"""
#### Test Files
{', '.join([f.name for f in test_files])}
"""
        
        if example_files:
            readme_content += f"""
#### Example Files
{', '.join([f.name for f in example_files])}
"""
        
        # Add installation instructions
        readme_content += """

## Installation

```bash
# Clone the repository
git clone {repository_url}

# Navigate to the repository
cd {repository_name}

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Import the main implementation
from {main_module} import {main_function}

# Example usage
result = {main_function}({example_input})
```

## Contributing

This repository is automatically generated by the Paper2Code agent system. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{{{paper_citation}},
  title={{{paper_title}}},
  author={{{paper_authors}}},
  journal={{{paper_journal}}},
  year={{{paper_year}}}
}
```

## Paper2Code Agent

This repository was generated using the [Paper2Code](https://github.com/paper2code/paper2code) agent system, which automatically converts scientific papers into production-ready code implementations.
"""
        
        return readme_content
    
    def _generate_gitignore(self, repository: Repository, 
                          code_implementation: CodeImplementation):
        """Generate .gitignore file"""
        try:
            # Get language from code implementation
            language = code_implementation.language_used if hasattr(code_implementation, 'language_used') else 'python'
            
            # Generate gitignore content based on language
            gitignore_content = self._get_gitignore_content(language)
            
            # Create .gitignore file
            repository.create_file(
                path=".gitignore",
                message="Add .gitignore",
                content=gitignore_content,
                branch="main"
            )
            
            logger.info(".gitignore file created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create .gitignore: {e}")
    
    def _get_gitignore_content(self, language: str) -> str:
        """Get gitignore content for specific language"""
        gitignore_templates = {
            'python': """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# PyCharm
.idea/

# VS Code
.vscode/
""",
            'javascript': """# Dependencies
node_modules/
jspm_packages/

# Production builds
/build/
/dist/
/*.prod.js

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# nyc test coverage
.nyc_output

# Grunt intermediate storage
.grunt

# Bower dependency directory
bower_components

# node-waf configuration
.lock-wscript

# Compiled binary addons
build/Release

# Dependency directories
node_modules/
jspm_packages/

# TypeScript v1 declaration files
typings/

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env
.env.test

# parcel-bundler cache
.cache
.parcel-cache

# next.js build output
.next

# nuxt.js build output
.nuxt

# vuepress build output
.vuepress/dist

# Serverless directories
.serverless

# FuseBox cache
.fusebox/

# DynamoDB Local files
.dynamodb/

# TernJS port file
.tern-port

# Stores VSCode versions used for testing VSCode extensions
.vscode-test

# yarn v2
.yarn/cache
.yarn/unplugged
.yarn/build-state.yml
.yarn/install-state.gz
.pnp.*
""",
            'java': """# Compiled class file
*.class

# Log file
*.log

# BlueJ files
*.ctxt

# Mobile Tools for Java (J2ME)
.mtj.tmp/

# Package Files #
*.jar
*.war
*.nar
*.ear
*.zip
*.tar.gz
*.rar

# virtual machine crash logs
hs_err_pid*
""",
            'generic': """# Ignore temporary files
*.tmp
*.temp
*.swp
*.swo
*~

# Ignore OS files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Ignore IDE files
.vscode/
.idea/
*.sublime-*
*.project
*.cproject
*.settings/

# Ignore build artifacts
build/
dist/
out/
target/
bin/
obj/

# Ignore logs
*.log
logs/
log/

# Ignore cache
.cache/
*.cache
"""
        }
        
        return gitignore_templates.get(language.lower(), gitignore_templates['generic'])
    
    def _create_issue_template(self, repository: Repository):
        """Create basic issue template"""
        try:
            # Create issue template directory
            repository.create_file(
                path=".github/ISSUE_TEMPLATE/bug_report.md",
                message="Add bug report template",
                content=self._get_bug_report_template(),
                branch="main"
            )
            
            repository.create_file(
                path=".github/ISSUE_TEMPLATE/feature_request.md",
                message="Add feature request template",
                content=self._get_feature_request_template(),
                branch="main"
            )
            
            logger.info("Issue templates created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create issue templates: {e}")
    
    def _get_bug_report_template(self) -> str:
        """Get bug report template"""
        return """---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run '...'
2. Input '...'
3. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment**
- Operating System:
- Python Version:
- Paper2Code Version:
- Paper Title/DOI:

**Additional context**
Add any other context about the problem here.
"""
    
    def _get_feature_request_template(self) -> str:
        """Get feature request template"""
        return """---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
"""
    
    def get_repository(self, repo_name: str) -> Optional[Repository]:
        """Get repository by name"""
        try:
            if self.organization == self.github.get_user().login:
                user = self.github.get_user()
                return user.get_repo(repo_name)
            else:
                org = self.github.get_organization(self.organization)
                return org.get_repo(repo_name)
        except GithubException:
            return None
    
    def delete_repository(self, repo_name: str) -> bool:
        """Delete repository by name"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                repository.delete()
                logger.info(f"Repository deleted: {repo_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete repository: {e}")
            return False
    
    def get_repository_info(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get repository information"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                return {
                    'name': repository.name,
                    'full_name': repository.full_name,
                    'description': repository.description,
                    'html_url': repository.html_url,
                    'clone_url': repository.clone_url,
                    'ssh_url': repository.ssh_url,
                    'language': repository.language,
                    'size': repository.size,
                    'stargazers_count': repository.stargazers_count,
                    'watchers_count': repository.watchers_count,
                    'forks_count': repository.forks_count,
                    'open_issues_count': repository.open_issues_count,
                    'has_issues': repository.has_issues,
                    'has_projects': repository.has_projects,
                    'has_wiki': repository.has_wiki,
                    'has_pages': repository.has_pages,
                    'archived': repository.archived,
                    'disabled': repository.disabled,
                    'pushed_at': repository.pushed_at,
                    'created_at': repository.created_at,
                    'updated_at': repository.updated_at,
                    'default_branch': repository.default_branch
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get repository info: {e}")
            return None
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """List all repositories"""
        try:
            repositories = []
            
            if self.organization == self.github.get_user().login:
                user = self.github.get_user()
                repos = user.get_repos()
            else:
                org = self.github.get_organization(self.organization)
                repos = org.get_repos()
            
            for repo in repos:
                repositories.append({
                    'name': repo.name,
                    'full_name': repo.full_name,
                    'description': repo.description,
                    'html_url': repo.html_url,
                    'language': repo.language,
                    'size': repo.size,
                    'stargazers_count': repo.stargazers_count,
                    'forks_count': repo.forks_count,
                    'open_issues_count': repo.open_issues_count,
                    'created_at': repo.created_at,
                    'updated_at': repo.updated_at
                })
            
            return repositories
            
        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            return []
    
    def update_repository_description(self, repo_name: str, description: str) -> bool:
        """Update repository description"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                repository.edit(description=description)
                logger.info(f"Repository description updated: {repo_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update repository description: {e}")
            return False
    
    def add_collaborator(self, repo_name: str, username: str, permission: str = 'read') -> bool:
        """Add collaborator to repository"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                repository.add_to_collaborators(username, permission)
                logger.info(f"Collaborator added: {username} to {repo_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to add collaborator: {e}")
            return False
    
    def remove_collaborator(self, repo_name: str, username: str) -> bool:
        """Remove collaborator from repository"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                repository.remove_from_collaborators(username)
                logger.info(f"Collaborator removed: {username} from {repo_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove collaborator: {e}")
            return False
    
    def create_release(self, repo_name: str, tag_name: str, name: str, 
                      body: str, target_commitish: str = 'main') -> Optional[Dict[str, Any]]:
        """Create a new release"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                release = repository.create_git_release(
                    tag_name=tag_name,
                    name=name,
                    description=body,
                    target_commitish=target_commitish,
                    draft=False,
                    prerelease=False
                )
                
                return {
                    'id': release.id,
                    'tag_name': release.tag_name,
                    'name': release.name,
                    'body': release.body,
                    'html_url': release.html_url,
                    'created_at': release.created_at,
                    'published_at': release.published_at,
                    'draft': release.draft,
                    'prerelease': release.prerelease
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create release: {e}")
            return None
    
    def get_releases(self, repo_name: str) -> List[Dict[str, Any]]:
        """Get all releases for a repository"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                releases = []
                for release in repository.get_releases():
                    releases.append({
                        'id': release.id,
                        'tag_name': release.tag_name,
                        'name': release.name,
                        'body': release.body,
                        'html_url': release.html_url,
                        'created_at': release.created_at,
                        'published_at': release.published_at,
                        'draft': release.draft,
                        'prerelease': release.prerelease,
                        'author': release.author.login if release.author else None
                    })
                return releases
            return []
        except Exception as e:
            logger.error(f"Failed to get releases: {e}")
            return []
    
    def close_issues(self, repo_name: str, issue_numbers: List[int]) -> bool:
        """Close multiple issues"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                for issue_number in issue_numbers:
                    issue = repository.get_issue(issue_number)
                    issue.edit(state='closed')
                logger.info(f"Issues closed: {issue_numbers}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to close issues: {e}")
            return False
    
    def get_issues(self, repo_name: str, state: str = 'open') -> List[Dict[str, Any]]:
        """Get issues for a repository"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                issues = []
                for issue in repository.get_issues(state=state):
                    issues.append({
                        'number': issue.number,
                        'title': issue.title,
                        'body': issue.body,
                        'state': issue.state,
                        'html_url': issue.html_url,
                        'created_at': issue.created_at,
                        'updated_at': issue.updated_at,
                        'closed_at': issue.closed_at,
                        'user': issue.user.login if issue.user else None,
                        'assignee': issue.assignee.login if issue.assignee else None,
                        'labels': [label.name for label in issue.labels],
                        'comments': issue.comments
                    })
                return issues
            return []
        except Exception as e:
            logger.error(f"Failed to get issues: {e}")
            return []
    
    def create_issue(self, repo_name: str, title: str, body: str, 
                    labels: List[str] = None, assignees: List[str] = None) -> Optional[Dict[str, Any]]:
        """Create a new issue"""
        try:
            repository = self.get_repository(repo_name)
            if repository:
                issue = repository.create_issue(
                    title=title,
                    body=body,
                    labels=labels or [],
                    assignees=assignees or []
                )
                
                return {
                    'number': issue.number,
                    'title': issue.title,
                    'body': issue.body,
                    'state': issue.state,
                    'html_url': issue.html_url,
                    'created_at': issue.created_at,
                    'updated_at': issue.updated_at,
                    'user': issue.user.login if issue.user else None,
                    'assignee': issue.assignee.login if issue.assignee else None,
                    'labels': [label.name for label in issue.labels]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GitHub integration statistics"""
        try:
            repositories = self.list_repositories()
            
            total_repos = len(repositories)
            total_stars = sum(repo.get('stargazers_count', 0) for repo in repositories)
            total_forks = sum(repo.get('forks_count', 0) for repo in repositories)
            total_issues = sum(len(self.get_issues(repo['name'])) for repo in repositories)
            
            return {
                'total_repositories': total_repos,
                'total_stars': total_stars,
                'total_forks': total_forks,
                'total_issues': total_issues,
                'repositories': repositories
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def validate_token(self) -> bool:
        """Validate GitHub token"""
        try:
            user = self.github.get_user()
            logger.info(f"Token validated for user: {user.login}")
            return True
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return False
    
    def get_rate_limit(self) -> Dict[str, Any]:
        """Get GitHub API rate limit information"""
        try:
            rate_limit = self.github.get_rate_limit()
            return {
                'core': {
                    'limit': rate_limit.core.limit,
                    'remaining': rate_limit.core.remaining,
                    'reset': rate_limit.core.reset,
                    'used': rate_limit.core.used
                },
                'search': {
                    'limit': rate_limit.search.limit,
                    'remaining': rate_limit.search.remaining,
                    'reset': rate_limit.search.reset,
                    'used': rate_limit.search.used
                },
                'graphql': {
                    'limit': rate_limit.graphql.limit,
                    'remaining': rate_limit.graphql.remaining,
                    'reset': rate_limit.graphql.reset,
                    'used': rate_limit.graphql.used
                }
            }
        except Exception as e:
            logger.error(f"Failed to get rate limit: {e}")
            return {}
    
    def __str__(self):
        return f"GitHubIntegration(organization={self.organization}, user={self.github.get_user().login if self.github else 'Not authenticated'})"


# Example usage
if __name__ == "__main__":
    # Initialize GitHub integration
    github_integration = GitHubIntegration()
    
    # Test token validation
    if github_integration.validate_token():
        print("✅ GitHub token is valid")
        
        # Get rate limit
        rate_limit = github_integration.get_rate_limit()
        print(f"Rate limit: {rate_limit}")
        
        # List repositories
        repositories = github_integration.list_repositories()
        print(f"Repositories: {len(repositories)}")
        
        for repo in repositories[:5]:  # Show first 5 repositories
            print(f"  - {repo['name']}: {repo['description']}")
    else:
        print("❌ GitHub token is invalid")