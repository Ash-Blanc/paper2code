"""
GitHub Integration Tests

This module contains comprehensive tests for the GitHub integration functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path().cwd().parent / "app"
sys.path.insert(0, str(app_dir))

from app.integrations import GitHubIntegration, GitHubRepositoryResult
from app.models.paper import PaperMetadata, Author
from app.models.code import CodeImplementation, CodeFile, Language
from app.models.repository import RepositoryConfig, Visibility


class TestGitHubIntegration:
    """Test suite for GitHub integration"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock GitHub token
        self.test_token = "test_github_token"
        self.test_organization = "test-org"
        
        # Mock paper metadata
        self.paper_metadata = PaperMetadata(
            title="Attention Is All You Need",
            authors=[
                Author(name="Ashish Vaswani", email="ashish@example.com"),
                Author(name="Noam Shazeer", email="noam@example.com")
            ],
            journal="NeurIPS",
            publication_year=2017,
            doi="10.48550/arXiv.1706.03762",
            url="https://arxiv.org/abs/1706.03762",
            domain="Natural Language Processing",
            abstract="The dominant sequence transduction models are based on complex recurrent neural networks...",
            research_questions=[
                "How can we improve sequence transduction without recurrence?",
                "Can attention mechanisms alone achieve good performance?"
            ]
        )
        
        # Mock code implementation
        self.code_implementation = CodeImplementation(
            language_used="Python",
            framework_used="PyTorch",
            generated_files=[
                CodeFile(
                    name="transformer.py",
                    file_type="main",
                    language=Language.PYTHON,
                    content="import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):\n    pass",
                    purpose="Main transformer implementation",
                    dependencies=["torch", "numpy"],
                    key_functions=["__init__", "forward"]
                ),
                CodeFile(
                    name="test_transformer.py",
                    file_type="test",
                    language=Language.PYTHON,
                    content="import unittest\nfrom transformer import Transformer\n\nclass TestTransformer(unittest.TestCase):\n    pass",
                    purpose="Test suite for transformer implementation",
                    dependencies=["unittest", "torch"],
                    key_functions=["test_forward_pass", "test_backward_pass"]
                )
            ]
        )
        
        # Mock repository config
        self.repository_config = RepositoryConfig(
            description="Code implementation for Attention Is All You Need",
            visibility=Visibility.PUBLIC,
            has_issues=True,
            has_wiki=True,
            has_projects=True,
            gitignore_template="Python"
        )
    
    @patch('app.integrations.github.Github')
    def test_github_integration_initialization(self, mock_github):
        """Test GitHub integration initialization"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization=self.test_organization
        )
        
        # Verify initialization
        assert github_integration.token == self.test_token
        assert github_integration.organization == self.test_organization
        assert github_integration.github == mock_github_instance
        
        # Verify GitHub client was called with correct token
        mock_github.assert_called_once_with(self.test_token)
    
    @patch('app.integrations.github.Github')
    def test_github_integration_initialization_without_token(self, mock_github):
        """Test GitHub integration initialization without token"""
        # Remove token from environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token is required"):
                GitHubIntegration(token=None, organization=self.test_organization)
    
    @patch('app.integrations.github.Github')
    def test_generate_repo_name(self, mock_github):
        """Test repository name generation"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization=self.test_organization
        )
        
        # Test repository name generation
        repo_name = github_integration._generate_repo_name(self.paper_metadata)
        
        # Verify repository name
        assert "attention-is-all-you-need" in repo_name.lower()
        assert "2017" in repo_name
        assert len(repo_name) <= 100
    
    @patch('app.integrations.github.Github')
    def test_clean_string(self, mock_github):
        """Test string cleaning for repository names"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization=self.test_organization
        )
        
        # Test string cleaning
        test_cases = [
            ("Attention Is All You Need", "attention-is-all-you-need"),
            ("Transformer: Architecture & Implementation", "transformer-architecture-implementation"),
            ("Deep Learning for NLP!!!", "deep-learning-for-nlp"),
            ("  Multiple   Spaces   Here  ", "multiple-spaces-here"),
            ("Special@Characters#Here", "specialcharactershere")
        ]
        
        for input_str, expected_output in test_cases:
            result = github_integration._clean_string(input_str)
            assert result == expected_output
    
    @patch('app.integrations.github.Github')
    def test_create_github_repository_user_account(self, mock_github):
        """Test GitHub repository creation in user account"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_user.create_repo.return_value = Mock(
            name="test-repo",
            html_url="https://github.com/testuser/test-repo",
            clone_url="https://github.com/testuser/test-repo.git"
        )
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"  # Same as user login
        )
        
        # Create repository
        repository = github_integration._create_github_repository(
            "test-repo", 
            self.repository_config
        )
        
        # Verify repository creation
        assert repository.name == "test-repo"
        mock_user.create_repo.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_create_github_repository_organization(self, mock_github):
        """Test GitHub repository creation in organization"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_org = Mock()
        mock_org.create_repo.return_value = Mock(
            name="test-repo",
            html_url="https://github.com/test-org/test-repo",
            clone_url="https://github.com/test-org/test-repo.git"
        )
        mock_github_instance.get_user.return_value = mock_user
        mock_github_instance.get_organization.return_value = mock_org
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="test-org"
        )
        
        # Create repository
        repository = github_integration._create_github_repository(
            "test-repo", 
            self.repository_config
        )
        
        # Verify repository creation
        assert repository.name == "test-repo"
        mock_org.create_repo.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_create_repository_success(self, mock_github):
        """Test successful repository creation"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.name = "attention-is-all-you-need-2017"
        mock_repo.html_url = "https://github.com/testuser/attention-is-all-you-need-2017"
        mock_repo.clone_url = "https://github.com/testuser/attention-is-all-you-need-2017.git"
        mock_repo.create_file = Mock()
        mock_user.create_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Create repository
        result = github_integration.create_repository(
            self.paper_metadata,
            self.code_implementation,
            self.repository_config
        )
        
        # Verify result
        assert result.success is True
        assert result.repository_url == "https://github.com/testuser/attention-is-all-you-need-2017"
        assert result.repository_name == "attention-is-all-you-need-2017"
        assert result.clone_url == "https://github.com/testuser/attention-is-all-you-need-2017.git"
        assert len(result.errors) == 0
        
        # Verify repository creation was called
        mock_user.create_repo.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_create_repository_failure(self, mock_github):
        """Test failed repository creation"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_user.create_repo.side_effect = Exception("Repository creation failed")
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Create repository
        result = github_integration.create_repository(
            self.paper_metadata,
            self.code_implementation,
            self.repository_config
        )
        
        # Verify result
        assert result.success is False
        assert len(result.errors) == 1
        assert "Repository creation failed" in result.errors[0]
    
    @patch('app.integrations.github.Github')
    def test_get_repository(self, mock_github):
        """Test getting repository by name"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get repository
        repository = github_integration.get_repository("test-repo")
        
        # Verify repository
        assert repository.name == "test-repo"
        mock_user.get_repo.assert_called_once_with("test-repo")
    
    @patch('app.integrations.github.Github')
    def test_get_repository_not_found(self, mock_github):
        """Test getting non-existent repository"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_user.get_repo.side_effect = Exception("Repository not found")
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get repository
        repository = github_integration.get_repository("non-existent-repo")
        
        # Verify repository is None
        assert repository is None
    
    @patch('app.integrations.github.Github')
    def test_delete_repository(self, mock_github):
        """Test repository deletion"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.delete = Mock()
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Delete repository
        result = github_integration.delete_repository("test-repo")
        
        # Verify result
        assert result is True
        mock_repo.delete.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_delete_repository_not_found(self, mock_github):
        """Test deletion of non-existent repository"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_user.get_repo.side_effect = Exception("Repository not found")
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Delete repository
        result = github_integration.delete_repository("non-existent-repo")
        
        # Verify result
        assert result is False
    
    @patch('app.integrations.github.Github')
    def test_list_repositories(self, mock_github):
        """Test listing repositories"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo1 = Mock()
        mock_repo1.name = "repo1"
        mock_repo1.description = "Repository 1"
        mock_repo1.language = "Python"
        mock_repo1.size = 1024
        mock_repo1.stargazers_count = 5
        mock_repo1.forks_count = 2
        mock_repo1.open_issues_count = 1
        mock_repo1.created_at = "2023-01-01T00:00:00Z"
        mock_repo1.updated_at = "2023-01-02T00:00:00Z"
        
        mock_repo2 = Mock()
        mock_repo2.name = "repo2"
        mock_repo2.description = "Repository 2"
        mock_repo2.language = "JavaScript"
        mock_repo2.size = 2048
        mock_repo2.stargazers_count = 10
        mock_repo2.forks_count = 3
        mock_repo2.open_issues_count = 0
        mock_repo2.created_at = "2023-01-03T00:00:00Z"
        mock_repo2.updated_at = "2023-01-04T00:00:00Z"
        
        mock_user.get_repos.return_value = [mock_repo1, mock_repo2]
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # List repositories
        repositories = github_integration.list_repositories()
        
        # Verify repositories
        assert len(repositories) == 2
        assert repositories[0]['name'] == "repo1"
        assert repositories[1]['name'] == "repo2"
        mock_user.get_repos.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_validate_token(self, mock_github):
        """Test token validation"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Validate token
        result = github_integration.validate_token()
        
        # Verify result
        assert result is True
        mock_github_instance.get_user.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_validate_token_failure(self, mock_github):
        """Test token validation failure"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_github_instance.get_user.side_effect = Exception("Authentication failed")
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Validate token
        result = github_integration.validate_token()
        
        # Verify result
        assert result is False
    
    @patch('app.integrations.github.Github')
    def test_get_rate_limit(self, mock_github):
        """Test getting rate limit"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_rate_limit = Mock()
        mock_rate_limit.core = Mock()
        mock_rate_limit.core.limit = 5000
        mock_rate_limit.core.remaining = 4500
        mock_rate_limit.core.reset = 1234567890
        mock_rate_limit.core.used = 500
        
        mock_rate_limit.search = Mock()
        mock_rate_limit.search.limit = 30
        mock_rate_limit.search.remaining = 25
        mock_rate_limit.search.reset = 1234567890
        mock_rate_limit.search.used = 5
        
        mock_rate_limit.graphql = Mock()
        mock_rate_limit.graphql.limit = 5000
        mock_rate_limit.graphql.remaining = 4800
        mock_rate_limit.graphql.reset = 1234567890
        mock_rate_limit.graphql.used = 200
        
        mock_github_instance.get_rate_limit.return_value = mock_rate_limit
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get rate limit
        rate_limit = github_integration.get_rate_limit()
        
        # Verify rate limit
        assert rate_limit['core']['limit'] == 5000
        assert rate_limit['core']['remaining'] == 4500
        assert rate_limit['search']['limit'] == 30
        assert rate_limit['search']['remaining'] == 25
        assert rate_limit['graphql']['limit'] == 5000
        assert rate_limit['graphql']['remaining'] == 4800
    
    @patch('app.integrations.github.Github')
    def test_get_rate_limit_failure(self, mock_github):
        """Test getting rate limit failure"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_github_instance.get_rate_limit.side_effect = Exception("Rate limit fetch failed")
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get rate limit
        rate_limit = github_integration.get_rate_limit()
        
        # Verify rate limit is empty
        assert rate_limit == {}
    
    @patch('app.integrations.github.Github')
    def test_get_repository_info(self, mock_github):
        """Test getting repository information"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_repo.full_name = "testuser/test-repo"
        mock_repo.description = "Test repository"
        mock_repo.html_url = "https://github.com/testuser/test-repo"
        mock_repo.clone_url = "https://github.com/testuser/test-repo.git"
        mock_repo.ssh_url = "git@github.com:testuser/test-repo.git"
        mock_repo.language = "Python"
        mock_repo.size = 1024
        mock_repo.stargazers_count = 5
        mock_repo.watchers_count = 3
        mock_repo.forks_count = 2
        mock_repo.open_issues_count = 1
        mock_repo.has_issues = True
        mock_repo.has_projects = True
        mock_repo.has_wiki = True
        mock_repo.has_pages = False
        mock_repo.archived = False
        mock_repo.disabled = False
        mock_repo.pushed_at = "2023-01-01T00:00:00Z"
        mock_repo.created_at = "2023-01-01T00:00:00Z"
        mock_repo.updated_at = "2023-01-02T00:00:00Z"
        mock_repo.default_branch = "main"
        
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get repository info
        repo_info = github_integration.get_repository_info("test-repo")
        
        # Verify repository info
        assert repo_info['name'] == "test-repo"
        assert repo_info['full_name'] == "testuser/test-repo"
        assert repo_info['description'] == "Test repository"
        assert repo_info['language'] == "Python"
        assert repo_info['size'] == 1024
        assert repo_info['stargazers_count'] == 5
        assert repo_info['forks_count'] == 2
        assert repo_info['open_issues_count'] == 1
    
    @patch('app.integrations.github.Github')
    def test_get_repository_info_not_found(self, mock_github):
        """Test getting non-existent repository info"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_user.get_repo.side_effect = Exception("Repository not found")
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get repository info
        repo_info = github_integration.get_repository_info("non-existent-repo")
        
        # Verify repository info is None
        assert repo_info is None
    
    @patch('app.integrations.github.Github')
    def test_update_repository_description(self, mock_github):
        """Test updating repository description"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.edit = Mock()
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Update repository description
        result = github_integration.update_repository_description(
            "test-repo", 
            "Updated description"
        )
        
        # Verify result
        assert result is True
        mock_repo.edit.assert_called_once_with(description="Updated description")
    
    @patch('app.integrations.github.Github')
    def test_update_repository_description_not_found(self, mock_github):
        """Test updating non-existent repository description"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_user.get_repo.side_effect = Exception("Repository not found")
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Update repository description
        result = github_integration.update_repository_description(
            "non-existent-repo", 
            "Updated description"
        )
        
        # Verify result
        assert result is False
    
    @patch('app.integrations.github.Github')
    def test_add_collaborator(self, mock_github):
        """Test adding collaborator"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.add_to_collaborators = Mock()
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Add collaborator
        result = github_integration.add_collaborator(
            "test-repo", 
            "collaborator", 
            "write"
        )
        
        # Verify result
        assert result is True
        mock_repo.add_to_collaborators.assert_called_once_with("collaborator", "write")
    
    @patch('app.integrations.github.Github')
    def test_remove_collaborator(self, mock_github):
        """Test removing collaborator"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_repo.remove_from_collaborators = Mock()
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Remove collaborator
        result = github_integration.remove_collaborator(
            "test-repo", 
            "collaborator"
        )
        
        # Verify result
        assert result is True
        mock_repo.remove_from_collaborators.assert_called_once_with("collaborator")
    
    @patch('app.integrations.github.Github')
    def test_create_release(self, mock_github):
        """Test creating release"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_release = Mock()
        mock_release.id = 1
        mock_release.tag_name = "v1.0.0"
        mock_release.name = "Version 1.0.0"
        mock_release.body = "Release notes"
        mock_release.html_url = "https://github.com/testuser/test-repo/releases/tag/v1.0.0"
        mock_release.created_at = "2023-01-01T00:00:00Z"
        mock_release.published_at = "2023-01-01T00:00:00Z"
        mock_release.draft = False
        mock_release.prerelease = False
        mock_repo.create_git_release.return_value = mock_release
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Create release
        result = github_integration.create_release(
            "test-repo", 
            "v1.0.0", 
            "Version 1.0.0", 
            "Release notes"
        )
        
        # Verify result
        assert result['id'] == 1
        assert result['tag_name'] == "v1.0.0"
        assert result['name'] == "Version 1.0.0"
        assert result['body'] == "Release notes"
        mock_repo.create_git_release.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_get_releases(self, mock_github):
        """Test getting releases"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_release1 = Mock()
        mock_release1.id = 1
        mock_release1.tag_name = "v1.0.0"
        mock_release1.name = "Version 1.0.0"
        mock_release1.body = "Release notes 1"
        mock_release1.html_url = "https://github.com/testuser/test-repo/releases/tag/v1.0.0"
        mock_release1.created_at = "2023-01-01T00:00:00Z"
        mock_release1.published_at = "2023-01-01T00:00:00Z"
        mock_release1.draft = False
        mock_release1.prerelease = False
        mock_release1.author = Mock()
        mock_release1.author.login = "testuser"
        
        mock_release2 = Mock()
        mock_release2.id = 2
        mock_release2.tag_name = "v1.1.0"
        mock_release2.name = "Version 1.1.0"
        mock_release2.body = "Release notes 2"
        mock_release2.html_url = "https://github.com/testuser/test-repo/releases/tag/v1.1.0"
        mock_release2.created_at = "2023-02-01T00:00:00Z"
        mock_release2.published_at = "2023-02-01T00:00:00Z"
        mock_release2.draft = False
        mock_release2.prerelease = False
        mock_release2.author = Mock()
        mock_release2.author.login = "testuser"
        
        mock_repo.get_releases.return_value = [mock_release1, mock_release2]
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get releases
        releases = github_integration.get_releases("test-repo")
        
        # Verify releases
        assert len(releases) == 2
        assert releases[0]['id'] == 1
        assert releases[1]['id'] == 2
        mock_repo.get_releases.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_create_issue(self, mock_github):
        """Test creating issue"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_issue = Mock()
        mock_issue.number = 1
        mock_issue.title = "Test Issue"
        mock_issue.body = "This is a test issue"
        mock_issue.state = "open"
        mock_issue.html_url = "https://github.com/testuser/test-repo/issues/1"
        mock_issue.created_at = "2023-01-01T00:00:00Z"
        mock_issue.updated_at = "2023-01-01T00:00:00Z"
        mock_issue.closed_at = None
        mock_issue.user = Mock()
        mock_issue.user.login = "testuser"
        mock_issue.assignee = None
        mock_issue.labels = [Mock(name="bug")]
        mock_issue.comments = 0
        
        mock_repo.create_issue.return_value = mock_issue
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Create issue
        result = github_integration.create_issue(
            "test-repo", 
            "Test Issue", 
            "This is a test issue",
            labels=["bug"],
            assignees=["testuser"]
        )
        
        # Verify result
        assert result['number'] == 1
        assert result['title'] == "Test Issue"
        assert result['body'] == "This is a test issue"
        mock_repo.create_issue.assert_called_once()
    
    @patch('app.integrations.github.Github')
    def test_get_issues(self, mock_github):
        """Test getting issues"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_issue1 = Mock()
        mock_issue1.number = 1
        mock_issue1.title = "Bug 1"
        mock_issue1.body = "This is bug 1"
        mock_issue1.state = "open"
        mock_issue1.html_url = "https://github.com/testuser/test-repo/issues/1"
        mock_issue1.created_at = "2023-01-01T00:00:00Z"
        mock_issue1.updated_at = "2023-01-01T00:00:00Z"
        mock_issue1.closed_at = None
        mock_issue1.user = Mock()
        mock_issue1.user.login = "testuser"
        mock_issue1.assignee = None
        mock_issue1.labels = [Mock(name="bug")]
        mock_issue1.comments = 2
        
        mock_issue2 = Mock()
        mock_issue2.number = 2
        mock_issue2.title = "Feature 1"
        mock_issue2.body = "This is feature 1"
        mock_issue2.state = "closed"
        mock_issue2.html_url = "https://github.com/testuser/test-repo/issues/2"
        mock_issue2.created_at = "2023-01-02T00:00:00Z"
        mock_issue2.updated_at = "2023-01-02T00:00:00Z"
        mock_issue2.closed_at = "2023-01-03T00:00:00Z"
        mock_issue2.user = Mock()
        mock_issue2.user.login = "testuser"
        mock_issue2.assignee = Mock()
        mock_issue2.assignee.login = "assignee"
        mock_issue2.labels = [Mock(name="enhancement")]
        mock_issue2.comments = 5
        
        mock_repo.get_issues.return_value = [mock_issue1, mock_issue2]
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get issues
        issues = github_integration.get_issues("test-repo", state="all")
        
        # Verify issues
        assert len(issues) == 2
        assert issues[0]['number'] == 1
        assert issues[1]['number'] == 2
        mock_repo.get_issues.assert_called_once_with(state="all")
    
    @patch('app.integrations.github.Github')
    def test_close_issues(self, mock_github):
        """Test closing issues"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_repo = Mock()
        mock_issue1 = Mock()
        mock_issue1.edit = Mock()
        mock_issue1.number = 1
        
        mock_issue2 = Mock()
        mock_issue2.edit = Mock()
        mock_issue2.number = 2
        
        mock_repo.get_issues.return_value = [mock_issue1, mock_issue2]
        mock_repo.name = "test-repo"
        mock_user.get_repo.return_value = mock_repo
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Close issues
        result = github_integration.close_issues("test-repo", [1, 2])
        
        # Verify result
        assert result is True
        mock_issue1.edit.assert_called_once_with(state='closed')
        mock_issue2.edit.assert_called_once_with(state='closed')
    
    @patch('app.integrations.github.Github')
    def test_get_statistics(self, mock_github):
        """Test getting statistics"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        
        # Mock repositories
        mock_repo1 = Mock()
        mock_repo1.name = "repo1"
        mock_repo1.get.return_value = Mock(stargazers_count=5, forks_count=2)
        mock_repo1.get_issues.return_value = []
        
        mock_repo2 = Mock()
        mock_repo2.name = "repo2"
        mock_repo2.get.return_value = Mock(stargazers_count=10, forks_count=3)
        mock_repo2.get_issues.return_value = [Mock()]
        
        mock_user.get_repos.return_value = [mock_repo1, mock_repo2]
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="testuser"
        )
        
        # Get statistics
        stats = github_integration.get_statistics()
        
        # Verify statistics
        assert stats['total_repositories'] == 2
        assert stats['total_stars'] == 15
        assert stats['total_forks'] == 5
        assert stats['total_issues'] == 1
        assert len(stats['repositories']) == 2
    
    @patch('app.integrations.github.Github')
    def test_str_representation(self, mock_github):
        """Test string representation"""
        # Mock GitHub client
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        # Initialize GitHub integration
        github_integration = GitHubIntegration(
            token=self.test_token,
            organization="test-org"
        )
        
        # Test string representation
        str_repr = str(github_integration)
        assert "GitHubIntegration" in str_repr
        assert "organization=test-org" in str_repr
        assert "user=testuser" in str_repr


class TestGitHubRepositoryResult:
    """Test suite for GitHubRepositoryResult"""
    
    def test_github_repository_result_success(self):
        """Test GitHubRepositoryResult for successful operation"""
        result = GitHubRepositoryResult(
            success=True,
            repository_url="https://github.com/user/repo",
            repository_name="repo",
            clone_url="https://github.com/user/repo.git",
            errors=[],
            warnings=[]
        )
        
        assert result.success is True
        assert result.repository_url == "https://github.com/user/repo"
        assert result.repository_name == "repo"
        assert result.clone_url == "https://github.com/user/repo.git"
        assert result.errors == []
        assert result.warnings == []
    
    def test_github_repository_result_failure(self):
        """Test GitHubRepositoryResult for failed operation"""
        result = GitHubRepositoryResult(
            success=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert result.success is False
        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == ["Warning 1"]
    
    def test_github_repository_result_defaults(self):
        """Test GitHubRepositoryResult with default values"""
        result = GitHubRepositoryResult(success=True)
        
        assert result.success is True
        assert result.repository_url is None
        assert result.repository_name is None
        assert result.clone_url is None
        assert result.errors == []
        assert result.warnings == []


if __name__ == "__main__":
    pytest.main([__file__])