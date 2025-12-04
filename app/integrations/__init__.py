"""
Integrations Package

This package provides integrations with external services and platforms
such as GitHub, arXiv, and other research repositories.
"""

from .github import GitHubIntegration, GitHubRepositoryResult

__all__ = [
    "GitHubIntegration",
    "GitHubRepositoryResult"
]