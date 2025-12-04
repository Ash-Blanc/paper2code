"""
Advanced Configuration Examples

This file demonstrates advanced configuration patterns for the Paper2Code agent system.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Add the app directory to the Python path
app_dir = Path().cwd().parent / "app"
sys.path.insert(0, str(app_dir))

from app.main import Paper2CodeAgent
from app.models.paper import PaperMetadata, Author
from app.models.code import CodeImplementation, CodeFile, Language
from app.models.repository import RepositoryConfig, Visibility
from app.cache import CacheManager, PredictiveCacheWarmingService
from app.integrations import GitHubIntegration
from app.monitoring import SystemMonitor, UsageAnalytics


class AdvancedPaper2CodeAgent(Paper2CodeAgent):
    """Extended Paper2Code agent with advanced features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with advanced configuration"""
        self.config = config or {}
        self.cache_manager = CacheManager(
            redis_host=self.config.get('redis_host', 'localhost'),
            redis_port=self.config.get('redis_port', 6379),
            redis_db=self.config.get('redis_db', 0),
            redis_password=self.config.get('redis_password'),
            default_ttl=self.config.get('cache_ttl', 2592000),  # 30 days
            warming_enabled=self.config.get('cache_warming_enabled', True),
            warming_interval=self.config.get('cache_warming_interval', 3600)  # 1 hour
        )
        
        self.monitor = SystemMonitor()
        self.analytics = UsageAnalytics()
        
        # Initialize parent with custom configuration
        super().__init__(
            cache_manager=self.cache_manager,
            max_retries=self.config.get('max_retries', 3),
            timeout=self.config.get('timeout', 300),
            parallel_processing=self.config.get('parallel_processing', True)
        )
    
    def process_paper_with_monitoring(self, paper_input: str, input_type: str, 
                                    repository_config: RepositoryConfig = None) -> Dict[str, Any]:
        """Process paper with comprehensive monitoring"""
        start_time = datetime.now()
        
        try:
            # Track request
            self.analytics.track_request(True, 0, paper_input)
            
            # Process paper
            result = super().process_paper(paper_input, input_type, repository_config)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update analytics
            self.analytics.track_request(True, processing_time, paper_input)
            
            # Add monitoring data
            result['processing_time'] = processing_time
            result['system_metrics'] = self.monitor.get_system_metrics()
            result['cache_metrics'] = self.cache_manager.get_cache_metrics()
            
            return result
            
        except Exception as e:
            # Track failed request
            processing_time = (datetime.now() - start_time).total_seconds()
            self.analytics.track_request(False, processing_time, paper_input)
            
            # Re-raise with additional context
            raise Exception(f"Processing failed after {processing_time:.2f}s: {str(e)}")


def example_custom_agent_configuration():
    """Example: Custom agent configuration"""
    print("=== Custom Agent Configuration Example ===")
    
    # Advanced configuration
    config = {
        'max_retries': 5,
        'timeout': 600,  # 10 minutes
        'parallel_processing': True,
        'cache_ttl': 604800,  # 7 days
        'cache_warming_enabled': True,
        'cache_warming_interval': 1800,  # 30 minutes
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_db': 0,
        'redis_password': None,
        'output_level': 'production-ready',
        'language_preference': 'python',
        'framework_preference': 'pytorch',
        'github_enabled': True,
        'github_organization': 'paper2code-repos',
        'github_token': os.getenv('GITHUB_TOKEN'),
        'log_level': 'INFO',
        'log_file': 'paper2code_advanced.log'
    }
    
    # Initialize custom agent
    agent = AdvancedPaper2CodeAgent(config)
    
    print(f"✅ Custom agent initialized with configuration:")
    print(f"  - Max retries: {config['max_retries']}")
    print(f"  - Timeout: {config['timeout']}s")
    print(f"  - Parallel processing: {config['parallel_processing']}")
    print(f"  - Cache TTL: {config['cache_ttl']}s")
    print(f"  - Cache warming: {config['cache_warming_enabled']}")
    print(f"  - Output level: {config['output_level']}")
    print(f"  - Language preference: {config['language_preference']}")
    print(f"  - Framework preference: {config['framework_preference']}")
    
    return agent


def example_cache_configuration():
    """Example: Advanced cache configuration"""
    print("\n=== Advanced Cache Configuration Example ===")
    
    # Initialize cache manager with advanced configuration
    cache_manager = CacheManager(
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        default_ttl=2592000,  # 30 days
        max_memory_usage='2GB',
        eviction_policy='allkeys-lru',
        warming_enabled=True,
        warming_interval=3600,  # 1 hour
        warming_batch_size=100,
        analytics_enabled=True,
        analytics_retention_days=30
    )
    
    # Test cache operations
    print("Testing cache operations...")
    
    # Set cache value
    cache_manager.set('test_key', 'test_value', ttl=3600)
    print(f"✅ Cache set: test_key = {cache_manager.get('test_key')}")
    
    # Check cache existence
    print(f"✅ Cache exists: test_key = {cache_manager.exists('test_key')}")
    
    # Get cache metrics
    metrics = cache_manager.get_cache_metrics()
    print(f"✅ Cache metrics: {metrics}")
    
    # Get cache statistics
    stats = cache_manager.get_cache_stats()
    print(f"✅ Cache statistics: {stats}")
    
    # Clear cache
    cache_manager.clear()
    print(f"✅ Cache cleared")
    
    return cache_manager


def example_predictive_cache_warming():
    """Example: Predictive cache warming"""
    print("\n=== Predictive Cache Warming Example ===")
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Initialize predictive cache warming service
    warming_service = PredictiveCacheWarmingService(cache_manager)
    
    # Start predictive warming in background
    print("Starting predictive cache warming service...")
    warming_service.start_predictive_warming()
    
    # Generate warming plan
    print("Generating predictive warming plan...")
    warming_plan = warming_service.generate_predictive_warming_plan()
    
    print(f"✅ Warming plan generated:")
    print(f"  - Conference-based tasks: {len(warming_plan.get('conference_based', []))}")
    print(f"  - Journal-based tasks: {len(warming_plan.get('journal_based', []))}")
    print(f"  - Hotspot-based tasks: {len(warming_plan.get('hotspot_based', []))}")
    print(f"  - Trend-based tasks: {len(warming_plan.get('trend_based', []))}")
    print(f"  - ML-optimized tasks: {len(warming_plan.get('ml_optimized', []))}")
    
    # Execute warming tasks
    print("Executing predictive warming tasks...")
    warming_service.execute_predictive_warming()
    
    return warming_service


def example_github_advanced_configuration():
    """Example: Advanced GitHub configuration"""
    print("\n=== Advanced GitHub Configuration Example ===")
    
    # Initialize GitHub integration with advanced configuration
    github_integration = GitHubIntegration(
        token=os.getenv('GITHUB_TOKEN', 'your_token_here'),
        organization='paper2code-repos'
    )
    
    # Test token validation
    if github_integration.validate_token():
        print("✅ GitHub token is valid")
        
        # Get rate limit
        rate_limit = github_integration.get_rate_limit()
        print(f"✅ Rate limit: {rate_limit}")
        
        # Get repository statistics
        stats = github_integration.get_statistics()
        print(f"✅ Repository statistics:")
        print(f"  - Total repositories: {stats.get('total_repositories', 0)}")
        print(f"  - Total stars: {stats.get('total_stars', 0)}")
        print(f"  - Total forks: {stats.get('total_forks', 0)}")
        print(f"  - Total issues: {stats.get('total_issues', 0)}")
        
        # List repositories with filtering
        repositories = github_integration.list_repositories()
        print(f"✅ Total repositories: {len(repositories)}")
        
        # Filter repositories by language
        python_repos = [r for r in repositories if r.get('language') == 'Python']
        print(f"✅ Python repositories: {len(python_repos)}")
        
        # Show repository details
        if repositories:
            repo = repositories[0]
            repo_info = github_integration.get_repository_info(repo['name'])
            if repo_info:
                print(f"✅ Repository details for {repo['name']}:")
                print(f"  - Full name: {repo_info.get('full_name')}")
                print(f"  - Description: {repo_info.get('description')}")
                print(f"  - Language: {repo_info.get('language')}")
                print(f"  - Size: {repo_info.get('size')} KB")
                print(f"  - Stars: {repo_info.get('stargazers_count')}")
                print(f"  - Forks: {repo_info.get('forks_count')}")
                print(f"  - Issues: {repo_info.get('open_issues_count')}")
        
        # Create advanced repository
        print("\nCreating advanced repository...")
        
        # Create advanced paper metadata
        paper_metadata = PaperMetadata(
            title="Advanced Transformer Architecture",
            authors=[
                Author(name="Advanced Author", email="advanced@example.com"),
                Author(name="Research Team", email="research@example.com")
            ],
            journal="Advanced Journal of AI",
            publication_year=2024,
            doi="10.48550/arXiv.2401.00000",
            url="https://arxiv.org/abs/2401.00000",
            domain="Advanced Natural Language Processing",
            abstract="This paper presents advanced transformer architectures with improved performance and efficiency.",
            research_questions=[
                "How can we improve transformer efficiency?",
                "What are the latest advances in attention mechanisms?"
            ]
        )
        
        # Create advanced code implementation
        code_implementation = CodeImplementation(
            language_used="Python",
            framework_used="PyTorch",
            generated_files=[
                CodeFile(
                    name="advanced_transformer.py",
                    file_type="main",
                    language=Language.PYTHON,
                    content="""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(AdvancedTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Advanced transformer layers
        self.layers = nn.ModuleList([
            AdvancedTransformerLayer(d_model, nhead) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class AdvancedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(AdvancedTransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear transformations
        Q = self.w_q(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attn_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
""",
                    purpose="Advanced transformer implementation with multi-head attention",
                    dependencies=["torch", "numpy"],
                    key_functions=["__init__", "forward", "scaled_dot_product_attention"]
                ),
                CodeFile(
                    name="requirements.txt",
                    file_type="config",
                    language=Language.PLAIN_TEXT,
                    content="""
torch>=1.9.0
numpy>=1.21.0
transformers>=4.15.0
datasets>=1.18.0
accelerate>=0.5.0
""",
                    purpose="Python dependencies",
                    dependencies=[],
                    key_functions=[]
                ),
                CodeFile(
                    name="README.md",
                    file_type="documentation",
                    language=Language.MARKDOWN,
                    content="""
# Advanced Transformer Implementation

This repository contains an advanced implementation of the Transformer architecture based on the latest research.

## Features

- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Efficient memory usage

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from advanced_transformer import AdvancedTransformer

# Initialize model
model = AdvancedTransformer(d_model=512, nhead=8, num_layers=6)

# Forward pass
output = model(input_sequence, mask=None)
```

## License

MIT License
""",
                    purpose="Repository documentation",
                    dependencies=[],
                    key_functions=[]
                )
            ]
        )
        
        # Create advanced repository configuration
        repo_config = RepositoryConfig(
            description="Advanced transformer implementation with state-of-the-art features",
            visibility=Visibility.PUBLIC,
            has_issues=True,
            has_wiki=True,
            has_projects=True,
            gitignore_template="Python"
        )
        
        # Create repository
        result = github_integration.create_repository(
            paper_metadata,
            code_implementation,
            repo_config
        )
        
        if result.success:
            print(f"✅ Advanced repository created: {result.repository_url}")
            print(f"Repository name: {result.repository_name}")
            print(f"Clone URL: {result.clone_url}")
            
            # Get repository information
            repo_info = github_integration.get_repository_info(result.repository_name)
            if repo_info:
                print(f"✅ Repository information:")
                print(f"  - Full name: {repo_info.get('full_name')}")
                print(f"  - Language: {repo_info.get('language')}")
                print(f"  - Size: {repo_info.get('size')} KB")
                print(f"  - Stars: {repo_info.get('stargazers_count')}")
                print(f"  - Forks: {repo_info.get('forks_count')}")
                print(f"  - Issues: {repo_info.get('open_issues_count')}")
        else:
            print(f"❌ Repository creation failed: {result.errors}")
        
        return result
        
    else:
        print("❌ GitHub token is invalid")
        return None


def example_monitoring_and_analytics():
    """Example: Monitoring and analytics"""
    print("\n=== Monitoring and Analytics Example ===")
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    # Get system metrics
    metrics = monitor.get_system_metrics()
    print(f"✅ System metrics:")
    print(f"  - CPU usage: {metrics.get('cpu_usage')}%")
    print(f"  - Memory usage: {metrics.get('memory_usage')}%")
    print(f"  - Disk usage: {metrics.get('disk_usage')}%")
    print(f"  - Active connections: {metrics.get('active_connections')}")
    
    # Initialize usage analytics
    analytics = UsageAnalytics()
    
    # Track some sample requests
    analytics.track_request(True, 120.5, "https://arxiv.org/abs/1706.03762")
    analytics.track_request(True, 95.3, "https://arxiv.org/abs/1810.04805")
    analytics.track_request(False, 45.2, "invalid_input")
    analytics.track_request(True, 200.8, "10.48550/arXiv.1805.04833")
    
    # Get analytics metrics
    analytics_metrics = analytics.get_metrics()
    print(f"✅ Analytics metrics:")
    print(f"  - Total requests: {analytics_metrics.get('total_requests')}")
    print(f"  - Successful requests: {analytics_metrics.get('successful_requests')}")
    print(f"  - Failed requests: {analytics_metrics.get('failed_requests')}")
    print(f"  - Success rate: {analytics_metrics.get('success_rate')}%")
    print(f"  - Average processing time: {analytics_metrics.get('average_processing_time'):.2f}s")
    
    # Get popular papers
    popular_papers = analytics.get_popular_papers()
    print(f"✅ Popular papers:")
    for paper, count in popular_papers.items():
        print(f"  - {paper}: {count} requests")
    
    return analytics


def example_error_handling_and_recovery():
    """Example: Advanced error handling and recovery"""
    print("\n=== Advanced Error Handling Example ===")
    
    # Initialize agent with error handling
    agent = AdvancedPaper2CodeAgent({
        'max_retries': 5,
        'timeout': 600,
        'parallel_processing': True
    })
    
    # Test various error scenarios
    error_scenarios = [
        {
            'name': 'Invalid input type',
            'input': 'https://arxiv.org/abs/1706.03762',
            'input_type': 'invalid_type',
            'expected_error': True
        },
        {
            'name': 'Invalid paper URL',
            'input': 'https://invalid-arxiv.org/abs/1234.5678',
            'input_type': 'arxiv_url',
            'expected_error': True
        },
        {
            'name': 'Invalid repository config',
            'input': 'https://arxiv.org/abs/1706.03762',
            'input_type': 'arxiv_url',
            'repository_config': None,
            'expected_error': True
        },
        {
            'name': 'Valid input',
            'input': 'https://arxiv.org/abs/1706.03762',
            'input_type': 'arxiv_url',
            'repository_config': RepositoryConfig(),
            'expected_error': False
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        try:
            result = agent.process_paper_with_monitoring(
                scenario['input'],
                scenario['input_type'],
                scenario.get('repository_config')
            )
            
            if scenario['expected_error']:
                print(f"❌ Expected error but got success: {result}")
            else:
                print(f"✅ Success: {result.get('repository_url', 'No URL')}")
                
        except Exception as e:
            if scenario['expected_error']:
                print(f"✅ Expected error: {str(e)}")
            else:
                print(f"❌ Unexpected error: {str(e)}")
    
    return agent


def example_batch_processing_with_monitoring():
    """Example: Batch processing with monitoring"""
    print("\n=== Batch Processing with Monitoring Example ===")
    
    # Initialize agent
    agent = AdvancedPaper2CodeAgent({
        'parallel_processing': True,
        'max_retries': 3,
        'timeout': 300
    })
    
    # List of papers to process
    papers = [
        {
            'input': 'https://arxiv.org/abs/1706.03762',
            'input_type': 'arxiv_url',
            'title': 'Attention Is All You Need',
            'description': 'Transformer architecture implementation'
        },
        {
            'input': 'https://arxiv.org/abs/1810.04805',
            'input_type': 'arxiv_url',
            'title': 'BERT',
            'description': 'BERT implementation with pre-training'
        },
        {
            'input': '10.48550/arXiv.1805.04833',
            'input_type': 'doi',
            'title': 'GPT-2',
            'description': 'GPT-2 implementation with training'
        },
        {
            'input': 'https://arxiv.org/abs/2005.14165',
            'input_type': 'arxiv_url',
            'title': 'GPT-3',
            'description': 'GPT-3 implementation with few-shot learning'
        },
        {
            'input': 'https://arxiv.org/abs/2103.03206',
            'input_type': 'arxiv_url',
            'title': 'ViT',
            'description': 'Vision Transformer implementation'
        }
    ]
    
    results = []
    start_time = datetime.now()
    
    print(f"Processing {len(papers)} papers...")
    
    for i, paper in enumerate(papers, 1):
        print(f"\nProcessing paper {i}/{len(papers)}: {paper['title']}")
        
        try:
            # Create repository configuration
            repo_config = RepositoryConfig(
                description=paper['description'],
                visibility=Visibility.PUBLIC,
                has_issues=True,
                has_wiki=True,
                has_projects=True,
                gitignore_template="Python"
            )
            
            # Process the paper
            result = agent.process_paper_with_monitoring(
                paper['input'],
                paper['input_type'],
                repo_config
            )
            
            results.append(result)
            
            # Display results
            if result.success:
                print(f"✅ {paper['title']}: {result.get('repository_url', 'No URL')}")
                print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            else:
                print(f"❌ {paper['title']}: {result.get('errors', ['Unknown error'])}")
                
        except Exception as e:
            print(f"❌ {paper['title']}: {str(e)}")
            results.append({'success': False, 'errors': [str(e)]})
    
    # Calculate summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total papers: {len(papers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(papers)*100):.1f}%")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per paper: {total_time/len(papers):.2f}s")
    
    # Get system metrics
    if results:
        latest_metrics = results[-1].get('system_metrics', {})
        print(f"Final system metrics:")
        print(f"  - CPU usage: {latest_metrics.get('cpu_usage')}%")
        print(f"  - Memory usage: {latest_metrics.get('memory_usage')}%")
    
    return results


def main():
    """Run all advanced examples"""
    print("Paper2Code Agent - Advanced Configuration Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_custom_agent_configuration()
        example_cache_configuration()
        example_predictive_cache_warming()
        example_github_advanced_configuration()
        example_monitoring_and_analytics()
        example_error_handling_and_recovery()
        example_batch_processing_with_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ All advanced examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running advanced examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()