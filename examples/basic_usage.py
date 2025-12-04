"""
Basic Usage Examples

This file demonstrates basic usage patterns for the Paper2Code agent system.
"""

import os
import sys
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path().cwd().parent / "app"
sys.path.insert(0, str(app_dir))

from app.main import Paper2CodeAgent
from app.models.paper import PaperMetadata, Author
from app.models.code import CodeImplementation, CodeFile, Language
from app.models.repository import RepositoryConfig, Visibility
from app.integrations import GitHubIntegration, GitHubRepositoryResult


def example_basic_paper_processing():
    """Example: Basic paper processing with GitHub repository creation"""
    print("=== Basic Paper Processing Example ===")
    
    # Initialize the agent
    agent = Paper2CodeAgent()
    
    # Create paper metadata
    paper_metadata = PaperMetadata(
        title="Attention Is All You Need",
        authors=[
            Author(name="Ashish Vaswani", email="ashish@example.com"),
            Author(name="Noam Shazeer", email="noam@example.com"),
            Author(name="Niki Parmar", email="niki@example.com"),
            Author(name="Jakob Uszkoreit", email="jakob@example.com"),
            Author(name="Llion Jones", email="llion@example.com"),
            Author(name="Aidan N. Gomez", email="aidan@example.com"),
            Author(name="Łukasz Kaiser", email="lukasz@example.com"),
            Author(name="Illia Polosukhin", email="illia@example.com")
        ],
        journal="Advances in Neural Information Processing Systems",
        publication_year=2017,
        doi="10.48550/arXiv.1706.03762",
        url="https://arxiv.org/abs/1706.03762",
        domain="Natural Language Processing",
        abstract="The dominant sequence transduction models are based on complex recurrent neural networks that map an input sequence to an output sequence. The performance of these models is limited by the sequential nature of the recurrence and the difficulty of capturing long-range dependencies. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
        research_questions=[
            "How can we improve sequence transduction without recurrence?",
            "Can attention mechanisms alone achieve good performance?",
            "How does the Transformer compare to RNN-based models?"
        ]
    )
    
    # Create repository configuration
    repo_config = RepositoryConfig(
        description="Code implementation of the Transformer architecture from 'Attention Is All You Need'",
        visibility=Visibility.PUBLIC,
        has_issues=True,
        has_wiki=True,
        has_projects=True,
        gitignore_template="Python"
    )
    
    # Process the paper
    print("Processing paper...")
    result = agent.process_paper(
        paper_input="https://arxiv.org/abs/1706.03762",
        input_type="arxiv_url",
        repository_config=repo_config
    )
    
    # Display results
    print(f"✅ Processing completed!")
    print(f"Repository URL: {result.repository_url}")
    print(f"Repository Name: {result.repository_name}")
    print(f"Clone URL: {result.clone_url}")
    
    if result.errors:
        print(f"⚠️  Errors: {result.errors}")
    if result.warnings:
        print(f"⚠️  Warnings: {result.warnings}")
    
    return result


def example_pdf_paper_processing():
    """Example: Processing a PDF file"""
    print("\n=== PDF Paper Processing Example ===")
    
    # Initialize the agent
    agent = Paper2CodeAgent()
    
    # Create paper metadata (simulating PDF processing)
    paper_metadata = PaperMetadata(
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=[
            Author(name="Jacob Devlin", email="jacob@example.com"),
            Author(name="Ming-Wei Chang", email="mingwei@example.com"),
            Author(name="Kenton Lee", email="kenton@example.com"),
            Author(name="Kristina Toutanova", email="kristina@example.com")
        ],
        journal="Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
        publication_year=2019,
        doi="10.18653/v1/N19-1423",
        url="https://arxiv.org/abs/1810.04805",
        domain="Natural Language Processing",
        abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point improvement), MultiNLI accuracy to 86.7% (4.6% improvement) and SQuAD v1.1 question answering F1 to 93.2 (1.5 improvement)."
    )
    
    # Create repository configuration
    repo_config = RepositoryConfig(
        description="BERT implementation with pre-training and fine-tuning capabilities",
        visibility=Visibility.PRIVATE,  # Private repository
        has_issues=True,
        has_wiki=False,
        has_projects=False,
        gitignore_template="Python"
    )
    
    # Process the paper (simulating PDF input)
    print("Processing PDF paper...")
    result = agent.process_paper(
        paper_input="path/to/paper.pdf",  # This would be a real PDF file path
        input_type="pdf",
        repository_config=repo_config
    )
    
    # Display results
    print(f"✅ Processing completed!")
    print(f"Repository URL: {result.repository_url}")
    print(f"Repository Name: {result.repository_name}")
    print(f"Clone URL: {result.clone_url}")
    
    if result.errors:
        print(f"⚠️  Errors: {result.errors}")
    if result.warnings:
        print(f"⚠️  Warnings: {result.warnings}")
    
    return result


def example_doi_paper_processing():
    """Example: Processing a paper by DOI"""
    print("\n=== DOI Paper Processing Example ===")
    
    # Initialize the agent
    agent = Paper2CodeAgent()
    
    # Create paper metadata (simulating DOI processing)
    paper_metadata = PaperMetadata(
        title="GPT-2: Language Models are Unsupervised Multitask Learners",
        authors=[
            Author(name="Alec Radford", email="alec@example.com"),
            Author(name="Jesse Wu", email="jesse@example.com"),
            Author(name="Rewon Child", email="rewon@example.com"),
            Author(name="David Luan", email="david@example.com"),
            Author(name="Dario Amodei", email="dario@example.com"),
            Author(name="Ilya Sutskever", email="ilya@example.com")
        ],
        journal="arXiv preprint arXiv:1805.04833",
        publication_year=2019,
        doi="10.48550/arXiv.1805.04833",
        url="https://arxiv.org/abs/1805.04833",
        domain="Natural Language Processing",
        abstract="We show that the behaviors of large language models (LMs) on a range of tasks can be understood by considering them as performing unsupervised multitask learning. We train a 1.5B parameter LM on a new dataset of 8.3 million webpages, 40GB of text, and show that it achieves state-of-the-art performance on a variety of tasks. We also show that the LM can be used to generate synthetic tasks for training other models, and that it can be used to generate synthetic data for training other models. We also show that the LM can be used to generate synthetic tasks for training other models, and that it can be used to generate synthetic data for training other models."
    )
    
    # Create repository configuration
    repo_config = RepositoryConfig(
        description="GPT-2 implementation with training and inference capabilities",
        visibility=Visibility.PUBLIC,
        has_issues=True,
        has_wiki=True,
        has_projects=True,
        gitignore_template="Python"
    )
    
    # Process the paper (simulating DOI input)
    print("Processing DOI paper...")
    result = agent.process_paper(
        paper_input="10.48550/arXiv.1805.04833",
        input_type="doi",
        repository_config=repo_config
    )
    
    # Display results
    print(f"✅ Processing completed!")
    print(f"Repository URL: {result.repository_url}")
    print(f"Repository Name: {result.repository_name}")
    print(f"Clone URL: {result.clone_url}")
    
    if result.errors:
        print(f"⚠️  Errors: {result.errors}")
    if result.warnings:
        print(f"⚠️  Warnings: {result.warnings}")
    
    return result


def example_batch_processing():
    """Example: Processing multiple papers in batch"""
    print("\n=== Batch Processing Example ===")
    
    # Initialize the agent
    agent = Paper2CodeAgent()
    
    # List of papers to process
    papers = [
        {
            "input": "https://arxiv.org/abs/1706.03762",
            "input_type": "arxiv_url",
            "title": "Attention Is All You Need",
            "description": "Transformer architecture implementation"
        },
        {
            "input": "https://arxiv.org/abs/1810.04805",
            "input_type": "arxiv_url",
            "title": "BERT",
            "description": "BERT implementation with pre-training"
        },
        {
            "input": "10.48550/arXiv.1805.04833",
            "input_type": "doi",
            "title": "GPT-2",
            "description": "GPT-2 implementation with training"
        }
    ]
    
    results = []
    
    for i, paper in enumerate(papers, 1):
        print(f"\nProcessing paper {i}/{len(papers)}: {paper['title']}")
        
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
        result = agent.process_paper(
            paper_input=paper['input'],
            input_type=paper['input_type'],
            repository_config=repo_config
        )
        
        results.append(result)
        
        # Display results
        if result.success:
            print(f"✅ {paper['title']}: {result.repository_url}")
        else:
            print(f"❌ {paper['title']}: {result.errors}")
    
    # Summary
    successful = sum(1 for r in results if r.success)
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total papers: {len(papers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(papers) - successful}")
    
    return results


def example_github_integration():
    """Example: Direct GitHub integration usage"""
    print("\n=== GitHub Integration Example ===")
    
    # Initialize GitHub integration
    github_integration = GitHubIntegration(
        token=os.getenv("GITHUB_TOKEN", "your_github_token_here"),
        organization="paper2code-repos"
    )
    
    # Test token validation
    if github_integration.validate_token():
        print("✅ GitHub token is valid")
        
        # Get rate limit
        rate_limit = github_integration.get_rate_limit()
        print(f"Rate limit: {rate_limit}")
        
        # List repositories
        repositories = github_integration.list_repositories()
        print(f"Total repositories: {len(repositories)}")
        
        # Show first 3 repositories
        for repo in repositories[:3]:
            print(f"  - {repo['name']}: {repo['description']}")
        
        # Create a test repository
        print("\nCreating test repository...")
        
        # Create test paper metadata
        paper_metadata = PaperMetadata(
            title="Test Paper",
            authors=[Author(name="Test Author", email="test@example.com")],
            journal="Test Journal",
            publication_year=2023,
            doi="10.48550/arXiv.2301.00000",
            url="https://arxiv.org/abs/2301.00000",
            domain="Test Domain",
            abstract="This is a test paper for demonstration purposes."
        )
        
        # Create test code implementation
        code_implementation = CodeImplementation(
            language_used="Python",
            framework_used="PyTorch",
            generated_files=[
                CodeFile(
                    name="test.py",
                    file_type="main",
                    language=Language.PYTHON,
                    content="# Test implementation\nprint('Hello, World!')",
                    purpose="Test implementation",
                    dependencies=["torch"],
                    key_functions=["main"]
                )
            ]
        )
        
        # Create repository configuration
        repo_config = RepositoryConfig(
            description="Test repository for demonstration",
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
            print(f"✅ Repository created: {result.repository_url}")
            print(f"Repository name: {result.repository_name}")
            print(f"Clone URL: {result.clone_url}")
        else:
            print(f"❌ Repository creation failed: {result.errors}")
        
        # Get repository information
        if result.success:
            repo_info = github_integration.get_repository_info(result.repository_name)
            if repo_info:
                print(f"\nRepository Information:")
                print(f"  - Name: {repo_info['name']}")
                print(f"  - Language: {repo_info['language']}")
                print(f"  - Size: {repo_info['size']} KB")
                print(f"  - Stars: {repo_info['stargazers_count']}")
                print(f"  - Forks: {repo_info['forks_count']}")
                print(f"  - Issues: {repo_info['open_issues_count']}")
        
        # Clean up (optional)
        # if result.success:
        #     github_integration.delete_repository(result.repository_name)
        #     print("🗑️  Test repository deleted")
        
    else:
        print("❌ GitHub token is invalid")
        print("Please set GITHUB_TOKEN environment variable")


def example_error_handling():
    """Example: Error handling and recovery"""
    print("\n=== Error Handling Example ===")
    
    # Initialize the agent
    agent = Paper2CodeAgent()
    
    # Test with invalid input
    print("Testing with invalid input...")
    result = agent.process_paper(
        paper_input="invalid_input",
        input_type="arxiv_url",
        repository_config=RepositoryConfig()
    )
    
    if not result.success:
        print(f"❌ Expected error occurred: {result.errors}")
    
    # Test with invalid repository config
    print("\nTesting with invalid repository config...")
    result = agent.process_paper(
        paper_input="https://arxiv.org/abs/1706.03762",
        input_type="arxiv_url",
        repository_config=None  # Invalid config
    )
    
    if not result.success:
        print(f"❌ Expected error occurred: {result.errors}")
    
    # Test with network issues (simulated)
    print("\nTesting with network issues...")
    result = agent.process_paper(
        paper_input="https://nonexistent-arxiv.org/abs/1234.5678",
        input_type="arxiv_url",
        repository_config=RepositoryConfig()
    )
    
    if not result.success:
        print(f"❌ Expected error occurred: {result.errors}")
    
    print("✅ Error handling examples completed")


def main():
    """Run all examples"""
    print("Paper2Code Agent - Basic Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_paper_processing()
        example_pdf_paper_processing()
        example_doi_paper_processing()
        example_batch_processing()
        example_github_integration()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()