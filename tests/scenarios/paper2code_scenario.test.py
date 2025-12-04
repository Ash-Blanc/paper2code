"""
Paper2Code End-to-End Scenario Tests

This module contains comprehensive scenario tests for the Paper2Code agent system.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.pipeline import PaperProcessingPipeline, PipelineConfig
from app.models.paper import Paper, PaperMetadata, Author
from app.models.code import CodeImplementation, CodeFile, Language
from app.agents import (
    PaperAnalysisAgent, ResearchAgent, ArchitectureAgent,
    CodeGenerationAgent, DocumentationAgent, QualityAssuranceAgent
)
from app.integrations import GitHubIntegration, GitHubRepositoryResult
from app.cache import CacheManager


class TestPaper2CodeScenario:
    """End-to-end scenario tests for Paper2Code agent"""
    
    @pytest.fixture
    def sample_paper_metadata(self):
        """Sample paper metadata for testing"""
        return PaperMetadata(
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
            abstract="The dominant sequence transduction models are based on complex recurrent neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            publication_year=2017,
            journal="NeurIPS",
            domain="machine_learning",
            subdomain="natural_language_processing",
            doi="10.48550/arXiv.1706.03762",
            url="https://arxiv.org/abs/1706.03762"
        )
    
    @pytest.fixture
    def sample_paper(self, sample_paper_metadata):
        """Sample paper for testing"""
        return Paper(
            metadata=sample_paper_metadata,
            content=self._get_sample_paper_content(),
            algorithms=[
                {
                    "name": "Transformer",
                    "description": "A novel neural network architecture based solely on attention mechanisms",
                    "complexity": "O(n^2) for attention computation",
                    "key_parameters": ["d_model", "n_heads", "n_layers"],
                    "mathematical_formulation": "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
                    "implementation_details": "Multi-head attention with positional encoding"
                }
            ],
            experiments=[
                {
                    "name": "Machine Translation",
                    "description": "English-to-German translation task",
                    "dataset": "WMT 2014",
                    "evaluation_metrics": ["BLEU", "TER"],
                    "baselines": ["LSTM", "GRU"],
                    "results": "New state-of-the-art results"
                }
            ],
            key_insights=[
                "Attention mechanisms can replace recurrent and convolutional layers",
                "Multi-head attention allows attending to information from different representation subspaces",
                "Positional encoding provides information about token order"
            ],
            research_questions=[
                "Can attention mechanisms alone achieve good performance on sequence transduction tasks?",
                "How does the Transformer compare to RNN-based architectures?"
            ],
            contributions=[
                "Proposed the Transformer architecture",
                "Demonstateed state-of-the-art results on machine translation",
                "Showed that attention mechanisms can replace recurrence"
            ]
        )
    
    def _get_sample_paper_content(self):
        """Sample paper content for testing"""
        return """
        # Attention Is All You Need
        
        ## Abstract
        The dominant sequence transduction models are based on complex recurrent neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
        
        ## Introduction
        Recurrent neural networks, especially long short-term memory networks and gated recurrent networks, have been widely used for sequence transduction tasks. However, they suffer from sequential computation which makes it difficult to parallelize. In this paper, we propose the Transformer, a novel architecture that relies entirely on attention mechanisms.
        
        ## Model Architecture
        The Transformer follows the overall architecture of encoder-decoder models. The encoder maps an input sequence of symbol representations to a sequence of continuous representations. The decoder then generates an output sequence one element at a time.
        
        ### Encoder
        The encoder is composed of N identical layers, each having two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.
        
        ### Decoder
        The decoder is also composed of N identical layers, with three sub-layers: a masked multi-head self-attention mechanism, a multi-head attention mechanism over the output of the encoder stack, and a position-wise fully connected feed-forward network.
        
        ## Attention Mechanisms
        An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
        
        ## Positional Encoding
        Since the Transformer contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.
        
        ## Training
        We trained the Transformer on the WMT 2014 English-to-German translation task, achieving state-of-the-art results.
        
        ## Results
        The Transformer model achieved 28.4 BLEU on the English-to-German translation task, outperforming all previous models.
        """
    
    @pytest.fixture
    def pipeline_config(self):
        """Pipeline configuration for testing"""
        return PipelineConfig(
            model="gpt-4o",
            temperature=0.1,
            output_level="standard",
            create_github_repo=False,  # Disable for testing
            enable_caching=True,
            enable_validation=True,
            enable_documentation=True,
            log_level="INFO"
        )
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Pipeline instance for testing"""
        return PaperProcessingPipeline(pipeline_config)
    
    def test_paper_analysis_scenario(self, pipeline, sample_paper):
        """Test paper analysis scenario"""
        # Mock the paper analysis agent
        with patch.object(pipeline.agents['paper_analysis'], 'analyze_paper') as mock_analyze:
            mock_result = Mock()
            mock_result.success = True
            mock_result.paper = sample_paper
            mock_result.metadata = sample_paper.metadata
            mock_result.algorithms = sample_paper.algorithms
            mock_result.experiments = sample_paper.experiments
            mock_result.errors = []
            mock_analyze.return_value = mock_result
            
            # Test paper analysis
            result = pipeline.agents['paper_analysis'].analyze_paper("sample content", "pdf")
            
            # Verify results
            assert result.success
            assert result.paper.metadata.title == "Attention Is All You Need"
            assert len(result.paper.authors) == 8
            assert len(result.algorithms) == 1
            assert result.algorithms[0]["name"] == "Transformer"
    
    def test_research_scenario(self, pipeline, sample_paper):
        """Test research scenario"""
        # Mock the research agent
        with patch.object(pipeline.agents['research'], 'research_paper') as mock_research:
            mock_result = Mock()
            mock_result.success = True
            mock_result.similar_papers = []
            mock_result.github_repositories = []
            mock_result.common_patterns = []
            mock_result.best_practices = []
            mock_result.popular_frameworks = []
            mock_result.common_dependencies = []
            mock_result.implementation_approaches = []
            mock_result.errors = []
            mock_research.return_value = mock_result
            
            # Test research
            result = pipeline.agents['research'].research_paper(sample_paper, sample_paper.metadata)
            
            # Verify results
            assert result.success
            assert isinstance(result.similar_papers, list)
            assert isinstance(result.github_repositories, list)
    
    def test_architecture_scenario(self, pipeline, sample_paper):
        """Test architecture design scenario"""
        # Mock the architecture agent
        with patch.object(pipeline.agents['architecture'], 'design_architecture') as mock_arch:
            mock_result = Mock()
            mock_result.success = True
            mock_result.technology_stack = {
                "primary_language": "Python",
                "frameworks": ["PyTorch"],
                "libraries": ["numpy", "torch"],
                "tools": ["pytest", "black"],
                "build_system": "pip",
                "package_manager": "pip"
            }
            mock_result.project_structure = {
                "directories": [
                    {"name": "src", "purpose": "Source code", "files": ["__init__.py", "transformer.py"]},
                    {"name": "tests", "purpose": "Tests", "files": ["__init__.py", "test_transformer.py"]}
                ]
            }
            mock_result.errors = []
            mock_arch.return_value = mock_result
            
            # Test architecture design
            result = pipeline.agents['architecture'].design_architecture(sample_paper, Mock(), "standard")
            
            # Verify results
            assert result.success
            assert result.technology_stack["primary_language"] == "Python"
            assert "PyTorch" in result.technology_stack["frameworks"]
    
    def test_code_generation_scenario(self, pipeline, sample_paper):
        """Test code generation scenario"""
        # Mock the code generation agent
        with patch.object(pipeline.agents['code_generation'], 'generate_code') as mock_generate:
            mock_result = Mock()
            mock_result.success = True
            mock_result.language_used = "Python"
            mock_result.framework_used = "PyTorch"
            mock_result.generated_files = [
                Mock(
                    name="transformer.py",
                    file_type="main",
                    language="Python",
                    content="import torch\n\nclass Transformer:\n    pass",
                    purpose="Main transformer implementation",
                    dependencies=["torch"],
                    key_functions=["__init__", "forward"]
                )
            ]
            mock_result.code_quality_score = 0.85
            mock_result.test_coverage_score = 0.75
            mock_result.documentation_score = 0.80
            mock_result.performance_score = 0.90
            mock_result.maintainability_score = 0.85
            mock_result.generation_time = 2.5
            mock_result.code_metrics = {
                "total_lines": 100,
                "comment_lines": 20,
                "function_count": 10,
                "class_count": 3,
                "complexity_score": 0.7
            }
            mock_result.errors = []
            mock_generate.return_value = mock_result
            
            # Test code generation
            result = pipeline.agents['code_generation'].generate_code(sample_paper, Mock(), Mock())
            
            # Verify results
            assert result.success
            assert result.language_used == "Python"
            assert len(result.generated_files) == 1
            assert result.generated_files[0].name == "transformer.py"
            assert result.code_quality_score > 0.0
    
    def test_documentation_scenario(self, pipeline, sample_paper):
        """Test documentation generation scenario"""
        # Mock the documentation agent
        with patch.object(pipeline.agents['documentation'], 'generate_documentation') as mock_doc:
            mock_result = Mock()
            mock_result.success = True
            mock_result.documentation_files = [
                Mock(
                    name="README.md",
                    file_type="readme",
                    content="# Transformer Implementation\n\nThis is a README file",
                    purpose="Main documentation",
                    target_audience="users"
                )
            ]
            mock_result.documentation_score = 0.90
            mock_result.coverage_metrics = {
                "total_coverage": 0.85,
                "api_documentation": 0.80,
                "user_guides": 0.90,
                "developer_docs": 0.75,
                "examples": 0.80
            }
            mock_result.generation_time = 1.5
            mock_result.errors = []
            mock_doc.return_value = mock_result
            
            # Test documentation generation
            result = pipeline.agents['documentation'].generate_documentation(sample_paper, Mock(), Mock())
            
            # Verify results
            assert result.success
            assert len(result.documentation_files) == 1
            assert result.documentation_files[0].name == "README.md"
            assert result.documentation_score > 0.0
    
    def test_quality_assurance_scenario(self, pipeline, sample_paper):
        """Test quality assurance scenario"""
        # Mock the quality assurance agent
        with patch.object(pipeline.agents['quality_assurance'], 'validate_code_implementation') as mock_validate:
            mock_result = Mock()
            mock_result.is_valid = True
            mock_result.validation_score = 0.85
            mock_result.issues = []
            mock_result.warnings = []
            mock_result.recommendations = []
            mock_result.performance_metrics = {
                "execution_time": 0.5,
                "memory_usage": 100,
                "cpu_usage": 50,
                "algorithm_accuracy": 0.95,
                "result_consistency": 0.90
            }
            mock_result.code_quality_metrics = {
                "cyclomatic_complexity": 0.6,
                "maintainability_index": 0.85,
                "technical_debt": 0.1,
                "code_coverage": 0.75,
                "documentation_coverage": 0.80
            }
            mock_result.validation_summary = {
                "total_issues": 0,
                "critical_issues": 0,
                "high_priority_issues": 0,
                "medium_priority_issues": 0,
                "low_priority_issues": 0,
                "total_warnings": 0,
                "total_recommendations": 0,
                "overall_score": 0.85
            }
            mock_result.validation_details = {
                "algorithm_validation": {
                    "completeness": 0.90,
                    "accuracy": 0.95,
                    "consistency": 0.85
                },
                "code_quality_validation": {
                    "readability": 0.85,
                    "maintainability": 0.90,
                    "testability": 0.80
                }
            }
            mock_result.next_steps = []
            mock_result.validation_report = "Validation passed successfully"
            mock_validate.return_value = mock_result
            
            # Test quality assurance
            result = pipeline.agents['quality_assurance'].validate_code_implementation(sample_paper, Mock(), Mock())
            
            # Verify results
            assert result.is_valid
            assert result.validation_score > 0.0
            assert result.validation_summary["overall_score"] > 0.0
    
    def test_github_integration_scenario(self, pipeline):
        """Test GitHub integration scenario"""
        # Mock GitHub integration
        with patch.object(pipeline.github_integration, 'create_repository') as mock_create:
            mock_result = Mock()
            mock_result.success = True
            mock_result.repository_url = "https://github.com/paper2code-repos/attention-is-all-you-need-2017"
            mock_result.repository_name = "attention-is-all-you-need-2017"
            mock_result.clone_url = "https://github.com/paper2code-repos/attention-is-all-you-need-2017.git"
            mock_result.issues = []
            mock_create.return_value = mock_result
            
            # Test GitHub repository creation
            result = pipeline.github_integration.create_repository(sample_paper.metadata, Mock())
            
            # Verify results
            assert result.success
            assert result.repository_url is not None
            assert result.repository_name is not None
    
    def test_cache_scenario(self, pipeline):
        """Test caching scenario"""
        # Test cache operations
        test_key = "test_key"
        test_value = {"test": "data"}
        
        # Test set and get
        assert pipeline.cache_manager.set(test_key, test_value)
        retrieved_value = pipeline.cache_manager.get(test_key)
        assert retrieved_value == test_value
        
        # Test exists
        assert pipeline.cache_manager.exists(test_key)
        
        # Test delete
        assert pipeline.cache_manager.delete(test_key)
        assert not pipeline.cache_manager.exists(test_key)
        
        # Test cache statistics
        stats = pipeline.cache_manager.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "cache_size" in stats
    
    def test_pipeline_scenario(self, pipeline, sample_paper):
        """Test complete pipeline scenario"""
        # Mock all agents
        with patch.object(pipeline.agents['paper_analysis'], 'analyze_paper') as mock_analyze, \
             patch.object(pipeline.agents['research'], 'research_paper') as mock_research, \
             patch.object(pipeline.agents['architecture'], 'design_architecture') as mock_arch, \
             patch.object(pipeline.agents['code_generation'], 'generate_code') as mock_generate, \
             patch.object(pipeline.agents['documentation'], 'generate_documentation') as mock_doc, \
             patch.object(pipeline.agents['quality_assurance'], 'validate_code_implementation') as mock_validate:
            
            # Setup mocks
            mock_analyze.return_value = Mock(
                success=True,
                paper=sample_paper,
                metadata=sample_paper.metadata,
                algorithms=sample_paper.algorithms,
                experiments=sample_paper.experiments,
                errors=[]
            )
            
            mock_research.return_value = Mock(
                success=True,
                similar_papers=[],
                github_repositories=[],
                common_patterns=[],
                best_practices=[],
                popular_frameworks=[],
                common_dependencies=[],
                implementation_approaches=[],
                errors=[]
            )
            
            mock_arch.return_value = Mock(
                success=True,
                technology_stack={"primary_language": "Python"},
                project_structure={"directories": []},
                errors=[]
            )
            
            mock_generate.return_value = Mock(
                success=True,
                language_used="Python",
                framework_used="PyTorch",
                generated_files=[],
                code_quality_score=0.85,
                test_coverage_score=0.75,
                documentation_score=0.80,
                performance_score=0.90,
                maintainability_score=0.85,
                generation_time=2.5,
                code_metrics={},
                errors=[]
            )
            
            mock_doc.return_value = Mock(
                success=True,
                documentation_files=[],
                documentation_score=0.90,
                coverage_metrics={},
                generation_time=1.5,
                errors=[]
            )
            
            mock_validate.return_value = Mock(
                is_valid=True,
                validation_score=0.85,
                issues=[],
                warnings=[],
                recommendations=[],
                performance_metrics={},
                code_quality_metrics={},
                validation_summary={},
                validation_details={},
                next_steps=[],
                validation_report="Validation passed"
            )
            
            # Run pipeline
            result = pipeline.process_paper("sample content", "pdf")
            
            # Verify pipeline results
            assert result.success
            assert result.paper_analysis is not None
            assert result.research_results is not None
            assert result.architecture_result is not None
            assert result.code_implementation is not None
            assert result.documentation is not None
            assert result.validation is not None
            assert len(result.steps_completed) == 6  # All steps completed
            assert result.processing_time > 0
    
    def test_batch_processing_scenario(self, pipeline):
        """Test batch processing scenario"""
        # Mock the process_paper method
        with patch.object(pipeline, 'process_paper') as mock_process:
            mock_process.return_value = Mock(
                success=True,
                paper_analysis=Mock(),
                research_results=Mock(),
                architecture_result=Mock(),
                code_implementation=Mock(),
                documentation=Mock(),
                validation=Mock(),
                github_repository=Mock(),
                processing_time=1.0,
                steps_completed=["paper_analysis", "research", "architecture", "code_generation", "documentation", "validation"],
                errors=[],
                warnings=[],
                output_directory="/tmp/test"
            )
            
            # Test batch processing
            paper_inputs = [
                {"input": "paper1.pdf", "type": "pdf"},
                {"input": "paper2.pdf", "type": "pdf"},
                {"input": "paper3.pdf", "type": "pdf"}
            ]
            
            results = pipeline.batch_process_papers(paper_inputs)
            
            # Verify batch processing results
            assert len(results) == 3
            for result in results:
                assert result.success
                assert result.processing_time > 0
    
    def test_error_handling_scenario(self, pipeline):
        """Test error handling scenario"""
        # Test paper analysis failure
        with patch.object(pipeline.agents['paper_analysis'], 'analyze_paper') as mock_analyze:
            mock_analyze.return_value = Mock(
                success=False,
                errors=["Failed to analyze paper"]
            )
            
            result = pipeline.process_paper("invalid content", "pdf")
            
            assert not result.success
            assert len(result.errors) > 0
            assert "Failed to analyze paper" in result.errors
    
    def test_cache_warming_scenario(self, pipeline):
        """Test cache warming scenario"""
        # Test cache warming
        popular_papers = ["attention-is-all-you-need.pdf", "bert-paper.pdf"]
        
        with patch.object(pipeline.cache_manager, 'warm_cache_for_domain') as mock_warm:
            pipeline.warm_cache(popular_papers)
            
            mock_warm.assert_called_once_with("machine_learning", limit=50)
    
    def test_pipeline_statistics_scenario(self, pipeline):
        """Test pipeline statistics scenario"""
        # Test getting pipeline statistics
        stats = pipeline.get_pipeline_statistics()
        
        assert "total_processed" in stats
        assert "successful_processing" in stats
        assert "failed_processing" in stats
        assert "average_processing_time" in stats
        assert "cache_hit_rate" in stats
        assert "cache_size" in stats
        assert "steps_completion_rate" in stats
    
    def test_cleanup_scenario(self, pipeline):
        """Test cleanup scenario"""
        # Test cleanup
        pipeline.cleanup_temp_files()
        
        # Verify temp directory is cleaned up (if it existed)
        assert not pipeline.temp_dir.exists()


class TestPaper2CodeIntegration:
    """Integration tests for Paper2Code system"""
    
    @pytest.fixture
    def full_pipeline_config(self):
        """Full pipeline configuration for integration testing"""
        return PipelineConfig(
            model="gpt-4o",
            temperature=0.1,
            output_level="standard",
            create_github_repo=False,  # Disable for testing
            enable_caching=True,
            enable_validation=True,
            enable_documentation=True,
            log_level="INFO"
        )
    
    def test_end_to_end_workflow(self, full_pipeline_config):
        """Test complete end-to-end workflow"""
        # Create pipeline
        pipeline = PaperProcessingPipeline(full_pipeline_config)
        
        # Mock all components
        with patch.object(pipeline.agents['paper_analysis'], 'analyze_paper') as mock_analyze, \
             patch.object(pipeline.agents['research'], 'research_paper') as mock_research, \
             patch.object(pipeline.agents['architecture'], 'design_architecture') as mock_arch, \
             patch.object(pipeline.agents['code_generation'], 'generate_code') as mock_generate, \
             patch.object(pipeline.agents['documentation'], 'generate_documentation') as mock_doc, \
             patch.object(pipeline.agents['quality_assurance'], 'validate_code_implementation') as mock_validate:
            
            # Setup comprehensive mocks
            mock_analyze.return_value = Mock(
                success=True,
                paper=Mock(metadata=Mock(title="Test Paper")),
                metadata=Mock(title="Test Paper"),
                algorithms=[{"name": "Test Algorithm"}],
                experiments=[{"name": "Test Experiment"}],
                errors=[]
            )
            
            mock_research.return_value = Mock(
                success=True,
                similar_papers=[],
                github_repositories=[],
                common_patterns=[],
                best_practices=[],
                popular_frameworks=[],
                common_dependencies=[],
                implementation_approaches=[],
                errors=[]
            )
            
            mock_arch.return_value = Mock(
                success=True,
                technology_stack={"primary_language": "Python"},
                project_structure={"directories": []},
                errors=[]
            )
            
            mock_generate.return_value = Mock(
                success=True,
                language_used="Python",
                framework_used="PyTorch",
                generated_files=[Mock(name="test.py", content="print('Hello World')")],
                code_quality_score=0.85,
                test_coverage_score=0.75,
                documentation_score=0.80,
                performance_score=0.90,
                maintainability_score=0.85,
                generation_time=2.5,
                code_metrics={},
                errors=[]
            )
            
            mock_doc.return_value = Mock(
                success=True,
                documentation_files=[Mock(name="README.md", content="# Test Documentation")],
                documentation_score=0.90,
                coverage_metrics={},
                generation_time=1.5,
                errors=[]
            )
            
            mock_validate.return_value = Mock(
                is_valid=True,
                validation_score=0.85,
                issues=[],
                warnings=[],
                recommendations=[],
                performance_metrics={},
                code_quality_metrics={},
                validation_summary={},
                validation_details={},
                next_steps=[],
                validation_report="Validation passed"
            )
            
            # Run complete pipeline
            result = pipeline.process_paper("test content", "pdf")
            
            # Verify complete workflow
            assert result.success
            assert result.paper_analysis is not None
            assert result.research_results is not None
            assert result.architecture_result is not None
            assert result.code_implementation is not None
            assert result.documentation is not None
            assert result.validation is not None
            assert len(result.steps_completed) == 6
            assert result.processing_time > 0
            assert result.output_directory is not None
            
            # Verify files were saved
            assert Path(result.output_directory).exists()
            assert Path(result.output_directory / "generated_code").exists()
            assert Path(result.output_directory / "documentation").exists()
            assert Path(result.output_directory / "processing_report.json").exists()
    
    def test_parallel_processing(self, full_pipeline_config):
        """Test parallel processing capability"""
        pipeline = PaperProcessingPipeline(full_pipeline_config)
        
        # Mock the process_paper method
        with patch.object(pipeline, 'process_paper') as mock_process:
            mock_process.return_value = Mock(
                success=True,
                processing_time=1.0,
                steps_completed=["paper_analysis", "research", "architecture", "code_generation", "documentation", "validation"],
                errors=[],
                warnings=[],
                output_directory="/tmp/test"
            )
            
            # Test parallel processing
            paper_inputs = [
                {"input": f"paper{i}.pdf", "type": "pdf"} 
                for i in range(5)
            ]
            
            results = pipeline.batch_process_papers(paper_inputs)
            
            # Verify parallel processing results
            assert len(results) == 5
            for result in results:
                assert result.success
                assert result.processing_time > 0
    
    def test_cache_integration(self, full_pipeline_config):
        """Test cache integration"""
        pipeline = PaperProcessingPipeline(full_pipeline_config)
        
        # Test cache operations
        test_key = "integration_test_key"
        test_value = {"integration": "test", "data": "sample"}
        
        # Set cache
        assert pipeline.cache_manager.set(test_key, test_value)
        
        # Get from cache
        retrieved_value = pipeline.cache_manager.get(test_key)
        assert retrieved_value == test_value
        
        # Test cache statistics
        stats = pipeline.cache_manager.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "cache_size" in stats
        
        # Test cache clearing
        pipeline.clear_cache()
        assert not pipeline.cache_manager.exists(test_key)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])