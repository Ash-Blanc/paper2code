"""
Paper Processing Pipeline

This module implements the complete paper processing pipeline that coordinates
all agents and integrations to convert scientific papers into code implementations.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

from .agents import (
    PaperAnalysisAgent, PaperAnalysisResult,
    ResearchAgent, ResearchResult,
    ArchitectureAgent, ArchitectureResult,
    CodeGenerationAgent, CodeGenerationResult,
    DocumentationAgent, DocumentationResult,
    QualityAssuranceAgent, ValidationResult
)
from .models.paper import Paper, PaperMetadata
from .models.code import CodeImplementation
from .integrations import GitHubIntegration, GitHubRepositoryResult
from .cache import CacheManager
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the paper processing pipeline"""
    model: str = "gpt-4o"
    temperature: float = 0.1
    output_level: str = "standard"
    create_github_repo: bool = True
    github_token: Optional[str] = None
    github_organization: str = "paper2code-repos"
    enable_caching: bool = True
    cache_ttl: int = 2592000  # 30 days
    enable_parallel_processing: bool = True
    max_concurrent_tasks: int = 4
    enable_validation: bool = True
    enable_documentation: bool = True
    log_level: str = "INFO"
    working_directory: str = "./output"
    temp_directory: str = "/tmp/paper2code"


@dataclass
class PipelineResult:
    """Result from paper processing pipeline"""
    success: bool
    paper_analysis: Optional[PaperAnalysisResult] = None
    research_results: Optional[ResearchResult] = None
    architecture_result: Optional[ArchitectureResult] = None
    code_implementation: Optional[CodeGenerationResult] = None
    documentation: Optional[DocumentationResult] = None
    validation: Optional[ValidationResult] = None
    github_repository: Optional[GitHubRepositoryResult] = None
    processing_time: float = 0.0
    steps_completed: List[str] = None
    errors: List[str] = None
    warnings: List[str] = None
    output_directory: Optional[str] = None
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class PaperProcessingPipeline:
    """Main paper processing pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize components
        self.agents = self._initialize_agents()
        self.cache_manager = CacheManager() if config.enable_caching else None
        self.prompt_manager = PromptManager()
        self.github_integration = None
        
        if config.create_github_repo and config.github_token:
            self.github_integration = GitHubIntegration(
                token=config.github_token,
                organization=config.github_organization
            )
        
        # Create working directories
        self._create_directories()
        
        logger.info("PaperProcessingPipeline initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents"""
        agents = {}
        
        try:
            # Initialize each agent with the configured model
            agents['paper_analysis'] = PaperAnalysisAgent(self.config.model)
            agents['research'] = ResearchAgent(self.config.model)
            agents['architecture'] = ArchitectureAgent(self.config.model)
            agents['code_generation'] = CodeGenerationAgent(self.config.model)
            agents['documentation'] = DocumentationAgent(self.config.model)
            agents['quality_assurance'] = QualityAssuranceAgent(self.config.model)
            
            logger.info("All agents initialized successfully")
            return agents
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary directories"""
        try:
            # Create working directory
            self.working_dir = Path(self.config.working_directory)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temp directory
            self.temp_dir = Path(self.config.temp_directory)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output directory for this specific run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.working_dir / f"paper2code_{timestamp}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Directories created: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise
    
    def process_paper(self, paper_input: str, input_type: str = "pdf") -> PipelineResult:
        """Process a scientific paper through the complete pipeline"""
        start_time = time.time()
        
        logger.info(f"Starting paper processing: {paper_input} (type: {input_type})")
        
        try:
            # Check cache first
            if self.config.enable_caching and self.cache_manager:
                cache_key = self.cache_manager.get_paper_key(paper_input, input_type)
                cached_result = self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info("Returning cached result")
                    return cached_result
            
            # Process paper through pipeline
            result = self._process_paper_pipeline(paper_input, input_type)
            
            # Cache the result
            if self.config.enable_caching and self.cache_manager:
                result.processing_time = time.time() - start_time
                self.cache_manager.set(cache_key, result, self.config.cache_ttl)
            
            logger.info(f"Paper processing completed in {result.processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            return PipelineResult(
                success=False,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _process_paper_pipeline(self, paper_input: str, input_type: str) -> PipelineResult:
        """Process paper through the complete pipeline"""
        result = PipelineResult(success=True)
        
        try:
            # Step 1: Paper Analysis
            logger.info("Step 1: Analyzing paper...")
            paper_analysis = self.agents['paper_analysis'].analyze_paper(paper_input, input_type)
            
            if not paper_analysis.success:
                result.success = False
                result.errors.extend(paper_analysis.errors)
                return result
            
            result.paper_analysis = paper_analysis
            result.steps_completed.append("paper_analysis")
            
            # Step 2: Research
            logger.info("Step 2: Researching similar implementations...")
            research_results = self.agents['research'].research_paper(
                paper_analysis.paper, paper_analysis.metadata
            )
            
            if not research_results.success:
                result.success = False
                result.errors.extend(research_results.errors)
                return result
            
            result.research_results = research_results
            result.steps_completed.append("research")
            
            # Step 3: Architecture Design
            logger.info("Step 3: Designing architecture...")
            architecture_result = self.agents['architecture'].design_architecture(
                paper_analysis.paper, research_results, self.config.output_level
            )
            
            if not architecture_result.success:
                result.success = False
                result.errors.extend(architecture_result.errors)
                return result
            
            result.architecture_result = architecture_result
            result.steps_completed.append("architecture")
            
            # Step 4: Code Generation
            logger.info("Step 4: Generating code...")
            code_generation_result = self.agents['code_generation'].generate_code(
                paper_analysis.paper, research_results, architecture_result
            )
            
            if not code_generation_result.success:
                result.success = False
                result.errors.extend(code_generation_result.errors)
                return result
            
            result.code_implementation = code_generation_result
            result.steps_completed.append("code_generation")
            
            # Step 5: Save generated code to output directory
            self._save_generated_code(code_generation_result, result.output_directory)
            
            # Step 6: Documentation Generation
            if self.config.enable_documentation:
                logger.info("Step 6: Generating documentation...")
                documentation_result = self.agents['documentation'].generate_documentation(
                    paper_analysis.paper, code_generation_result.code_implementation, 
                    architecture_result
                )
                
                if not documentation_result.success:
                    result.warnings.extend(documentation_result.errors)
                else:
                    result.documentation = documentation_result
                    result.steps_completed.append("documentation")
                
                # Save documentation
                self._save_documentation(documentation_result, result.output_directory)
            
            # Step 7: Quality Assurance
            if self.config.enable_validation:
                logger.info("Step 7: Performing quality assurance...")
                validation_result = self.agents['quality_assurance'].validate_code_implementation(
                    paper_analysis.paper, code_generation_result.code_implementation,
                    architecture_result
                )
                
                result.validation = validation_result
                result.steps_completed.append("validation")
                
                # Check if validation passed
                if not validation_result.is_valid:
                    result.warnings.extend(validation_result.issues)
                    if validation_result.issues:
                        result.success = False
            
            # Step 8: GitHub Repository Creation
            if self.config.create_github_repo and self.github_integration:
                logger.info("Step 8: Creating GitHub repository...")
                github_result = self.github_integration.create_repository(
                    paper_analysis.paper.metadata,
                    code_generation_result.code_implementation
                )
                
                if not github_result.success:
                    result.warnings.extend(github_result.issues)
                else:
                    result.github_repository = github_result
                    result.steps_completed.append("github_repository")
            
            # Generate final report
            self._generate_final_report(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            result.success = False
            result.errors.append(str(e))
            return result
    
    def _save_generated_code(self, code_result: CodeGenerationResult, output_dir: str):
        """Save generated code to output directory"""
        try:
            output_path = Path(output_dir) / "generated_code"
            output_path.mkdir(parents=True, exist_ok=True)
            
            for file_info in code_result.generated_files:
                file_path = output_path / file_info.name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_info.content)
                
                logger.info(f"Saved code file: {file_path}")
            
            logger.info(f"Code saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving generated code: {e}")
    
    def _save_documentation(self, documentation_result: DocumentationResult, output_dir: str):
        """Save documentation to output directory"""
        try:
            output_path = Path(output_dir) / "documentation"
            output_path.mkdir(parents=True, exist_ok=True)
            
            for doc_file in documentation_result.documentation_files:
                file_path = output_path / doc_file.name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc_file.content)
                
                logger.info(f"Saved documentation file: {file_path}")
            
            logger.info(f"Documentation saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving documentation: {e}")
    
    def _generate_final_report(self, result: PipelineResult):
        """Generate final processing report"""
        try:
            report = {
                "paper_title": result.paper_analysis.paper.metadata.title if result.paper_analysis else "Unknown",
                "processing_date": datetime.now().isoformat(),
                "processing_time": result.processing_time,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "errors": result.errors,
                "warnings": result.warnings,
                "statistics": {
                    "cache_hit_rate": self.cache_manager.get_hit_rate() if self.cache_manager else 0,
                    "total_processed": self.cache_manager.get_total_processed() if self.cache_manager else 0
                },
                "paper_analysis": asdict(result.paper_analysis) if result.paper_analysis else None,
                "research_results": asdict(result.research_results) if result.research_results else None,
                "architecture_result": asdict(result.architecture_result) if result.architecture_result else None,
                "code_implementation": asdict(result.code_implementation) if result.code_implementation else None,
                "documentation": asdict(result.documentation) if result.documentation else None,
                "validation": asdict(result.validation) if result.validation else None,
                "github_repository": asdict(result.github_repository) if result.github_repository else None
            }
            
            # Save report
            report_path = Path(result.output_directory) / "processing_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Final report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    def batch_process_papers(self, paper_inputs: List[Dict[str, str]]) -> List[PipelineResult]:
        """Process multiple papers in batch"""
        logger.info(f"Starting batch processing of {len(paper_inputs)} papers")
        
        results = []
        
        if self.config.enable_parallel_processing:
            # Process papers in parallel
            results = self._process_papers_parallel(paper_inputs)
        else:
            # Process papers sequentially
            for paper_input in paper_inputs:
                result = self.process_paper(
                    paper_input['input'],
                    paper_input.get('type', 'pdf')
                )
                results.append(result)
        
        logger.info(f"Batch processing completed. Success: {sum(1 for r in results if r.success)}/{len(results)}")
        return results
    
    def _process_papers_parallel(self, paper_inputs: List[Dict[str, str]]) -> List[PipelineResult]:
        """Process papers in parallel using asyncio"""
        async def process_single_paper(paper_input):
            return self.process_paper(
                paper_input['input'],
                paper_input.get('type', 'pdf')
            )
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def process_with_semaphore(paper_input):
            async with semaphore:
                return await process_single_paper(paper_input)
        
        # Run all tasks concurrently
        tasks = [process_with_semaphore(paper_input) for paper_input in paper_inputs]
        
        # Wait for all tasks to complete
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        return results
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics"""
        stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "average_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "cache_size": 0,
            "steps_completion_rate": {}
        }
        
        if self.cache_manager:
            stats["cache_hit_rate"] = self.cache_manager.get_hit_rate()
            stats["cache_size"] = self.cache_manager.get_cache_size()
            stats["total_processed"] = self.cache_manager.get_total_processed()
        
        return stats
    
    def clear_cache(self):
        """Clear the cache"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
            logger.info("Cache cleared")
    
    def warm_cache(self, popular_papers: List[str]):
        """Warm cache with popular papers"""
        if self.cache_manager:
            logger.info("Warming cache with popular papers...")
            self.cache_manager.warm_cache_for_domain("machine_learning", limit=50)
            logger.info("Cache warming completed")
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
    
    def __del__(self):
        """Cleanup when pipeline is destroyed"""
        self.cleanup_temp_files()


# Factory function for easy pipeline creation
def create_pipeline(config: Optional[PipelineConfig] = None) -> PaperProcessingPipeline:
    """Create a paper processing pipeline with default or custom configuration"""
    if config is None:
        config = PipelineConfig()
    
    return PaperProcessingPipeline(config)


# CLI interface for the pipeline
def main():
    """CLI interface for paper processing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper2Code Processing Pipeline")
    parser.add_argument("paper_input", help="Paper input (PDF file path, arXiv URL, or DOI)")
    parser.add_argument("--type", choices=["pdf", "arxiv", "doi"], default="pdf", 
                       help="Input type (default: pdf)")
    parser.add_argument("--output-level", choices=["minimal", "standard", "production"], 
                       default="standard", help="Output level (default: standard)")
    parser.add_argument("--no-github", action="store_true", 
                       help="Don't create GitHub repository")
    parser.add_argument("--github-token", help="GitHub token for repository creation")
    parser.add_argument("--github-org", default="paper2code-repos", 
                       help="GitHub organization (default: paper2code-repos)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="Temperature for generation (default: 0.1)")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Disable caching")
    parser.add_argument("--no-validation", action="store_true", 
                       help="Disable validation")
    parser.add_argument("--no-documentation", action="store_true", 
                       help="Disable documentation")
    parser.add_argument("--output-dir", default="./output", 
                       help="Output directory (default: ./output)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Log level (default: INFO)")
    parser.add_argument("--stats", action="store_true", 
                       help="Show pipeline statistics")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Clear the cache")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        model=args.model,
        temperature=args.temperature,
        output_level=args.output_level,
        create_github_repo=not args.no_github,
        github_token=args.github_token,
        github_organization=args.github_org,
        enable_caching=not args.no_cache,
        enable_validation=not args.no_validation,
        enable_documentation=not args.no_documentation,
        log_level=args.log_level,
        working_directory=args.output_dir
    )
    
    # Create pipeline
    pipeline = create_pipeline(config)
    
    # Handle special commands
    if args.clear_cache:
        pipeline.clear_cache()
        print("Cache cleared")
        return
    
    if args.stats:
        stats = pipeline.get_pipeline_statistics()
        print("Pipeline Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return
    
    # Process paper
    result = pipeline.process_paper(args.paper_input, args.type)
    
    # Display results
    print(f"\nProcessing Results:")
    print(f"  Success: {'✅' if result.success else '❌'}")
    print(f"  Processing Time: {result.processing_time:.2f} seconds")
    print(f"  Steps Completed: {len(result.steps_completed)}/{8}")
    print(f"  Output Directory: {result.output_directory}")
    
    if result.github_repository:
        print(f"  GitHub Repository: {result.github_repository.repository_url}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Cleanup
    pipeline.cleanup_temp_files()
    
    # Exit with appropriate code
    exit(0 if result.success else 1)


if __name__ == "__main__":
    main()