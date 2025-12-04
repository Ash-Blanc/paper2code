"""
Paper2Code Agent - Main Orchestrator

This module implements the main Paper2Code agent that coordinates all subagents
to convert scientific papers into production-ready code implementations.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from agno import agent, team
from agno.models.openrouter import OpenRouter

from .agents.paper_analysis import PaperAnalysisAgent, PaperAnalysisResult
from .agents.research import ResearchAgent, ResearchResult
from .agents.architecture import ArchitectureAgent, ArchitectureResult
from .agents.code_generation import CodeGenerationAgent, CodeGenerationResult
from .agents.documentation import DocumentationAgent, DocumentationResult
from .agents.quality_assurance import QualityAssuranceAgent, ValidationResult
from .models.paper import Paper
from .models.code import CodeImplementation
from .integrations.github import GitHubIntegration
from .cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class Paper2CodeResult:
    """Result from Paper2Code processing"""
    success: bool
    paper_analysis: Optional[PaperAnalysisResult] = None
    research_results: Optional[ResearchResult] = None
    architecture_result: Optional[ArchitectureResult] = None
    code_implementation: Optional[CodeGenerationResult] = None
    documentation: Optional[DocumentationResult] = None
    validation: Optional[ValidationResult] = None
    github_repository: Optional[str] = None
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class Paper2CodeAgent:
    """Main orchestrator for paper-to-code conversion"""
    
    def __init__(self, model: str = "gpt-4o", github_token: Optional[str] = None):
        self.model = model
        self.llm = OpenRouter(model=model)
        
        # Initialize all subagents
        self.paper_analysis_agent = PaperAnalysisAgent(model)
        self.research_agent = ResearchAgent(model)
        self.architecture_agent = ArchitectureAgent(model)
        self.code_generation_agent = CodeGenerationAgent(model)
        self.documentation_agent = DocumentationAgent(model)
        self.quality_assurance_agent = QualityAssuranceAgent(model)
        
        # Initialize integrations
        self.github_integration = GitHubIntegration(github_token) if github_token else None
        
        # Initialize cache manager
        self.cache_manager = CacheManager()
        
        # Create team for coordinated execution
        self.team = team(
            name="paper2code_team",
            description="Convert scientific papers to production-ready code",
            agents=[
                self.paper_analysis_agent,
                self.research_agent,
                self.architecture_agent,
                self.code_generation_agent,
                self.documentation_agent,
                self.quality_assurance_agent
            ],
            workflow=self._workflow
        )
        
        logger.info("Paper2CodeAgent initialized successfully")
    
    def process_paper(self, paper_input: str, input_type: str = "pdf", 
                     output_level: str = "standard", 
                     create_github_repo: bool = True) -> Paper2CodeResult:
        """Process a scientific paper and generate code implementation"""
        start_time = time.time()
        
        logger.info(f"Processing paper: {paper_input} (type: {input_type})")
        
        try:
            # Check cache first
            cache_key = self.cache_manager.paper_key(paper_input, input_type)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info("Returning cached result")
                return cached_result
            
            # Process paper through the pipeline
            result = self._process_paper_pipeline(
                paper_input, input_type, output_level, create_github_repo
            )
            
            # Cache the result
            result.processing_time = time.time() - start_time
            self.cache_manager.set(cache_key, result, ttl=2592000)  # 30 days
            
            logger.info(f"Paper processing completed in {result.processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            return Paper2CodeResult(
                success=False,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _process_paper_pipeline(self, paper_input: str, input_type: str,
                               output_level: str, create_github_repo: bool) -> Paper2CodeResult:
        """Process paper through the complete pipeline"""
        
        # Step 1: Paper Analysis
        logger.info("Step 1: Analyzing paper...")
        paper_analysis = self.paper_analysis_agent.analyze_paper(paper_input, input_type)
        
        if not paper_analysis.success:
            return Paper2CodeResult(
                success=False,
                paper_analysis=paper_analysis,
                errors=paper_analysis.errors
            )
        
        # Step 2: Research
        logger.info("Step 2: Researching similar implementations...")
        research_results = self.research_agent.research_paper(
            paper_analysis.paper, paper_analysis.metadata
        )
        
        if not research_results.success:
            return Paper2CodeResult(
                success=False,
                paper_analysis=paper_analysis,
                research_results=research_results,
                errors=research_results.errors
            )
        
        # Step 3: Architecture Design
        logger.info("Step 3: Designing architecture...")
        architecture_result = self.architecture_agent.design_architecture(
            paper_analysis.paper, research_results, output_level
        )
        
        if not architecture_result.success:
            return Paper2CodeResult(
                success=False,
                paper_analysis=paper_analysis,
                research_results=research_results,
                architecture_result=architecture_result,
                errors=architecture_result.errors
            )
        
        # Step 4: Code Generation
        logger.info("Step 4: Generating code...")
        code_generation_result = self.code_generation_agent.generate_code(
            paper_analysis.paper, research_results, architecture_result
        )
        
        if not code_generation_result.success:
            return Paper2CodeResult(
                success=False,
                paper_analysis=paper_analysis,
                research_results=research_results,
                architecture_result=architecture_result,
                code_implementation=code_generation_result,
                errors=code_generation_result.errors
            )
        
        # Step 5: Documentation Generation
        logger.info("Step 5: Generating documentation...")
        documentation_result = self.documentation_agent.generate_documentation(
            paper_analysis.paper, code_generation_result.code_implementation, 
            architecture_result
        )
        
        if not documentation_result.success:
            return Paper2CodeResult(
                success=False,
                paper_analysis=paper_analysis,
                research_results=research_results,
                architecture_result=architecture_result,
                code_implementation=code_generation_result,
                documentation=documentation_result,
                errors=documentation_result.errors
            )
        
        # Step 6: Quality Assurance
        logger.info("Step 6: Performing quality assurance...")
        validation_result = self.quality_assurance_agent.validate_code_implementation(
            paper_analysis.paper, code_generation_result.code_implementation,
            architecture_result
        )
        
        # Step 7: GitHub Repository Creation (optional)
        github_repository = None
        if create_github_repo and self.github_integration:
            logger.info("Step 7: Creating GitHub repository...")
            try:
                github_repository = self.github_integration.create_repository(
                    paper_analysis.paper.metadata,
                    code_generation_result.code_implementation
                )
                github_repository = github_repository.html_url
            except Exception as e:
                logger.error(f"Error creating GitHub repository: {e}")
                validation_result.issues.append(f"GitHub repository creation failed: {e}")
        
        # Determine overall success
        success = validation_result.is_valid and not validation_result.issues
        
        return Paper2CodeResult(
            success=success,
            paper_analysis=paper_analysis,
            research_results=research_results,
            architecture_result=architecture_result,
            code_implementation=code_generation_result,
            documentation=documentation_result,
            validation=validation_result,
            github_repository=github_repository,
            errors=validation_result.issues if not success else []
        )
    
    def _workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Define the team workflow"""
        # This method is called by the team for coordinated execution
        # For now, we'll use the sequential pipeline approach above
        return context
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'cache_hit_rate': self.cache_manager.get_hit_rate(),
            'total_processed': self.cache_manager.get_total_processed(),
            'average_processing_time': self.cache_manager.get_average_processing_time(),
            'cache_size': self.cache_manager.get_cache_size()
        }
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache_manager.clear_cache()
        logger.info("Cache cleared")
    
    def warm_cache(self, popular_papers: List[str]) -> None:
        """Warm cache with popular papers"""
        logger.info("Warming cache with popular papers...")
        for paper in popular_papers:
            try:
                # Process paper to warm cache
                self.process_paper(paper, input_type="arxiv", create_github_repo=False)
            except Exception as e:
                logger.error(f"Error warming cache for paper {paper}: {e}")
        logger.info("Cache warming completed")
    
    def generate_report(self, result: Paper2CodeResult) -> str:
        """Generate a comprehensive processing report"""
        report = f"""
# Paper2Code Processing Report

## Summary
- **Status**: {'✅ Success' if result.success else '❌ Failed'}
- **Processing Time**: {result.processing_time:.2f} seconds
- **GitHub Repository**: {result.github_repository or 'Not created'}

## Paper Analysis
{self._generate_paper_analysis_report(result.paper_analysis)}

## Research Results
{self._generate_research_report(result.research_results)}

## Architecture Design
{self._generate_architecture_report(result.architecture_result)}

## Code Generation
{self._generate_code_generation_report(result.code_implementation)}

## Documentation
{self._generate_documentation_report(result.documentation)}

## Quality Assurance
{self._generate_validation_report(result.validation)}

## GitHub Repository
{self._generate_github_report(result.github_repository)}

## Errors
{self._generate_errors_report(result.errors)}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report
    
    def _generate_paper_analysis_report(self, analysis: Optional[PaperAnalysisResult]) -> str:
        """Generate paper analysis report"""
        if not analysis:
            return "No paper analysis available"
        
        return f"""
- **Title**: {analysis.paper.metadata.title}
- **Authors**: {', '.join([author.name for author in analysis.paper.metadata.authors])}
- **Domain**: {analysis.paper.metadata.domain}
- **Abstract**: {analysis.paper.metadata.abstract[:200]}...
- **Algorithms**: {len(analysis.paper.algorithms)}
- **Experiments**: {len(analysis.paper.experiments)}
"""
    
    def _generate_research_report(self, research: Optional[ResearchResult]) -> str:
        """Generate research report"""
        if not research:
            return "No research results available"
        
        return f"""
- **Similar Papers Found**: {len(research.similar_papers)}
- **GitHub Repositories**: {len(research.github_repositories)}
- **Common Patterns**: {len(research.common_patterns)}
- **Best Practices**: {len(research.best_practices)}
"""
    
    def _generate_architecture_report(self, architecture: Optional[ArchitectureResult]) -> str:
        """Generate architecture report"""
        if not architecture:
            return "No architecture design available"
        
        return f"""
- **Primary Language**: {architecture.technology_stack['primary_language']}
- **Frameworks**: {', '.join(architecture.technology_stack.get('frameworks', []))}
- **Output Level**: {architecture.output_level}
- **Project Structure**: {len(architecture.project_structure)} files
"""
    
    def _generate_code_generation_report(self, code: Optional[CodeGenerationResult]) -> str:
        """Generate code generation report"""
        if not code:
            return "No code generation available"
        
        return f"""
- **Language**: {code.language_used.value}
- **Framework**: {code.framework_used.value if code.framework_used else 'None'}
- **Files Generated**: {len(code.generated_files)}
- **Code Quality Score**: {code.code_quality_score:.2f}
- **Generation Time**: {code.generation_time:.2f} seconds
"""
    
    def _generate_documentation_report(self, documentation: Optional[DocumentationResult]) -> str:
        """Generate documentation report"""
        if not documentation:
            return "No documentation available"
        
        return f"""
- **Documentation Files**: {len(documentation.documentation_files)}
- **Documentation Score**: {documentation.documentation_score:.2f}
- **Generation Time**: {documentation.generation_time:.2f} seconds
- **Coverage Metrics**: {documentation.coverage_metrics.get('total_coverage', 0):.1%}
"""
    
    def _generate_validation_report(self, validation: Optional[ValidationResult]) -> str:
        """Generate validation report"""
        if not validation:
            return "No validation available"
        
        return f"""
- **Validation Score**: {validation.validation_score:.2f}
- **Is Valid**: {'✅ Yes' if validation.is_valid else '❌ No'}
- **Issues**: {len(validation.issues)}
- **Warnings**: {len(validation.warnings)}
- **Recommendations**: {len(validation.recommendations)}
"""
    
    def _generate_github_report(self, github_url: Optional[str]) -> str:
        """Generate GitHub repository report"""
        if not github_url:
            return "GitHub repository not created"
        
        return f"""
- **Repository URL**: {github_url}
- **Status**: ✅ Created successfully
"""
    
    def _generate_errors_report(self, errors: List[str]) -> str:
        """Generate errors report"""
        if not errors:
            return "No errors"
        
        return "\n".join([f"- {error}" for error in errors])


# Factory function for easy initialization
def create_paper2code_agent(model: str = "gpt-4o", github_token: Optional[str] = None) -> Paper2CodeAgent:
    """Create a Paper2Code agent instance"""
    return Paper2CodeAgent(model=model, github_token=github_token)


# CLI interface
def main():
    """CLI interface for Paper2Code agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper2Code Agent - Convert scientific papers to code")
    parser.add_argument("paper_input", help="Paper input (PDF file path, arXiv URL, or DOI)")
    parser.add_argument("--type", choices=["pdf", "arxiv", "doi"], default="pdf", 
                       help="Input type (default: pdf)")
    parser.add_argument("--output-level", choices=["minimal", "standard", "production"], 
                       default="standard", help="Output level (default: standard)")
    parser.add_argument("--no-github", action="store_true", 
                       help="Don't create GitHub repository")
    parser.add_argument("--github-token", help="GitHub token for repository creation")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--report", help="Save processing report to file")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")
    
    args = parser.parse_args()
    
    # Create agent
    agent = create_paper2code_agent(
        model=args.model,
        github_token=args.github_token
    )
    
    # Process paper
    result = agent.process_paper(
        paper_input=args.paper_input,
        input_type=args.type,
        output_level=args.output_level,
        create_github_repo=not args.no_github
    )
    
    # Generate and display report
    report = agent.generate_report(result)
    print(report)
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.report}")
    
    # Show statistics if requested
    if args.stats:
        stats = agent.get_processing_stats()
        print("\nProcessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Exit with appropriate code
    exit(0 if result.success else 1)


if __name__ == "__main__":
    main()