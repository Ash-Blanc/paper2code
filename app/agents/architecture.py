"""
Architecture Agent

This agent designs project structure and technology stack based on
paper domain and research findings, with configurable output levels.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from agno import agent
from agno.models.openrouter import OpenRouter

from ..models.paper import Paper
from ..models.code import Language, Framework, CodeImplementation
from ..models.repository import RepositoryConfig

logger = logging.getLogger(__name__)


class OutputLevel(Enum):
    """Output level configuration"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    PRODUCTION = "production"


@dataclass
class ArchitectureResult:
    """Result from architecture design"""
    project_structure: Dict[str, Any]
    technology_stack: Dict[str, Any]
    output_level: OutputLevel
    framework_recommendations: List[str]
    dependencies: List[str]
    architecture_confidence: float
    reasoning: str


class ArchitectureAgent:
    """Agent for designing project architecture and technology stack"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.llm = OpenRouter(model=model)
        
        # Create agent for architecture design
        self.agent = agent(
            name="architecture_designer",
            model=self.llm,
            description="Design project architecture and technology stack for scientific papers"
        )
    
    def determine_output_level(self, user_preference: Optional[str] = None) -> OutputLevel:
        """Determine the appropriate output level"""
        if user_preference:
            try:
                return OutputLevel(user_preference.lower())
            except ValueError:
                logger.warning(f"Invalid output level: {user_preference}, using STANDARD")
        
        # Default to STANDARD level
        return OutputLevel.STANDARD
    
    def analyze_research_domain(self, paper: Paper) -> Dict[str, Any]:
        """Analyze the research domain to determine appropriate technology"""
        domain_analysis = {
            'domain': paper.metadata.domain,
            'subdomains': [],
            'technical_requirements': [],
            'common_frameworks': [],
            'performance_requirements': []
        }
        
        # Analyze algorithms for technical requirements
        for algorithm in paper.algorithms:
            if 'neural' in algorithm.name.lower():
                domain_analysis['technical_requirements'].append('deep_learning')
                domain_analysis['common_frameworks'].extend(['pytorch', 'tensorflow'])
            elif 'machine' in algorithm.name.lower():
                domain_analysis['technical_requirements'].append('ml_framework')
                domain_analysis['common_frameworks'].extend(['scikit-learn', 'xgboost'])
            elif 'statistical' in algorithm.name.lower():
                domain_analysis['technical_requirements'].append('statistical_analysis')
                domain_analysis['common_frameworks'].extend(['pandas', 'numpy', 'statsmodels'])
        
        # Analyze experiments for performance requirements
        for experiment in paper.experiments:
            if 'large_dataset' in experiment.description.lower():
                domain_analysis['performance_requirements'].append('big_data_processing')
            if 'real_time' in experiment.description.lower():
                domain_analysis['performance_requirements'].append('real_time_processing')
        
        # Remove duplicates
        domain_analysis['common_frameworks'] = list(set(domain_analysis['common_frameworks']))
        
        return domain_analysis
    
    def recommend_technology_stack(self, paper: Paper, research_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend technology stack based on paper and research patterns"""
        tech_stack = {
            'primary_language': Language.PYTHON,
            'frameworks': [],
            'libraries': [],
            'tools': [],
            'testing_frameworks': [],
            'documentation_tools': []
        }
        
        # Analyze research patterns
        if research_patterns.get('programming_languages'):
            # Use most common language from research
            most_common_lang = max(
                research_patterns['programming_languages'].items(),
                key=lambda x: x[1]
            )[0]
            
            if most_common_lang.lower() in ['python', 'py']:
                tech_stack['primary_language'] = Language.PYTHON
            elif most_common_lang.lower() in ['javascript', 'js', 'typescript', 'ts']:
                tech_stack['primary_language'] = Language.JAVASCRIPT
            elif most_common_lang.lower() in ['java']:
                tech_stack['primary_language'] = Language.JAVA
        
        # Recommend frameworks based on domain
        domain_analysis = self.analyze_research_domain(paper)
        
        # Deep learning frameworks
        if any(req in domain_analysis['technical_requirements'] for req in ['deep_learning']):
            tech_stack['frameworks'].extend([
                Framework.PYTORCH,
                Framework.TENSORFLOW
            ])
        
        # Machine learning frameworks
        if any(req in domain_analysis['technical_requirements'] for req in ['ml_framework']):
            tech_stack['frameworks'].extend([
                Framework.SCIKITLEARN,
                Framework.XGBOOST
            ])
        
        # Statistical analysis
        if any(req in domain_analysis['technical_requirements'] for req in ['statistical_analysis']):
            tech_stack['frameworks'].extend([
                Framework.PANDAS,
                Framework.NUMPY,
                Framework.STATSMODELS
            ])
        
        # Core libraries
        tech_stack['libraries'].extend([
            'requests',  # HTTP requests
            'matplotlib',  # Plotting
            'seaborn',  # Statistical plotting
            'jupyter',  # Interactive notebooks
            'pytest'  # Testing
        ])
        
        # Tools
        tech_stack['tools'].extend([
            'git',  # Version control
            'black',  # Code formatting
            'flake8',  # Linting
            'mypy'  # Type checking
        ])
        
        # Testing frameworks
        tech_stack['testing_frameworks'].extend([
            'pytest',
            'pytest-cov',
            'unittest'
        ])
        
        # Documentation tools
        tech_stack['documentation_tools'].extend([
            'sphinx',
            'mkdocs',
            'jupyter-doc'
        ])
        
        return tech_stack
    
    def design_project_structure(self, paper: Paper, tech_stack: Dict[str, Any], 
                               output_level: OutputLevel) -> Dict[str, Any]:
        """Design project structure based on technology and output level"""
        structure = {
            'root_directory': paper.metadata.title.lower().replace(' ', '_'),
            'directories': {},
            'files': {},
            'output_level': output_level.value
        }
        
        # Basic directories
        structure['directories']['src'] = {
            'type': 'source',
            'description': 'Source code directory'
        }
        
        structure['directories']['tests'] = {
            'type': 'test',
            'description': 'Test files directory'
        }
        
        structure['directories']['docs'] = {
            'type': 'documentation',
            'description': 'Documentation directory'
        }
        
        # Additional directories based on output level
        if output_level in [OutputLevel.STANDARD, OutputLevel.PRODUCTION]:
            structure['directories']['data'] = {
                'type': 'data',
                'description': 'Data files and datasets'
            }
            
            structure['directories']['notebooks'] = {
                'type': 'notebook',
                'description': 'Jupyter notebooks for experimentation'
            }
        
        if output_level == OutputLevel.PRODUCTION:
            structure['directories']['config'] = {
                'type': 'config',
                'description': 'Configuration files'
            }
            
            structure['directories']['scripts'] = {
                'type': 'script',
                'description': 'Utility scripts'
            }
        
        # Design source code structure
        source_structure = self._design_source_structure(paper, tech_stack)
        structure['directories']['src'].update(source_structure)
        
        # Design test structure
        test_structure = self._design_test_structure(paper, tech_stack)
        structure['directories']['tests'].update(test_structure)
        
        # Design documentation structure
        doc_structure = self._design_doc_structure(paper, output_level)
        structure['directories']['docs'].update(doc_structure)
        
        # Root level files
        structure['files'] = self._design_root_files(paper, tech_stack, output_level)
        
        return structure
    
    def _design_source_structure(self, paper: Paper, tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Design source code directory structure"""
        source_structure = {
            'main_file': f"main.{tech_stack['primary_language'].value}",
            'modules': [],
            'utils': {},
            'algorithms': {}
        }
        
        # Create modules for each algorithm
        for algorithm in paper.algorithms:
            module_name = algorithm.name.lower().replace(' ', '_')
            source_structure['algorithms'][module_name] = {
                'type': 'algorithm_module',
                'description': f"Implementation of {algorithm.name}",
                'files': [f"{module_name}.py"]
            }
        
        # Utility modules
        source_structure['utils'] = {
            'data_processing': {
                'type': 'utility_module',
                'description': 'Data processing utilities',
                'files': ['data_processor.py']
            },
            'visualization': {
                'type': 'utility_module',
                'description': 'Visualization utilities',
                'files': ['visualizer.py']
            },
            'evaluation': {
                'type': 'utility_module',
                'description': 'Evaluation metrics and utilities',
                'files': ['evaluator.py']
            }
        }
        
        return source_structure
    
    def _design_test_structure(self, paper: Paper, tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Design test directory structure"""
        test_structure = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {}
        }
        
        # Unit tests for algorithms
        for algorithm in paper.algorithms:
            module_name = algorithm.name.lower().replace(' ', '_')
            test_structure['unit_tests'][f"test_{module_name}"] = {
                'type': 'unit_test',
                'description': f"Unit tests for {algorithm.name}",
                'files': [f"test_{module_name}.py"]
            }
        
        # Integration tests
        test_structure['integration_tests']['test_integration'] = {
            'type': 'integration_test',
            'description': 'Integration tests for the entire system',
            'files': ['test_integration.py']
        }
        
        # Performance tests
        test_structure['performance_tests']['test_performance'] = {
            'type': 'performance_test',
            'description': 'Performance benchmarks',
            'files': ['test_performance.py']
        }
        
        return test_structure
    
    def _design_doc_structure(self, paper: Paper, output_level: OutputLevel) -> Dict[str, Any]:
        """Design documentation directory structure"""
        doc_structure = {
            'api': {
                'type': 'api_documentation',
                'description': 'API documentation',
                'files': ['api.md']
            },
            'user_guide': {
                'type': 'user_guide',
                'description': 'User guide and tutorials',
                'files': ['user_guide.md']
            },
            'paper_summary': {
                'type': 'paper_summary',
                'description': 'Summary of the research paper',
                'files': ['paper_summary.md']
            }
        }
        
        if output_level == OutputLevel.PRODUCTION:
            doc_structure['developer_guide'] = {
                'type': 'developer_guide',
                'description': 'Developer documentation and contribution guidelines',
                'files': ['developer_guide.md']
            }
        
        return doc_structure
    
    def _design_root_files(self, paper: Paper, tech_stack: Dict[str, Any], 
                          output_level: OutputLevel) -> Dict[str, Any]:
        """Design root level files"""
        files = {
            'readme': {
                'type': 'readme',
                'description': 'Project README',
                'filename': 'README.md'
            },
            'requirements': {
                'type': 'requirements',
                'description': 'Python dependencies',
                'filename': 'requirements.txt'
            },
            'gitignore': {
                'type': 'gitignore',
                'description': 'Git ignore file',
                'filename': '.gitignore'
            }
        }
        
        if output_level == OutputLevel.PRODUCTION:
            files['setup'] = {
                'type': 'setup',
                'description': 'Package setup configuration',
                'filename': 'setup.py'
            }
            
            files['license'] = {
                'type': 'license',
                'description': 'MIT License',
                'filename': 'LICENSE'
            }
        
        return files
    
    def generate_framework_recommendations(self, paper: Paper, 
                                         research_patterns: Dict[str, Any]) -> List[str]:
        """Generate framework recommendations based on research"""
        recommendations = []
        
        # Analyze what successful implementations used
        if research_patterns.get('programming_languages'):
            most_common_lang = max(
                research_patterns['programming_languages'].items(),
                key=lambda x: x[1]
            )[0]
            
            if most_common_lang.lower() == 'python':
                recommendations.append("Python is the most common language for this type of research")
                
                if research_patterns.get('dependencies'):
                    top_deps = sorted(
                        research_patterns['dependencies'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    for dep, count in top_deps:
                        recommendations.append(f"Popular dependency: {dep} (used in {count} repositories)")
        
        return recommendations
    
    def run(self, paper: Paper, research_patterns: Dict[str, Any], 
            user_preference: Optional[str] = None) -> ArchitectureResult:
        """Main method to design architecture"""
        logger.info(f"Designing architecture for: {paper.metadata.title}")
        
        try:
            # Determine output level
            output_level = self.determine_output_level(user_preference)
            
            # Analyze research domain
            domain_analysis = self.analyze_research_domain(paper)
            
            # Recommend technology stack
            tech_stack = self.recommend_technology_stack(paper, research_patterns)
            
            # Design project structure
            project_structure = self.design_project_structure(paper, tech_stack, output_level)
            
            # Generate framework recommendations
            framework_recommendations = self.generate_framework_recommendations(paper, research_patterns)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(tech_stack)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(paper, domain_analysis, tech_stack, output_level)
            
            # Calculate architecture confidence
            architecture_confidence = self._calculate_architecture_confidence(
                paper, research_patterns, tech_stack
            )
            
            return ArchitectureResult(
                project_structure=project_structure,
                technology_stack=tech_stack,
                output_level=output_level,
                framework_recommendations=framework_recommendations,
                dependencies=dependencies,
                architecture_confidence=architecture_confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in architecture design: {e}")
            raise
    
    def _extract_dependencies(self, tech_stack: Dict[str, Any]) -> List[str]:
        """Extract list of dependencies from tech stack"""
        dependencies = []
        
        # Add frameworks
        for framework in tech_stack.get('frameworks', []):
            dependencies.append(framework.value)
        
        # Add libraries
        for lib in tech_stack.get('libraries', []):
            dependencies.append(lib)
        
        # Add testing frameworks
        for test_framework in tech_stack.get('testing_frameworks', []):
            dependencies.append(test_framework)
        
        return dependencies
    
    def _generate_reasoning(self, paper: Paper, domain_analysis: Dict[str, Any], 
                           tech_stack: Dict[str, Any], output_level: OutputLevel) -> str:
        """Generate reasoning for architecture decisions"""
        reasoning = f"""
Architecture Design Reasoning for: {paper.metadata.title}

1. Research Domain Analysis:
   - Domain: {domain_analysis['domain']}
   - Technical Requirements: {', '.join(domain_analysis['technical_requirements'])}
   - Performance Requirements: {', '.join(domain_analysis['performance_requirements'])}

2. Technology Stack Selection:
   - Primary Language: {tech_stack['primary_language'].value}
   - Frameworks: {', '.join([f.value for f in tech_stack.get('frameworks', [])])}
   - Libraries: {', '.join(tech_stack.get('libraries', []))}

3. Project Structure:
   - Output Level: {output_level.value}
   - Core Directories: src, tests, docs
   - Additional Directories: Based on output level requirements

4. Architecture Decisions:
   - Modular design with separate algorithm implementations
   - Comprehensive testing structure
   - Documentation focused on user needs
   - Scalable structure for future enhancements
"""
        return reasoning
    
    def _calculate_architecture_confidence(self, paper: Paper, 
                                         research_patterns: Dict[str, Any], 
                                         tech_stack: Dict[str, Any]) -> float:
        """Calculate confidence score for architecture design"""
        score = 0.0
        
        # Paper has clear domain
        if paper.metadata.domain:
            score += 0.2
        
        # Research patterns available
        if research_patterns:
            score += 0.3
        
        # Technology stack is well-defined
        if tech_stack['primary_language']:
            score += 0.2
        
        # Algorithms identified
        if paper.algorithms:
            score += 0.2
        
        # Output level determined
        if tech_stack.get('output_level'):
            score += 0.1
        
        return min(1.0, score)