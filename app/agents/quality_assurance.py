"""
Quality Assurance Agent

This agent validates code correctness, performance, and quality
to ensure the generated implementation matches the original paper specifications.
"""

import logging
import ast
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from pathlib import Path

from agno import agent
from agno.models.openrouter import OpenRouter

from ..models.paper import Paper, Algorithm, Experiment
from ..models.code import CodeImplementation, CodeFile

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from validation"""
    is_valid: bool
    validation_score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    performance_metrics: Dict[str, Any]
    code_quality_metrics: Dict[str, Any]


class QualityAssuranceAgent:
    """Agent for validating code quality and correctness"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.llm = OpenRouter(model=model)
        
        # Create agent for quality assurance
        self.agent = agent(
            name="quality_assurance",
            model=self.llm,
            description="Validate code quality and correctness for scientific paper implementations"
        )
    
    def validate_code_implementation(self, paper: Paper, code_implementation: CodeImplementation,
                                   architecture_result: Dict[str, Any]) -> ValidationResult:
        """Main validation method for code implementation"""
        start_time = datetime.now()
        
        logger.info(f"Validating code implementation for: {paper.metadata.title}")
        
        try:
            issues = []
            warnings = []
            recommendations = []
            
            # 1. Validate algorithm implementations
            algorithm_validation = self._validate_algorithms(paper, code_implementation)
            issues.extend(algorithm_validation['issues'])
            warnings.extend(algorithm_validation['warnings'])
            recommendations.extend(algorithm_validation['recommendations'])
            
            # 2. Validate code structure and quality
            structure_validation = self._validate_code_structure(code_implementation)
            issues.extend(structure_validation['issues'])
            warnings.extend(structure_validation['warnings'])
            recommendations.extend(structure_validation['recommendations'])
            
            # 3. Validate dependencies and imports
            dependency_validation = self._validate_dependencies(code_implementation)
            issues.extend(dependency_validation['issues'])
            warnings.extend(dependency_validation['warnings'])
            recommendations.extend(dependency_validation['recommendations'])
            
            # 4. Validate documentation quality
            doc_validation = self._validate_documentation(code_implementation)
            issues.extend(doc_validation['issues'])
            warnings.extend(doc_validation['warnings'])
            recommendations.extend(doc_validation['recommendations'])
            
            # 5. Validate performance characteristics
            performance_validation = self._validate_performance(code_implementation)
            issues.extend(performance_validation['issues'])
            warnings.extend(performance_validation['warnings'])
            recommendations.extend(performance_validation['recommendations'])
            
            # 6. Validate test coverage
            test_validation = self._validate_tests(code_implementation)
            issues.extend(test_validation['issues'])
            warnings.extend(test_validation['warnings'])
            recommendations.extend(test_validation['recommendations'])
            
            # 7. Validate against paper specifications
            paper_validation = self._validate_against_paper(paper, code_implementation)
            issues.extend(paper_validation['issues'])
            warnings.extend(paper_validation['warnings'])
            recommendations.extend(paper_validation['recommendations'])
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(
                issues, warnings, recommendations, paper, code_implementation
            )
            
            # Generate performance metrics
            performance_metrics = self._generate_performance_metrics(
                code_implementation, performance_validation
            )
            
            # Generate code quality metrics
            code_quality_metrics = self._generate_code_quality_metrics(
                code_implementation, structure_validation
            )
            
            return ValidationResult(
                is_valid=validation_score >= 0.7,  # 70% threshold for validity
                validation_score=validation_score,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                performance_metrics=performance_metrics,
                code_quality_metrics=code_quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in code validation: {e}")
            raise
    
    def _validate_algorithms(self, paper: Paper, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate algorithm implementations"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check if all paper algorithms have implementations
        implemented_algorithms = []
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                # Extract algorithm name from filename
                algo_name = file.name.replace('.py', '').replace('_', ' ').title()
                implemented_algorithms.append(algo_name)
        
        # Check for missing algorithms
        for paper_algo in paper.algorithms:
            algo_found = False
            for impl_algo in implemented_algorithms:
                if paper_algo.name.lower() in impl_algo.lower():
                    algo_found = True
                    break
            
            if not algo_found:
                issues.append(f"Missing implementation for algorithm: {paper_algo.name}")
                recommendations.append(f"Implement {paper_algo.name} algorithm as described in the paper")
            else:
                # Validate algorithm implementation
                algo_validation = self._validate_single_algorithm(paper_algo, code_implementation)
                issues.extend(algo_validation['issues'])
                warnings.extend(algo_validation['warnings'])
                recommendations.extend(algo_validation['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_single_algorithm(self, algorithm: Algorithm, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate a single algorithm implementation"""
        issues = []
        warnings = []
        recommendations = []
        
        # Find the algorithm file
        algorithm_file = None
        for file in code_implementation.files:
            if file.file_type == "algorithm" and algorithm.name.lower() in file.name.lower():
                algorithm_file = file
                break
        
        if not algorithm_file:
            issues.append(f"Algorithm file not found: {algorithm.name}")
            return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
        
        try:
            # Parse the algorithm file
            tree = ast.parse(algorithm_file.content)
            
            # Check for required methods
            required_methods = ['__init__', 'train', 'predict']
            found_methods = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_methods.append(node.name)
            
            for method in required_methods:
                if method not in found_methods:
                    issues.append(f"Missing required method: {method}")
                    recommendations.append(f"Implement {method} method for {algorithm.name}")
            
            # Check for proper parameter handling
            if '__init__' in found_methods:
                init_method = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                        init_method = node
                        break
                
                if init_method:
                    # Check for parameters dictionary
                    has_params_dict = False
                    for node in ast.walk(init_method):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == 'parameters':
                                    has_params_dict = True
                                    break
                    
                    if not has_params_dict:
                        warnings.append(f"Algorithm {algorithm.name} doesn't use parameters dictionary")
                        recommendations.append("Consider using a parameters dictionary for better configuration management")
            
            # Check for proper error handling
            has_error_handling = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Raise) or isinstance(node, ast.Try):
                    has_error_handling = True
                    break
            
            if not has_error_handling:
                warnings.append(f"Algorithm {algorithm.name} lacks error handling")
                recommendations.append("Add proper error handling for robust implementation")
            
            # Check for documentation
            has_docstrings = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
                    has_docstrings = True
                    break
            
            if not has_docstrings:
                warnings.append(f"Algorithm {algorithm.name} lacks documentation")
                recommendations.append("Add comprehensive docstrings for better code maintainability")
            
        except SyntaxError as e:
            issues.append(f"Syntax error in algorithm file {algorithm_file.name}: {e}")
        except Exception as e:
            issues.append(f"Error validating algorithm {algorithm.name}: {e}")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_code_structure(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate code structure and organization"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for proper file organization
        file_types = {}
        for file in code_implementation.files:
            file_type = file.file_type
            if file_type not in file_types:
                file_types[file_type] = []
            file_types[file_type].append(file.name)
        
        # Check for essential file types
        essential_files = ['main', 'algorithm', 'requirements']
        for essential in essential_files:
            if essential not in file_types:
                issues.append(f"Missing essential file type: {essential}")
                recommendations.append(f"Add {essential} file for complete implementation")
        
        # Check for proper naming conventions
        for file in code_implementation.files:
            if not self._is_valid_filename(file.name):
                warnings.append(f"Invalid filename format: {file.name}")
                recommendations.append("Use snake_case for filenames with descriptive names")
        
        # Check for proper imports
        import_issues = self._validate_imports(code_implementation)
        issues.extend(import_issues['issues'])
        warnings.extend(import_issues['warnings'])
        recommendations.extend(import_issues['recommendations'])
        
        # Check for code duplication
        duplication_issues = self._check_code_duplication(code_implementation)
        issues.extend(duplication_issues['issues'])
        warnings.extend(duplication_issues['warnings'])
        recommendations.extend(duplication_issues['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_dependencies(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate dependencies and imports"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for requirements file
        requirements_file = None
        for file in code_implementation.files:
            if file.file_type == "requirements" and file.name == "requirements.txt":
                requirements_file = file
                break
        
        if not requirements_file:
            issues.append("Missing requirements.txt file")
            recommendations.append("Add requirements.txt with all necessary dependencies")
        else:
            # Parse requirements file
            try:
                requirements = []
                for line in requirements_file.content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        requirements.append(line)
                
                # Check for common dependencies
                common_deps = ['numpy', 'pandas', 'matplotlib', 'scikit-learn']
                for dep in common_deps:
                    found = any(dep in req for req in requirements)
                    if not found:
                        warnings.append(f"Common dependency missing: {dep}")
                        recommendations.append(f"Consider adding {dep} for data processing and visualization")
                
                # Check for version specifications
                has_versions = any('==' in req or '>=' in req for req in requirements)
                if not has_versions:
                    warnings.append("Requirements file lacks version specifications")
                    recommendations.append("Add version specifications for reproducible builds")
                
            except Exception as e:
                issues.append(f"Error parsing requirements file: {e}")
        
        # Check for import consistency
        import_consistency = self._check_import_consistency(code_implementation)
        issues.extend(import_consistency['issues'])
        warnings.extend(import_consistency['warnings'])
        recommendations.extend(import_consistency['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_documentation(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate documentation quality"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for README
        readme_file = None
        for file in code_implementation.files:
            if file.file_type == "readme":
                readme_file = file
                break
        
        if not readme_file:
            issues.append("Missing README.md file")
            recommendations.append("Add comprehensive README.md with usage instructions")
        else:
            # Check README content
            readme_content = readme_file.content.lower()
            
            # Check for essential sections
            essential_sections = ['installation', 'usage', 'examples', 'citation']
            for section in essential_sections:
                if section not in readme_content:
                    warnings.append(f"README missing section: {section}")
                    recommendations.append(f"Add {section} section to README")
            
            # Check for code examples
            if '```' not in readme_content:
                warnings.append("README lacks code examples")
                recommendations.append("Add code examples in README for quick start")
        
        # Check for API documentation
        api_doc = None
        for file in code_implementation.files:
            if file.file_type == "api":
                api_doc = file
                break
        
        if not api_doc:
            warnings.append("Missing API documentation")
            recommendations.append("Add API documentation for detailed function reference")
        
        # Check for docstrings in code
        docstring_issues = self._check_docstrings(code_implementation)
        issues.extend(docstring_issues['issues'])
        warnings.extend(docstring_issues['warnings'])
        recommendations.extend(docstring_issues['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_performance(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate performance characteristics"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for performance optimizations
        perf_issues = self._check_performance_optimizations(code_implementation)
        issues.extend(perf_issues['issues'])
        warnings.extend(perf_issues['warnings'])
        recommendations.extend(perf_issues['recommendations'])
        
        # Check for memory efficiency
        memory_issues = self._check_memory_efficiency(code_implementation)
        issues.extend(memory_issues['issues'])
        warnings.extend(memory_issues['warnings'])
        recommendations.extend(memory_issues['recommendations'])
        
        # Check for computational complexity
        complexity_issues = self._check computational_complexity(code_implementation)
        issues.extend(complexity_issues['issues'])
        warnings.extend(complexity_issues['warnings'])
        recommendations.extend(complexity_issues['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_tests(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate test coverage and quality"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for test files
        test_files = [f for f in code_implementation.files if 'test' in f.name.lower()]
        
        if not test_files:
            issues.append("Missing test files")
            recommendations.append("Add comprehensive test suite for all algorithms")
        else:
            # Check test coverage
            coverage_issues = self._check_test_coverage(code_implementation, test_files)
            issues.extend(coverage_issues['issues'])
            warnings.extend(coverage_issues['warnings'])
            recommendations.extend(coverage_issues['recommendations'])
        
        # Check for test framework usage
        framework_usage = self._check_test_framework_usage(code_implementation)
        issues.extend(framework_usage['issues'])
        warnings.extend(framework_usage['warnings'])
        recommendations.extend(framework_usage['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _validate_against_paper(self, paper: Paper, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate implementation against paper specifications"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check if implementation matches paper domain
        domain_match = self._check_domain_match(paper, code_implementation)
        issues.extend(domain_match['issues'])
        warnings.extend(domain_match['warnings'])
        recommendations.extend(domain_match['recommendations'])
        
        # Check if algorithms match paper descriptions
        algorithm_match = self._check_algorithm_match(paper, code_implementation)
        issues.extend(algorithm_match['issues'])
        warnings.extend(algorithm_match['warnings'])
        recommendations.extend(algorithm_match['recommendations'])
        
        # Check if experiments are reproducible
        reproducibility = self._check_reproducibility(paper, code_implementation)
        issues.extend(reproducibility['issues'])
        warnings.extend(reproducibility['warnings'])
        recommendations.extend(reproducibility['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _calculate_validation_score(self, issues: List[str], warnings: List[str], 
                                  recommendations: List[str], paper: Paper, 
                                  code_implementation: CodeImplementation) -> float:
        """Calculate overall validation score"""
        score = 1.0
        
        # Deduct points for issues (critical problems)
        score -= len(issues) * 0.1
        
        # Deduct points for warnings (minor problems)
        score -= len(warnings) * 0.05
        
        # Bonus points for recommendations (good practices)
        score += min(len(recommendations) * 0.02, 0.1)
        
        # Check algorithm coverage
        algorithm_coverage = self._calculate_algorithm_coverage(paper, code_implementation)
        score *= algorithm_coverage
        
        # Check documentation coverage
        doc_coverage = self._calculate_documentation_coverage(code_implementation)
        score *= doc_coverage
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _generate_performance_metrics(self, code_implementation: CodeImplementation,
                                    performance_validation: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate performance metrics"""
        metrics = {
            'total_files': len(code_implementation.files),
            'algorithm_files': len([f for f in code_implementation.files if f.file_type == "algorithm"]),
            'utility_files': len([f for f in code_implementation.files if f.file_type == "utility"]),
            'documentation_files': len([f for f in code_implementation.files if f.file_type in ["readme", "api", "user_guide", "paper_summary", "developer_guide"]]),
            'notebook_files': len([f for f in code_implementation.files if f.file_type == "notebook"]),
            'performance_warnings': len(performance_validation['warnings']),
            'performance_issues': len(performance_validation['issues']),
            'performance_recommendations': len(performance_validation['recommendations'])
        }
        
        return metrics
    
    def _generate_code_quality_metrics(self, code_implementation: CodeImplementation,
                                      structure_validation: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate code quality metrics"""
        metrics = {
            'total_lines': self._count_total_lines(code_implementation),
            'average_file_size': self._calculate_average_file_size(code_implementation),
            'complexity_score': self._calculate_complexity_score(code_implementation),
            'documentation_score': self._calculate_documentation_score(code_implementation),
            'structure_warnings': len(structure_validation['warnings']),
            'structure_issues': len(structure_validation['issues']),
            'structure_recommendations': len(structure_validation['recommendations'])
        }
        
        return metrics
    
    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename follows proper conventions"""
        # Check for valid characters
        valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-')
        if not all(c in valid_chars for c in filename):
            return False
        
        # Check for snake_case or PascalCase
        if '.' in filename:
            name_part = filename.split('.')[0]
            if not (name_part.islower() or name_part.istitle() or '_' in name_part):
                return False
        
        return True
    
    def _validate_imports(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Validate import statements"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for circular imports
        circular_imports = self._check_circular_imports(code_implementation)
        issues.extend(circular_imports['issues'])
        warnings.extend(circular_imports['warnings'])
        recommendations.extend(circular_imports['recommendations'])
        
        # Check for unused imports
        unused_imports = self._check_unused_imports(code_implementation)
        issues.extend(unused_imports['issues'])
        warnings.extend(unused_imports['warnings'])
        recommendations.extend(unused_imports['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_circular_imports(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for circular imports"""
        issues = []
        warnings = []
        recommendations = []
        
        # This is a simplified check - in practice, you'd need a more sophisticated analysis
        # For now, we'll just check for obvious patterns
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                # Check if algorithm files import each other
                # This is a simplified check
                pass
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_unused_imports(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for unused imports"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require static analysis to detect unused imports
        # For now, we'll just add a general warning
        warnings.append("Consider running static analysis to detect unused imports")
        recommendations.append("Use tools like flake8 or pylint to identify and remove unused imports")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_code_duplication(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for code duplication"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require code analysis tools
        # For now, we'll just add a general warning
        warnings.append("Consider checking for code duplication")
        recommendations.append("Use tools like jplag or duplication to identify duplicate code")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_import_consistency(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check import consistency across files"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for inconsistent import styles
        import_styles = {'from x import y': 0, 'import x': 0}
        
        for file in code_implementation.files:
            if file.language.value == 'python':
                for line in file.content.split('\n'):
                    line = line.strip()
                    if line.startswith('from ') and ' import ' in line:
                        import_styles['from x import y'] += 1
                    elif line.startswith('import '):
                        import_styles['import x'] += 1
        
        # Check for mixed import styles
        if import_styles['from x import y'] > 0 and import_styles['import x'] > 0:
            warnings.append("Mixed import styles detected")
            recommendations.append("Use consistent import style throughout the codebase")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_docstrings(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for docstrings in code"""
        issues = []
        warnings = []
        recommendations = []
        
        docstring_count = 0
        total_functions = 0
        
        for file in code_implementation.files:
            if file.language.value == 'python':
                try:
                    tree = ast.parse(file.content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                docstring_count += 1
                except SyntaxError:
                    continue
        
        if total_functions > 0:
            docstring_coverage = docstring_count / total_functions
            if docstring_coverage < 0.5:
                warnings.append(f"Low docstring coverage: {docstring_coverage:.1%}")
                recommendations.append("Add docstrings to at least 50% of functions and classes")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_performance_optimizations(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for performance optimizations"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for vectorization opportunities
        vectorization_issues = self._check_vectorization(code_implementation)
        issues.extend(vectorization_issues['issues'])
        warnings.extend(vectorization_issues['warnings'])
        recommendations.extend(vectorization_issues['recommendations'])
        
        # Check for caching opportunities
        caching_issues = self._check_caching_opportunities(code_implementation)
        issues.extend(caching_issues['issues'])
        warnings.extend(caching_issues['warnings'])
        recommendations.extend(caching_issues['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_vectorization(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for vectorization opportunities"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require static analysis to detect loops that could be vectorized
        # For now, we'll just add a general recommendation
        recommendations.append("Consider using NumPy vectorization for numerical computations")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_caching_opportunities(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for caching opportunities"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require static analysis to detect repeated computations
        # For now, we'll just add a general recommendation
        recommendations.append("Consider adding caching for expensive computations")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_memory_efficiency(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for memory efficiency issues"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for potential memory leaks
        memory_issues = self._check_memory_leaks(code_implementation)
        issues.extend(memory_issues['issues'])
        warnings.extend(memory_issues['warnings'])
        recommendations.extend(memory_issues['recommendations'])
        
        # Check for large data structures
        data_structure_issues = self._check_large_data_structures(code_implementation)
        issues.extend(data_structure_issues['issues'])
        warnings.extend(data_structure_issues['warnings'])
        recommendations.extend(data_structure_issues['recommendations'])
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_memory_leaks(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for potential memory leaks"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require static analysis to detect memory leaks
        # For now, we'll just add a general warning
        warnings.append("Consider checking for memory leaks in long-running processes")
        recommendations.append("Use memory profiling tools to identify and fix memory leaks")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_large_data_structures(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check for large data structures"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for large lists or arrays
        for file in code_implementation.files:
            if file.language.value == 'python':
                lines = file.content.split('\n')
                for i, line in enumerate(lines, 1):
                    if 'list(' in line or 'array(' in line or 'np.' in line:
                        # This is a simplified check
                        pass
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check computational_complexity(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check computational complexity"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require static analysis to detect high-complexity algorithms
        # For now, we'll just add a general recommendation
        recommendations.append("Consider computational complexity for large datasets")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_test_coverage(self, code_implementation: CodeImplementation, test_files: List[CodeFile]) -> Dict[str, List[str]]:
        """Check test coverage"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check if there are tests for all algorithms
        algorithm_files = [f for f in code_implementation.files if f.file_type == "algorithm"]
        test_coverage = len(test_files) / len(algorithm_files) if algorithm_files else 0
        
        if test_coverage < 0.5:
            warnings.append(f"Low test coverage: {test_coverage:.1%}")
            recommendations.append("Add tests for at least 50% of algorithms")
        
        # Check test quality
        for test_file in test_files:
            if 'test_' not in test_file.name:
                warnings.append(f"Non-standard test filename: {test_file.name}")
                recommendations.append("Use test_ prefix for test files")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_test_framework_usage(self, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check test framework usage"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for proper test framework usage
        has_unittest = False
        has_pytest = False
        
        for file in code_implementation.files:
            if 'test' in file.name.lower():
                content = file.content.lower()
                if 'unittest' in content:
                    has_unittest = True
                if 'pytest' in content:
                    has_pytest = True
        
        if not has_unittest and not has_pytest:
            warnings.append("No test framework detected")
            recommendations.append("Use unittest or pytest for testing")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_domain_match(self, paper: Paper, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check if implementation matches paper domain"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require domain-specific validation
        # For now, we'll just add a general check
        domain = paper.metadata.domain.lower()
        
        # Check for domain-specific dependencies
        domain_specific_deps = {
            'machine learning': ['scikit-learn', 'tensorflow', 'pytorch'],
            'deep learning': ['tensorflow', 'pytorch', 'keras'],
            'computer vision': ['opencv', 'pillow', 'scikit-image'],
            'natural language processing': ['nltk', 'spacy', 'transformers'],
            'data science': ['pandas', 'numpy', 'matplotlib']
        }
        
        for file in code_implementation.files:
            if file.file_type == "requirements":
                content = file.content.lower()
                for dep_type, deps in domain_specific_deps.items():
                    if dep_type in domain:
                        for dep in deps:
                            if dep not in content:
                                warnings.append(f"Missing domain-specific dependency: {dep}")
                                recommendations.append(f"Consider adding {dep} for {dep_type} tasks")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_algorithm_match(self, paper: Paper, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check if algorithms match paper descriptions"""
        issues = []
        warnings = []
        recommendations = []
        
        # This would require algorithm-specific validation
        # For now, we'll just check if all algorithms are implemented
        implemented_algorithms = []
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                algo_name = file.name.replace('.py', '').replace('_', ' ').title()
                implemented_algorithms.append(algo_name)
        
        for paper_algo in paper.algorithms:
            algo_found = False
            for impl_algo in implemented_algorithms:
                if paper_algo.name.lower() in impl_algo.lower():
                    algo_found = True
                    break
            
            if not algo_found:
                issues.append(f"Algorithm not implemented: {paper_algo.name}")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _check_reproducibility(self, paper: Paper, code_implementation: CodeImplementation) -> Dict[str, List[str]]:
        """Check if experiments are reproducible"""
        issues = []
        warnings = []
        recommendations = []
        
        # Check for random seed setting
        has_random_seed = False
        for file in code_implementation.files:
            if file.language.value == 'python':
                content = file.content.lower()
                if 'random.seed' in content or 'np.random.seed' in content:
                    has_random_seed = True
                    break
        
        if not has_random_seed:
            warnings.append("No random seed setting found")
            recommendations.append("Set random seeds for reproducible results")
        
        # Check for version pinning
        version_pinning = False
        for file in code_implementation.files:
            if file.file_type == "requirements":
                content = file.content
                if '==' in content or '>=' in content:
                    version_pinning = True
                    break
        
        if not version_pinning:
            warnings.append("No version pinning in requirements")
            recommendations.append("Pin dependency versions for reproducible builds")
        
        return {'issues': issues, 'warnings': warnings, 'recommendations': recommendations}
    
    def _calculate_algorithm_coverage(self, paper: Paper, code_implementation: CodeImplementation) -> float:
        """Calculate algorithm coverage score"""
        if not paper.algorithms:
            return 1.0
        
        implemented_count = 0
        for paper_algo in paper.algorithms:
            for file in code_implementation.files:
                if file.file_type == "algorithm" and paper_algo.name.lower() in file.name.lower():
                    implemented_count += 1
                    break
        
        return implemented_count / len(paper.algorithms)
    
    def _calculate_documentation_coverage(self, code_implementation: CodeImplementation) -> float:
        """Calculate documentation coverage score"""
        doc_types = ['readme', 'api', 'user_guide', 'paper_summary', 'developer_guide']
        implemented_count = 0
        
        for doc_type in doc_types:
            for file in code_implementation.files:
                if file.file_type == doc_type:
                    implemented_count += 1
                    break
        
        return implemented_count / len(doc_types)
    
    def _count_total_lines(self, code_implementation: CodeImplementation) -> int:
        """Count total lines of code"""
        total_lines = 0
        for file in code_implementation.files:
            if file.language.value in ['python', 'javascript']:
                total_lines += len(file.content.split('\n'))
        return total_lines
    
    def _calculate_average_file_size(self, code_implementation: CodeImplementation) -> float:
        """Calculate average file size in lines"""
        if not code_implementation.files:
            return 0.0
        
        total_lines = self._count_total_lines(code_implementation)
        return total_lines / len(code_implementation.files)
    
    def _calculate_complexity_score(self, code_implementation: CodeImplementation) -> float:
        """Calculate code complexity score"""
        # This would require static analysis tools
        # For now, return a placeholder score
        return 0.7
    
    def _calculate_documentation_score(self, code_implementation: CodeImplementation) -> float:
        """Calculate documentation score"""
        doc_files = [f for f in code_implementation.files if f.file_type in ['readme', 'api', 'user_guide', 'paper_summary', 'developer_guide']]
        
        if not doc_files:
            return 0.0
        
        total_lines = 0
        for file in doc_files:
            total_lines += len(file.content.split('\n'))
        
        return min(total_lines / 1000, 1.0)  # Normalize to 0-1 range
    
    def generate_validation_report(self, validation_result: ValidationResult, 
                                 output_path: str) -> None:
        """Generate comprehensive validation report"""
        report = {
            'validation_summary': {
                'is_valid': validation_result.is_valid,
                'validation_score': validation_result.validation_score,
                'total_issues': len(validation_result.issues),
                'total_warnings': len(validation_result.warnings),
                'total_recommendations': len(validation_result.recommendations)
            },
            'issues': validation_result.issues,
            'warnings': validation_result.warnings,
            'recommendations': validation_result.recommendations,
            'performance_metrics': validation_result.performance_metrics,
            'code_quality_metrics': validation_result.code_quality_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report to file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def run(self, paper: Paper, code_implementation: CodeImplementation,
            architecture_result: Dict[str, Any]) -> ValidationResult:
        """Main method to validate code implementation"""
        return self.validate_code_implementation(paper, code_implementation, architecture_result)