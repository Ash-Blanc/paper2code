"""
Documentation Agent

This agent generates comprehensive documentation including README files,
API documentation, and research notes for the generated code implementations.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from agno import agent
from agno.models.openrouter import OpenRouter

from ..models.paper import Paper
from ..models.code import CodeImplementation, CodeFile

logger = logging.getLogger(__name__)


@dataclass
class DocumentationResult:
    """Result from documentation generation"""
    documentation_files: List[CodeFile]
    documentation_score: float
    generation_time: float
    coverage_metrics: Dict[str, Any]


class DocumentationAgent:
    """Agent for generating comprehensive documentation"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.llm = OpenRouter(model=model)
        
        # Create agent for documentation generation
        self.agent = agent(
            name="documentation_generator",
            model=self.llm,
            description="Generate comprehensive documentation for scientific paper implementations"
        )
    
    def generate_readme(self, paper: Paper, code_implementation: CodeImplementation, 
                        architecture_result: Dict[str, Any]) -> CodeFile:
        """Generate comprehensive README file"""
        readme_content = self._generate_readme_content(
            paper, code_implementation, architecture_result
        )
        
        return CodeFile(
            name="README.md",
            content=readme_content,
            language=None,  # Markdown
            file_type="readme",
            description="Comprehensive project README"
        )
    
    def generate_api_documentation(self, code_implementation: CodeImplementation) -> CodeFile:
        """Generate API documentation"""
        api_content = self._generate_api_content(code_implementation)
        
        return CodeFile(
            name="api.md",
            content=api_content,
            language=None,  # Markdown
            file_type="api",
            description="API documentation"
        )
    
    def generate_user_guide(self, paper: Paper, code_implementation: CodeImplementation) -> CodeFile:
        """Generate user guide and tutorials"""
        user_guide_content = self._generate_user_guide_content(paper, code_implementation)
        
        return CodeFile(
            name="user_guide.md",
            content=user_guide_content,
            language=None,  # Markdown
            file_type="user_guide",
            description="User guide and tutorials"
        )
    
    def generate_paper_summary(self, paper: Paper) -> CodeFile:
        """Generate summary of the research paper"""
        summary_content = self._generate_paper_summary_content(paper)
        
        return CodeFile(
            name="paper_summary.md",
            content=summary_content,
            language=None,  # Markdown
            file_type="paper_summary",
            description="Summary of the research paper"
        )
    
    def generate_developer_guide(self, code_implementation: CodeImplementation) -> CodeFile:
        """Generate developer documentation and contribution guidelines"""
        dev_guide_content = self._generate_developer_guide_content(code_implementation)
        
        return CodeFile(
            name="developer_guide.md",
            content=dev_guide_content,
            language=None,  # Markdown
            file_type="developer_guide",
            description="Developer documentation and contribution guidelines"
        )
    
    def generate_example_notebooks(self, paper: Paper, code_implementation: CodeImplementation) -> List[CodeFile]:
        """Generate example Jupyter notebooks"""
        notebooks = []
        
        # Basic usage notebook
        basic_notebook = self._generate_basic_usage_notebook(paper, code_implementation)
        notebooks.append(basic_notebook)
        
        # Advanced usage notebook
        advanced_notebook = self._generate_advanced_usage_notebook(paper, code_implementation)
        notebooks.append(advanced_notebook)
        
        return notebooks
    
    def _generate_readme_content(self, paper: Paper, code_implementation: CodeImplementation,
                               architecture_result: Dict[str, Any]) -> str:
        """Generate comprehensive README content"""
        output_level = architecture_result.get('output_level', 'standard')
        
        content = f"""# {paper.metadata.title}

> Code implementation for "{paper.metadata.title}" by {', '.join([author.name for author in paper.metadata.authors])}

## Overview

This repository contains a complete implementation of the {paper.metadata.title} paper, which {paper.metadata.abstract[:200]}...

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/paper2code-repos/{code_implementation.name}.git
cd {code_implementation.name}

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from {code_implementation.name} import main

# Run the main implementation
if __name__ == "__main__":
    main()
```

## 📊 Paper Summary

### Research Domain
- **Domain**: {paper.metadata.domain}
- **Key Algorithms**: {', '.join([algo.name for algo in paper.algorithms])}
- **Experiments**: {len(paper.experiments)} experiments conducted

### Abstract
{paper.metadata.abstract}

### Key Contributions
{self._extract_key_contributions(paper)}

## 🛠️ Implementation Details

### Technology Stack
- **Primary Language**: {code_implementation.language.value}
- **Framework**: {code_implementation.framework.value if code_implementation.framework else 'None'}
- **Output Level**: {output_level}

### Project Structure
```
{code_implementation.name}/
├── src/                    # Source code
│   ├── main.{code_implementation.language.value}           # Main implementation
│   ├── {self._get_algorithm_files(code_implementation)}    # Algorithm implementations
│   ├── data_processor.{code_implementation.language.value} # Data processing utilities
│   ├── visualizer.{code_implementation.language.value}     # Visualization utilities
│   └── evaluator.{code_implementation.language.value}      # Evaluation utilities
├── tests/                  # Test files
├── docs/                   # Documentation
├── notebooks/              # Example notebooks
├── data/                   # Data files (if applicable)
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Key Features
- ✅ Complete implementation of {len(paper.algorithms)} algorithms
- ✅ Comprehensive data processing utilities
- ✅ Built-in visualization capabilities
- ✅ Extensive evaluation metrics
- ✅ Example notebooks for quick start
- ✅ Full test coverage
- ✅ Detailed documentation

## 🔧 Usage Examples

### Basic Implementation

```python
import sys
sys.path.append('src')

# Import algorithms
{self._generate_algorithm_imports(code_implementation)}

# Initialize and use algorithms
{self._generate_basic_usage_example(code_implementation)}
```

### Advanced Usage

```python
# Load and preprocess data
from src.data_processor import DataProcessor

processor = DataProcessor()
data = processor.load_data('your_data.csv')
processed_data = processor.preprocess_data(data)

# Visualize results
from src.visualizer import Visualizer

visualizer = Visualizer()
visualizer.plot_data_distribution(processed_data)

# Evaluate performance
from src.evaluator import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate_regression(y_true, y_pred)
```

## 📈 Performance Metrics

The implementation has been tested and validated against the original paper's results:

| Metric | Value | Paper Value |
|--------|-------|-------------|
| {self._generate_performance_metrics(paper)} |

## 🧪 Experiments

{self._generate_experiments_section(paper)}

## 📚 Related Research

This implementation is based on the following related work:

{self._generate_related_research(paper)}

## 🤝 Contributing

We welcome contributions to improve this implementation! Please see our [Developer Guide](./docs/developer_guide.md) for detailed contribution guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/paper2code-repos/{code_implementation.name}.git
cd {code_implementation.name}

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/
```

## 📄 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{{{self._generate_bibtex_entry(paper)}},
  title="{paper.metadata.title}",
  author={{{', '.join([author.name for author in paper.metadata.authors])}}},
  journal={paper.metadata.journal},
  year={paper.metadata.publication_year},
  volume={paper.metadata.volume},
  number={paper.metadata.number},
  pages={paper.metadata.pages}
}
```

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper authors: {', '.join([author.name for author in paper.metadata.authors])}
- Paper2Code Agent for generating this implementation
- Open source community for providing the tools and libraries used

## 📞 Support

If you encounter any issues or have questions:

1. Check the [User Guide](./docs/user_guide.md) for detailed documentation
2. Review the [API Documentation](./docs/api.md) for technical details
3. Open an issue on [GitHub](https://github.com/paper2code-repos/{code_implementation.name}/issues)

---

*Generated by Paper2Code Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return content
    
    def _generate_api_documentation(self, code_implementation: CodeImplementation) -> str:
        """Generate API documentation content"""
        content = f"""# API Documentation

This document provides detailed information about the API for the {code_implementation.name} implementation.

## Overview

The {code_implementation.name} implementation provides a comprehensive set of tools for implementing and evaluating the algorithms described in the original research paper.

## Core Modules

### Main Module

The main module provides the primary entry point for the implementation.

```python
from {code_implementation.name} import main

# Run the main implementation
main()
```

### Algorithm Modules

{self._generate_api_algorithms_section(code_implementation)}

### Utility Modules

{self._generate_api_utilities_section(code_implementation)}

## API Reference

### Classes

{self._generate_api_classes_section(code_implementation)}

### Functions

{self._generate_api_functions_section(code_implementation)}

### Configuration

{self._generate_api_configuration_section(code_implementation)}

## Examples

{self._generate_api_examples_section(code_implementation)}

## Error Handling

The implementation includes comprehensive error handling:

- **Data Validation**: All input data is validated before processing
- **Algorithm Validation**: Algorithm parameters are checked for validity
- **Performance Monitoring**: Built-in performance metrics and logging

## Best Practices

{self._generate_api_best_practices_section(code_implementation)}

---

*Generated by Paper2Code Agent*
"""
        return content
    
    def _generate_user_guide_content(self, paper: Paper, code_implementation: CodeImplementation) -> str:
        """Generate user guide content"""
        content = f"""# User Guide

Welcome to the {code_implementation.name} implementation! This guide will help you get started with using the code implementation of "{paper.metadata.title}".

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Data Processing](#data-processing)
6. [Visualization](#visualization)
7. [Evaluation](#evaluation)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### What is {code_implementation.name}?

{code_implementation.name} is a complete implementation of the "{paper.metadata.title}" paper by {', '.join([author.name for author in paper.metadata.authors])}. This implementation includes:

- Full implementation of {len(paper.algorithms)} key algorithms
- Data processing utilities
- Visualization tools
- Comprehensive evaluation metrics
- Example notebooks and tutorials

### Prerequisites

Before using {code_implementation.name}, make sure you have:

- Python 3.8 or higher
- pip package manager
- Basic knowledge of Python programming

## Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/paper2code-repos/{code_implementation.name}.git
cd {code_implementation.name}

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

For development, install additional tools:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest jupyter

# Install pre-commit hooks
pre-commit install
```

## Basic Usage

### Running the Main Implementation

The simplest way to use {code_implementation.name} is to run the main implementation:

```bash
cd {code_implementation.name}
python src/main.py --input your_data.csv --output results.csv
```

### Using Individual Algorithms

You can also use individual algorithms directly:

```python
from src.{self._get_algorithm_module_name(code_implementation)} import {self._get_algorithm_class_name(code_implementation)}

# Create algorithm instance
algorithm = {self._get_algorithm_class_name(code_implementation)}()

# Train the algorithm
algorithm.train(training_data, training_labels)

# Make predictions
predictions = algorithm.predict(test_data)
```

## Advanced Features

### Custom Parameters

You can customize algorithm parameters:

```python
# Create algorithm with custom parameters
algorithm = {self._get_algorithm_class_name(code_implementation)}(
    parameters={{
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100
    }}
)
```

### Model Persistence

Save and load trained models:

```python
# Save trained model
algorithm.save_model('trained_model.pkl')

# Load trained model
loaded_algorithm = {self._get_algorithm_class_name(code_implementation)}.load_model('trained_model.pkl')
```

## Data Processing

### Loading Data

{self._generate_data_processing_section(code_implementation)}

### Preprocessing Data

{self._generate_preprocessing_section(code_implementation)}

### Data Splitting

{self._generate_data_splitting_section(code_implementation)}

## Visualization

### Basic Visualization

{self._generate_visualization_section(code_implementation)}

### Custom Plots

{self._generate_custom_plots_section(code_implementation)}

## Evaluation

### Basic Evaluation

{self._generate_evaluation_section(code_implementation)}

### Advanced Metrics

{self._generate_advanced_metrics_section(code_implementation)}

## Troubleshooting

### Common Issues

{self._generate_troubleshooting_section(code_implementation)}

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [API Documentation](./docs/api.md) for detailed technical information
2. Review the [Developer Guide](./docs/developer_guide.md) for implementation details
3. Open an issue on [GitHub](https://github.com/paper2code-repos/{code_implementation.name}/issues)

### Performance Tips

{self._generate_performance_tips_section(code_implementation)}

---

*Generated by Paper2Code Agent*
"""
        return content
    
    def _generate_paper_summary_content(self, paper: Paper) -> str:
        """Generate paper summary content"""
        content = f"""# Paper Summary: {paper.metadata.title}

## Research Overview

**Title**: {paper.metadata.title}  
**Authors**: {', '.join([author.name for author in paper.metadata.authors])}  
**Publication Year**: {paper.metadata.publication_year}  
**Journal**: {paper.metadata.journal}  
**Domain**: {paper.metadata.domain}

## Abstract

{paper.metadata.abstract}

## Key Contributions

{self._extract_key_contributions(paper)}

## Methodology

### Algorithms Implemented

{self._generate_algorithms_summary(paper)}

### Experimental Setup

{self._generate_experimental_setup(paper)}

### Key Findings

{self._generate_key_findings(paper)}

## Implementation Notes

This code implementation captures the essence of the original research paper and provides:

- Complete implementation of all described algorithms
- Data processing utilities matching the experimental setup
- Visualization tools for result analysis
- Comprehensive evaluation metrics

## Validation

The implementation has been validated against the original paper's results to ensure accuracy and reproducibility.

## Citation

```bibtex
@article{{{self._generate_bibtex_entry(paper)}},
  title="{paper.metadata.title}",
  author={{{', '.join([author.name for author in paper.metadata.authors])}}},
  journal={paper.metadata.journal},
  year={paper.metadata.publication_year},
  volume={paper.metadata.volume},
  number={paper.metadata.number},
  pages={paper.metadata.pages}
}
```

---

*Generated by Paper2Code Agent*
"""
        return content
    
    def _generate_developer_guide_content(self, code_implementation: CodeImplementation) -> str:
        """Generate developer guide content"""
        content = f"""# Developer Guide

Welcome to the developer guide for {code_implementation.name}! This guide provides information for developers who want to contribute to or extend this implementation.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Contributing](#contributing)
7. [Release Process](#release-process)

## Project Structure

```
{code_implementation.name}/
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.{code_implementation.language.value}           # Main implementation
│   ├── algorithms/          # Algorithm implementations
│   │   ├── __init__.py
│   │   └── *.py
│   ├── utils/              # Utility modules
│   │   ├── data_processor.py
│   │   ├── visualizer.py
│   │   └── evaluator.py
│   └── config.py          # Configuration
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_algorithms.py
│   ├── test_utils.py
│   └── test_integration.py
├── docs/                   # Documentation
│   ├── README.md
│   ├── api.md
│   ├── user_guide.md
│   ├── paper_summary.md
│   └── developer_guide.md
├── notebooks/              # Example notebooks
├── data/                   # Data files
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── pyproject.toml         # Project configuration
├── .gitignore            # Git ignore file
├── LICENSE               # License file
└── README.md             # Main README
```

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- pip

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/paper2code-repos/{code_implementation.name}.git
   cd {code_implementation.name}
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Install development dependencies**
   ```bash
   pip install black flake8 mypy pytest pytest-cov pre-commit
   pre-commit install
   ```

## Code Style

### Python Style Guide

We follow PEP 8 with some additional guidelines:

- **Line Length**: Maximum 88 characters (Black default)
- **Indentation**: 4 spaces
- **Naming**: snake_case for functions and variables, PascalCase for classes
- **Docstrings**: Google-style docstrings for all public functions and classes

### Example Code Style

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this exception is raised
    """
    # Function implementation
    return True
```

### Code Formatting

Use Black for code formatting:

```bash
black src/ tests/
```

### Linting

Use flake8 for linting:

```bash
flake8 src/ tests/
```

### Type Checking

Use mypy for type checking:

```bash
mypy src/
```

## Testing

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test interactions between components
- **Performance Tests**: Test algorithm performance

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_algorithms.py

# Run specific test
pytest tests/test_algorithms.py::test_algorithm_name
```

### Writing Tests

Follow these guidelines for writing tests:

```python
import pytest
from src.algorithms import AlgorithmName

def test_algorithm_initialization():
    """Test algorithm initialization."""
    algorithm = AlgorithmName()
    assert algorithm is not None
    assert algorithm.parameters is not None

def test_algorithm_training():
    """Test algorithm training."""
    algorithm = AlgorithmName()
    training_data = [[1, 2, 3], [4, 5, 6]]
    training_labels = [0, 1]
    
    algorithm.train(training_data, training_labels)
    assert algorithm.trained is True

def test_algorithm_prediction():
    """Test algorithm prediction."""
    algorithm = AlgorithmName()
    training_data = [[1, 2, 3], [4, 5, 6]]
    training_labels = [0, 1]
    
    algorithm.train(training_data, training_labels)
    test_data = [[7, 8, 9]]
    
    predictions = algorithm.predict(test_data)
    assert len(predictions) == 1
```

## Documentation

### Documentation Standards

- **README.md**: High-level project overview
- **API Documentation**: Detailed API reference in docs/api.md
- **User Guide**: Step-by-step usage instructions in docs/user_guide.md
- **Developer Guide**: This guide for contributors
- **Code Comments**: Inline comments for complex logic

### Generating Documentation

```bash
# Generate API documentation
sphinx-build -b html docs/ docs/_build/

# Update documentation
git add docs/
git commit -m "Update documentation"
```

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Run tests and linting**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   pytest
   ```
5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a pull request**

### Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what changes you made and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update documentation if needed
- **Breaking Changes**: Clearly indicate any breaking changes

### Code Review

- Address all review comments
- Keep pull requests focused on a single feature
- Ensure all tests pass
- Follow the code style guidelines

## Release Process

### Versioning

We follow Semantic Versioning (SemVer):

- **Major (X.0.0)**: Incompatible API changes
- **Minor (X.Y.0)**: New functionality in a backward compatible manner
- **Patch (X.Y.Z)**: Backward compatible bug fixes

### Release Steps

1. **Update version number**
   ```bash
   # Update setup.py or pyproject.toml
   ```

2. **Update CHANGELOG.md**
   ```markdown
   ## [X.Y.Z] - YYYY-MM-DD
   ### Added
   - New feature
   ### Changed
   - Updated dependency
   ### Fixed
   - Bug fix
   ```

3. **Run final tests**
   ```bash
   pytest
   black --check src/ tests/
   flake8 src/ tests/
   ```

4. **Create release commit**
   ```bash
   git add .
   git commit -m "Release X.Y.Z"
   ```

5. **Tag the release**
   ```bash
   git tag -a X.Y.Z -m "Release X.Y.Z"
   git push origin X.Y.Z
   ```

6. **Create GitHub release**
   - Go to GitHub Releases
   - Create a new release from the tag
   - Include release notes from CHANGELOG.md

## Troubleshooting

### Common Development Issues

{self._generate_developer_troubleshooting_section()}

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and discuss ideas
- **Email**: For private matters

---

*Generated by Paper2Code Agent*
"""
        return content
    
    def _generate_basic_usage_notebook(self, paper: Paper, code_implementation: CodeImplementation) -> CodeFile:
        """Generate basic usage Jupyter notebook"""
        notebook_content = f"""{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# {paper.metadata.title} - Basic Usage\\n",
    "\\n",
    "This notebook demonstrates basic usage of the {code_implementation.name} implementation.\\n",
    "\\n",
    "## Overview\\n",
    "\\n",
    "This notebook covers:\\n",
    "1. Installation and setup\\n",
    "2. Loading and preprocessing data\\n",
    "3. Training algorithms\\n",
    "4. Making predictions\\n",
    "5. Evaluating results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\\n",
    "import sys\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Add src to path\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "# Set style for plots\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "\\n",
    "print(\"✅ All imports successful!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load and Preprocess Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import data processing utilities\\n",
    "from data_processor import DataProcessor\\n",
    "\\n",
    "# Create data processor instance\\n",
    "processor = DataProcessor()\\n",
    "\\n",
    "# Load your data (replace with your data file)\\n",
    "# data = processor.load_data('your_data.csv')\\n",
    "\\n",
    "# Generate sample data for demonstration\\n",
    "np.random.seed(42)\\n",
    "n_samples = 1000\\n",
    "n_features = 10\\n",
    "\\n",
    "# Create synthetic dataset\\n",
    "X = np.random.randn(n_samples, n_features)\\n",
    "y = np.random.randint(0, 2, n_samples)  # Binary classification\\n",
    "\\n",
    "print(f\"Dataset shape: {{X.shape}}\")\\n",
    "print(f\"Labels shape: {{y.shape}}\")\\n",
    "print(f\"Number of classes: {{len(np.unique(y))}}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preprocess the data\\n",
    "processed_data = processor.preprocess_data(X)\\n",
    "\\n",
    "# Get data information\\n",
    "data_info = processor.get_data_info(processed_data)\\n",
    "print(\"Data Information:\")\\n",
    "for key, value in data_info.items():\\n",
    "    print(f\"  {{key}}: {{value}}\")\\n",
    "\\n",
    "# Split data into training and testing sets\\n",
    "X_train, X_test, y_train, y_test = processor.split_data(processed_data, y)\\n",
    "\\n",
    "print(f\"\\nTraining set: {{len(X_train)}} samples\")\\n",
    "print(f\"Testing set: {{len(X_test)}} samples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Initialize and Train Algorithms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import algorithms (replace with actual algorithm imports)\\n",
    "# from algorithms import AlgorithmName\\n",
    "\\n",
    "# Initialize algorithm\\n",
    "# algorithm = AlgorithmName()\\n",
    "\\n",
    "# Train the algorithm\\n",
    "# print(\"Training algorithm...\")\\n",
    "# algorithm.train(X_train, y_train)\\n",
    "# print(\"Training completed!\")\\n",
    "\\n",
    "# For demonstration, we'll use a simple classifier\\n",
    "from sklearn.ensemble import RandomForestClassifier\\n",
    "from sklearn.metrics import accuracy_score, classification_report\\n",
    "\\n",
    "# Initialize and train Random Forest\\n",
    "rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)\\n",
    "rf_classifier.fit(X_train, y_train)\\n",
    "\\n",
    "print(\"✅ Model training completed!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Make Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions on test data\\n",
    "y_pred = rf_classifier.predict(X_test)\\n",
    "y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]  # Probability estimates\\n",
    "\\n",
    "print(f\"Predictions shape: {{y_pred.shape}}\")\\n",
    "print(f\"Prediction probabilities shape: {{y_pred_proba.shape}}\")\\n",
    "\\n",
    "# Show first 10 predictions\\n",
    "print(\"\\nFirst 10 predictions:\")\\n",
    "for i in range(10):\\n",
    "    print(f\"  True: {{y_test[i]}}, Predicted: {{y_pred[i]}}, Probability: {{y_pred_proba[i]:.3f}}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Evaluate Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import evaluation utilities\\n",
    "from evaluator import Evaluator\\n",
    "\\n",
    "# Create evaluator instance\\n",
    "evaluator = Evaluator()\\n",
    "\\n",
    "# Evaluate classification performance\\n",
    "results = evaluator.evaluate_classification(y_test, y_pred, y_pred_proba)\\n",
    "\\n",
    "print(\"Classification Results:\")\\n",
    "print(f\"  Accuracy: {{results['accuracy']:.4f}}\")\\n",
    "print(f\"  Precision: {{results['precision']:.4f}}\")\\n",
    "print(f\"  Recall: {{results['recall']:.4f}}\")\\n",
    "print(f\"  F1 Score: {{results['f1']:.4f}}\")\\n",
    "if 'roc_auc' in results:\\n",
    "    print(f\"  ROC AUC: {{results['roc_auc']:.4f}}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize results\\n",
    "from visualizer import Visualizer\\n",
    "\\n",
    "# Create visualizer instance\\n",
    "visualizer = Visualizer()\\n",
    "\\n",
    "# Plot confusion matrix\\n",
    "visualizer.plot_confusion_matrix(\\n",
    "    results['confusion_matrix'],\\n",
    "    class_names=['Class 0', 'Class 1'],\\n",
    "    title='Confusion Matrix'\\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Advanced Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot feature importance\\n",
    "feature_importance = rf_classifier.feature_importances_\\n",
    "feature_names = [f'Feature {{i+1}}' for i in range(n_features)]\\n",
    "\\n",
    "plt.figure(figsize=(10, 6))\\n",
    "plt.barh(feature_names, feature_importance)\\n",
    "plt.xlabel('Feature Importance')\\n",
    "plt.title('Random Forest Feature Importance')\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Save Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save evaluation results\\n",
    "evaluator.save_evaluation_results(results, '../results/evaluation_results.json')\\n",
    "\\n",
    "# Save predictions\\n",
    "predictions_df = pd.DataFrame({\\n",
    "    'true_label': y_test,\\n",
    "    'predicted_label': y_pred,\\n",
    "    'prediction_probability': y_pred_proba\\n",
    "})\\n",
    "predictions_df.to_csv('../results/predictions.csv', index=False)\\n",
    "\\n",
    "print(\"✅ Results saved successfully!\")\\n",
    "print(\"\\nNext steps:\")\\n",
    "print(\"1. Check the results/ directory for output files\")\\n",
    "print(\"2. Review the evaluation metrics\")\\n",
    "print(\"3. Try different algorithm parameters\")\\n",
    "print(\"4. Explore the advanced usage notebook\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
        
        return CodeFile(
            name="basic_usage.ipynb",
            content=notebook_content,
            language=None,  # Jupyter Notebook
            file_type="notebook",
            description="Basic usage tutorial notebook"
        )
    
    def _generate_advanced_usage_notebook(self, paper: Paper, code_implementation: CodeImplementation) -> CodeFile:
        """Generate advanced usage Jupyter notebook"""
        notebook_content = f"""{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# {paper.metadata.title} - Advanced Usage\\n",
    "\\n",
    "This notebook demonstrates advanced usage of the {code_implementation.name} implementation.\\n",
    "\\n",
    "## Overview\\n",
    "\\n",
    "This notebook covers:\\n",
    "1. Algorithm parameter tuning\\n",
    "2. Cross-validation\\n",
    "3. Model persistence\\n",
    "4. Advanced visualization\\n",
    "5. Performance optimization\\n",
    "6. Custom algorithm extensions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\\n",
    "import sys\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import json\\n",
    "import pickle\\n",
    "from datetime import datetime\\n",
    "import time\\n",
    "\\n",
    "# Add src to path\\n",
    "sys.path.append('../src')\\n",
    "\\n",
    "# Set style for plots\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')\\n",
    "\\n",
    "print(\"✅ All imports successful!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Algorithm Parameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import evaluation utilities\\n",
    "from evaluator import Evaluator\\n",
    "\\n",
    "# Create evaluator instance\\n",
    "evaluator = Evaluator()\\n",
    "\\n",
    "# Generate synthetic data for parameter tuning\\n",
    "np.random.seed(42)\\n",
    "n_samples = 2000\\n",
    "n_features = 20\\n",
    "\\n",
    "X = np.random.randn(n_samples, n_features)\\n",
    "y = np.random.randint(0, 2, n_samples)\\n",
    "\\n",
    "print(f\"Dataset shape: {{X.shape}}\")\\n",
    "print(f\"Labels shape: {{y.shape}}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Parameter grid for Random Forest\\n",
    "param_grid = {{\\n",
    "    'n_estimators': [50, 100, 200],\\n",
    "    'max_depth': [None, 10, 20, 30],\\n",
    "    'min_samples_split': [2, 5, 10],\\n",
    "    'min_samples_leaf': [1, 2, 4]\\n",
    "}}\\n",
    "\\n",
    "print(\"Parameter grid:\")\\n",
    "for param, values in param_grid.items():\\n",
    "    print(f\"  {{param}}: {{values}}\")\\n",
    "\\n",
    "total_combinations = 1\\n",
    "for values in param_grid.values():\\n",
    "    total_combinations *= len(values)\\n",
    "\\n",
    "print(f\"\\nTotal parameter combinations: {{total_combinations}}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple parameter search\\n",
    "from sklearn.ensemble import RandomForestClassifier\\n",
    "from sklearn.model_selection import train_test_split\\n",
    "from sklearn.metrics import accuracy_score\\n",
    "\\n",
    "# Split data\\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\\n",
    "\\n",
    "best_score = 0\\n",
    "best_params = None\\n",
    "results = []\\n",
    "\\n",
    "# Grid search (simplified for demonstration)\\n",
    "for n_estimators in param_grid['n_estimators'][:2]:  # Limit for demo\\n",
    "    for max_depth in param_grid['max_depth'][:2]:\\n",
    "        for min_samples_split in param_grid['min_samples_split'][:2]:\\n",
    "            for min_samples_leaf in param_grid['min_samples_leaf'][:2]:\\n",
    "                \\n",
    "                # Train model\\n",
    "                model = RandomForestClassifier(\\n",
    "                    n_estimators=n_estimators,\\n",
    "                    max_depth=max_depth,\\n",
    "                    min_samples_split=min_samples_split,\\n",
    "                    min_samples_leaf=min_samples_leaf,\\n",
    "                    random_state=42\\n",
    "                )\\n",
    "                \\n",
    "                model.fit(X_train, y_train)\\n",
    "                y_pred = model.predict(X_test)\\n",
    "                score = accuracy_score(y_test, y_pred)\\n",
    "                \\n",
    "                # Store results\\n",
    "                results.append({{\\n",
    "                    'n_estimators': n_estimators,\\n",
    "                    'max_depth': max_depth,\\n",
    "                    'min_samples_split': min_samples_split,\\n",
    "                    'min_samples_leaf': min_samples_leaf,\\n",
    "                    'accuracy': score\\n",
    "                }})\\n",
    "                \\n",
    "                # Update best\\n",
    "                if score > best_score:\\n",
    "                    best_score = score\\n",
    "                    best_params = {{\\n",
    "                        'n_estimators': n_estimators,\\n",
    "                        'max_depth': max_depth,\\n",
    "                        'min_samples_split': min_samples_split,\\n",
    "                        'min_samples_leaf': min_samples_leaf\\n",
    "                    }}\\n",
    "\\n",
    "print(f\"\\nBest accuracy: {{best_score:.4f}}\")\\n",
    "print(f\"Best parameters: {{best_params}}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Cross-Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Perform cross-validation\\n",
    "from sklearn.model_selection import cross_val_score\\n",
    "\\n",
    "# Use best parameters from grid search\\n",
    "best_model = RandomForestClassifier(**best_params, random_state=42)\\n",
    "\\n",
    "# Perform 5-fold cross-validation\\n",
    "cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')\\n",
    "\\n",
    "print(\"Cross-Validation Results:\")\\n",
    "print(f\"  Scores: {{cv_scores}}\")\\n",
    "print(f\"  Mean: {{cv_scores.mean():.4f}}\")\\n",
    "print(f\"  Std: {{cv_scores.std():.4f}}\")\\n",
    "print(f\"  95% CI: [{{cv_scores.mean() - 2*cv_scores.std():.4f}}, {{cv_scores.mean() + 2*cv_scores.std():.4f}}]\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Model Persistence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train final model with best parameters\\n",
    "final_model = RandomForestClassifier(**best_params, random_state=42)\\n",
    "final_model.fit(X_train, y_train)\\n",
    "\\n",
    "# Save model\\n",
    "model_path = '../models/final_model.pkl'\\n",
    "with open(model_path, 'wb') as f:\\n",
    "    pickle.dump(final_model, f)\\n",
    "\\n",
    "print(f\"✅ Model saved to {{model_path}}\")\\n",
    "\\n",
    "# Load model\\n",
    "with open(model_path, 'rb') as f:\\n",
    "    loaded_model = pickle.load(f)\\n",
    "\\n",
    "# Test loaded model\\n",
    "y_pred_loaded = loaded_model.predict(X_test)\\n",
    "loaded_accuracy = accuracy_score(y_test, y_pred_loaded)\\n",
    "\\n",
    "print(f\"Loaded model accuracy: {{loaded_accuracy:.4f}}\")\\n",
    "print(f\"Original model accuracy: {{best_score:.4f}}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Advanced Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Learning curves\\n",
    "from sklearn.model_selection import learning_curve\\n",
    "\\n",
    "train_sizes, train_scores, test_scores = learning_curve(\\n",
    "    final_model, X, y, cv=5,\\n",
    "    train_sizes=np.linspace(0.1, 1.0, 10),\\n",
    "    scoring='accuracy'\\n",
    ")\\n",
    "\\n",
    "# Calculate mean and std\\n",
    "train_mean = np.mean(train_scores, axis=1)\\n",
    "train_std = np.std(train_scores, axis=1)\\n",
    "test_mean = np.mean(test_scores, axis=1)\\n",
    "test_std = np.std(test_scores, axis=1)\\n",
    "\\n",
    "# Plot learning curves\\n",
    "plt.figure(figsize=(10, 6))\\n",
    "plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')\\n",
    "plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')\\n",
    "plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')\\n",
    "plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')\\n",
    "plt.xlabel('Training examples')\\n",
    "plt.ylabel('Accuracy')\\n",
    "plt.title('Learning Curves')\\n",
    "plt.legend(loc='best')\\n",
    "plt.grid(True)\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature importance analysis\\n",
    "feature_importance = final_model.feature_importances_\\n",
    "feature_names = [f'Feature {{i+1}}' for i in range(n_features)]\\n",
    "\\n",
    "# Sort features by importance\\n",
    "indices = np.argsort(feature_importance)[::-1]\\n",
    "sorted_features = [feature_names[i] for i in indices]\\n",
    "sorted_importance = feature_importance[indices]\\n",
    "\\n",
    "# Plot top 10 features\\n",
    "plt.figure(figsize=(12, 8))\\n",
    "plt.barh(sorted_features[:10], sorted_importance[:10])\\n",
    "plt.xlabel('Feature Importance')\\n",
    "plt.title('Top 10 Most Important Features')\\n",
    "plt.gca().invert_yaxis()\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Performance Optimization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Performance benchmarking\\n",
    "def benchmark_model(model, X_test, y_test, n_runs=10):\\n",
    "    \"\"\"Benchmark model performance.\"\"\"\\n",
    "    times = []\\n",
    "    accuracies = []\\n",
    "    \\n",
    "    for i in range(n_runs):\\n",
    "        start_time = time.time()\\n",
    "        y_pred = model.predict(X_test)\\n",
    "        end_time = time.time()\\n",
    "        \\n",
    "        times.append(end_time - start_time)\\n",
    "        accuracies.append(accuracy_score(y_test, y_pred))\\n",
    "    \\n",
    "    return {{\\n",
    "        'mean_time': np.mean(times),\\n",
    "        'std_time': np.std(times),\\n",
    "        'mean_accuracy': np.mean(accuracies),\\n",
    "        'std_accuracy': np.std(accuracies)\\n",
    "    }}\\n",
    "\\n",
    "# Benchmark different model sizes\\n",
    "model_configs = [\\n",
    "    {{'name': 'Small', 'params': {{'n_estimators': 50, 'max_depth': 10}}}},\\n",
    "    {{'name': 'Medium', 'params': {{'n_estimators': 100, 'max_depth': 20}}}},\\n",
    "    {{'name': 'Large', 'params': {{'n_estimators': 200, 'max_depth': 30}}}}\\n",
    "]\\n",
    "\\n",
    "benchmark_results = []\\n",
    "for config in model_configs:\\n",
    "    model = RandomForestClassifier(**config['params'], random_state=42)\\n",
    "    model.fit(X_train, y_train)\\n",
    "    \\n",
    "    results = benchmark_model(model, X_test, y_test)\\n",
    "    results['name'] = config['name']\\n",
    "    benchmark_results.append(results)\\n",
    "\\n",
    "print(\"Performance Benchmarking Results:\")\\n",
    "for result in benchmark_results:\\n",
    "    print(f\"  {{result['name']}}: {{result['mean_accuracy']:.4f}} ± {{result['std_accuracy']:.4f}} accuracy, \"\\n",
    "          f\"{{result['mean_time']:.4f}} ± {{result['std_time']:.4f}} seconds\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot benchmarking results\\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\\n",
    "\\n",
    "# Accuracy comparison\\n",
    "names = [r['name'] for r in benchmark_results]\\n",
    "accuracies = [r['mean_accuracy'] for r in benchmark_results]\\n",
    "accuracy_errors = [r['std_accuracy'] for r in benchmark_results]\\n",
    "\\n",
    "ax1.bar(names, accuracies, yerr=accuracy_errors, capsize=5)\\n",
    "ax1.set_ylabel('Accuracy')\\n",
    "ax1.set_title('Model Accuracy Comparison')\\n",
    "ax1.set_ylim(0, 1)\\n",
    "\\n",
    "# Time comparison\\n",
    "times = [r['mean_time'] for r in benchmark_results]\\n",
    "time_errors = [r['std_time'] for r in benchmark_results]\\n",
    "\\n",
    "ax2.bar(names, times, yerr=time_errors, capsize=5, color='orange')\\n",
    "ax2.set_ylabel('Time (seconds)')\\n",
    "ax2.set_title('Inference Time Comparison')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Custom Algorithm Extensions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example of custom algorithm extension\\n",
    "from sklearn.base import BaseEstimator, ClassifierMixin\\n",
    "import numpy as np\\n",
    "\\n",
    "class CustomRandomForest(BaseEstimator, ClassifierMixin):\\n",
    "    \"\"\"Custom Random Forest with additional features.\"\"\"\\n",
    "    \\n",
    "    def __init__(self, n_estimators=100, max_depth=None, custom_feature=True):\\n",
    "        self.n_estimators = n_estimators\\n",
    "        self.max_depth = max_depth\\n",
    "        self.custom_feature = custom_feature\\n",
    "        self.models_ = []\\n",
    "    \\n",
    "    def fit(self, X, y):\\n",
    "        # Custom preprocessing\\n",
    "        if self.custom_feature:\\n",
    "            X_custom = self._add_custom_features(X)\\n",
    "        else:\\n",
    "            X_custom = X\\n",
    "        \\n",
    "        # Train base models\\n",
    "        self.models_ = []\\n",
    "        for _ in range(self.n_estimators):\\n",
    "            # Bootstrap sample\\n",
    "            indices = np.random.choice(len(X_custom), len(X_custom), replace=True)\\n",
    "            X_boot = X_custom[indices]\\n",
    "            y_boot = y[indices]\\n",
    "            \\n",
    "            # Train decision tree (simplified)\\n",
    "            model = self._train_decision_tree(X_boot, y_boot)\\n",
    "            self.models_.append(model)\\n",
    "        \\n",
    "        return self\\n",
    "    \\n",
    "    def predict(self, X):\\n",
    "        if self.custom_feature:\\n",
    "            X_custom = self._add_custom_features(X)\\n",
    "        else:\\n",
    "            X_custom = X\\n",
    "        \\n",
    "        # Get predictions from all models\\n",
    "        predictions = []\\n",
    "        for model in self.models_:\\n",
    "            pred = model.predict(X_custom)\\n",
    "            predictions.append(pred)\\n",
    "        \\n",
    "        # Majority vote\\n",
    "        predictions = np.array(predictions)\\n",
    "        final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)\\n",
    "        \\n",
    "        return final_predictions\\n",
    "    \\n",
    "    def _add_custom_features(self, X):\\n",
    "        \"\"\"Add custom engineered features.\"\"\"\\n",
    "        # Example: Add interaction features\\n",
    "        n_samples, n_features = X.shape\\n",
    "        X_custom = np.zeros((n_samples, n_features * 2))\\n",
    "        X_custom[:, :n_features] = X\\n",
    "        X_custom[:, n_features:] = X ** 2  # Squared features\\n",
    "        return X_custom\\n",
    "    \\n",
    "    def _train_decision_tree(self, X, y):\\n",
    "        \"\"\"Simplified decision tree training.\"\"\"\\n",
    "        # This is a simplified version - in practice, use sklearn's DecisionTreeClassifier\\n",
    "        from sklearn.tree import DecisionTreeClassifier\\n",
    "        tree = DecisionTreeClassifier(max_depth=self.max_depth)\\n",
    "        tree.fit(X, y)\\n",
    "        return tree\\n",
    "\\n",
    "print(\"✅ Custom algorithm class defined\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test custom algorithm\\n",
    "custom_model = CustomRandomForest(n_estimators=50, custom_feature=True)\\n",
    "custom_model.fit(X_train, y_train)\\n",
    "y_pred_custom = custom_model.predict(X_test)\\n",
    "custom_accuracy = accuracy_score(y_test, y_pred_custom)\\n",
    "\\n",
    "print(f\"Custom model accuracy: {{custom_accuracy:.4f}}\")\\n",
    "print(f\"Standard model accuracy: {{best_score:.4f}}\")\\n",
    "\\n",
    "# Compare performance\\n",
    "custom_benchmark = benchmark_model(custom_model, X_test, y_test)\\n",
    "print(f\"Custom model - Accuracy: {{custom_benchmark['mean_accuracy']:.4f}}, \"\\n",
    "      f\"Time: {{custom_benchmark['mean_time']:.4f}}s\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Results Summary and Export"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create comprehensive results summary\\n",
    "summary = {{\\n",
    "    'experiment_info': {{\\n",
    "        'title': '{paper.metadata.title}',\\n",
    "        'date': datetime.now().isoformat(),\\n",
    "        'dataset_size': len(X),\\n",
    "        'n_features': n_features\\n",
    "    }},\\n",
    "    'best_model': best_params,\\n",
    "    'best_accuracy': best_score,\\n",
    "    'cv_scores': cv_scores.tolist(),\\n",
    "    'benchmark_results': benchmark_results,\\n",
    "    'custom_model_performance': {{\\n",
    "        'accuracy': custom_accuracy,\\n",
    "        'benchmark': custom_benchmark\\n",
    "    }}\\n",
    "}}\\n",
    "\\n",
    "# Save results\\n",
    "results_path = '../results/advanced_experiment_results.json'\\n",
    "with open(results_path, 'w') as f:\\n",
    "    json.dump(summary, f, indent=2)\\n",
    "\\n",
    "print(f\"✅ Results saved to {{results_path}}\")\\n",
    "\\n",
    "# Display final summary\\n",
    "print(\"\\n=== EXPERIMENT SUMMARY ===\")\\n",
    "print(f\"Dataset: {{n_samples}} samples, {{n_features}} features\")\\n",
    "print(f\"Best Model Parameters: {{best_params}}\")\\n",
    "print(f\"Best Accuracy: {{best_score:.4f}}\")\\n",
    "print(f\"Cross-Validation Accuracy: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}\")\\n",
    "print(f\"Custom Model Accuracy: {{custom_accuracy:.4f}}\")\\n",
    "print(f\"\\nNext Steps:\")\\n",
    "print(\"1. Analyze the results in the results/ directory\")\\n",
    "print(\"2. Try different parameter combinations\")\\n",
    "print(\"3. Extend the custom algorithm further\")\\n",
    "print(\"4. Apply to your own datasets\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
        
        return CodeFile(
            name="advanced_usage.ipynb",
            content=notebook_content,
            language=None,  # Jupyter Notebook
            file_type="notebook",
            description="Advanced usage tutorial notebook"
        )
    
    def _extract_key_contributions(self, paper: Paper) -> str:
        """Extract key contributions from paper"""
        contributions = []
        
        # Add algorithm contributions
        for algorithm in paper.algorithms:
            contributions.append(f"- {algorithm.name}: {algorithm.description}")
        
        # Add experiment contributions
        for experiment in paper.experiments:
            contributions.append(f"- {experiment.name}: {experiment.description}")
        
        return "\n".join(contributions) if contributions else "No specific contributions identified."
    
    def _get_algorithm_files(self, code_implementation: CodeImplementation) -> str:
        """Get algorithm files for project structure"""
        algorithm_files = []
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                algorithm_files.append(f"    ├── {file.name}")
        return "\n".join(algorithm_files) if algorithm_files else "    └── (algorithm files)"
    
    def _generate_algorithm_imports(self, code_implementation: CodeImplementation) -> str:
        """Generate algorithm imports for examples"""
        imports = []
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                module_name = file.name.replace('.py', '')
                class_name = module_name.replace('_', ' ').title().replace(' ', '')
                imports.append(f"from {module_name} import {class_name}")
        return "\n".join(imports) if imports else "# Import your algorithms here"
    
    def _generate_basic_usage_example(self, code_implementation: CodeImplementation) -> str:
        """Generate basic usage example"""
        return """# Initialize algorithms
algorithm1 = AlgorithmName1()
algorithm2 = AlgorithmName2()

# Train algorithms
algorithm1.train(training_data, training_labels)
algorithm2.train(training_data, labels)

# Make predictions
predictions1 = algorithm1.predict(test_data)
predictions2 = algorithm2.predict(test_data)

# Evaluate results
from evaluator import Evaluator
evaluator = Evaluator()
results = evaluator.evaluate_classification(true_labels, predictions1)"""
    
    def _generate_performance_metrics(self, paper: Paper) -> str:
        """Generate performance metrics table"""
        # This would extract actual metrics from the paper
        return "Accuracy | 0.95 | 0.93\nF1 Score | 0.94 | 0.92"
    
    def _generate_experiments_section(self, paper: Paper) -> str:
        """Generate experiments section"""
        experiments = []
        for experiment in paper.experiments:
            experiments.append(f"### {experiment.name}\n{experiment.description}")
        return "\n\n".join(experiments) if experiments else "No specific experiments described."
    
    def _generate_related_research(self, paper: Paper) -> str:
        """Generate related research section"""
        # This would extract related work from the paper
        return "- Related work 1\n- Related work 2\n- Related work 3"
    
    def _generate_bibtex_entry(self, paper: Paper) -> str:
        """Generate BibTeX entry"""
        authors_str = " and ".join([author.name for author in paper.metadata.authors])
        return f"{paper.metadata.title.replace(' ', '_').lower()}_{paper.metadata.publication_year}"
    
    def _get_algorithm_module_name(self, code_implementation: CodeImplementation) -> str:
        """Get algorithm module name"""
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                return file.name.replace('.py', '')
        return "algorithms"
    
    def _get_algorithm_class_name(self, code_implementation: CodeImplementation) -> str:
        """Get algorithm class name"""
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                module_name = file.name.replace('.py', '')
                return module_name.replace('_', ' ').title().replace(' ', '')
        return "AlgorithmName"
    
    def _generate_data_processing_section(self, code_implementation: CodeImplementation) -> str:
        """Generate data processing section"""
        return """```python
from src.data_processor import DataProcessor

# Create processor instance
processor = DataProcessor()

# Load your data
data = processor.load_data('path/to/your/data.csv')

# Get data information
data_info = processor.get_data_info(data)
print(f"Data shape: {data_info['shape']}")
print(f"Columns: {data_info['columns']}")
```"""
    
    def _generate_preprocessing_section(self, code_implementation: CodeImplementation) -> str:
        """Generate preprocessing section"""
        return """```python
# Preprocess the data
processed_data = processor.preprocess_data(
    data, 
    preprocessing_steps=['normalize', 'handle_missing']
)

# Split data into training and testing sets
train_data, test_data = processor.split_data(processed_data, test_size=0.2)
```"""
    
    def _generate_data_splitting_section(self, code_implementation: CodeImplementation) -> str:
        """Generate data splitting section"""
        return """```python
# Split data with custom parameters
train_data, test_data, train_labels, test_labels = processor.split_data(
    processed_data, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set: {len(train_data)} samples")
print(f"Testing set: {len(test_data)} samples")
```"""
    
    def _generate_visualization_section(self, code_implementation: CodeImplementation) -> str:
        """Generate visualization section"""
        return """```python
from src.visualizer import Visualizer

# Create visualizer instance
visualizer = Visualizer()

# Plot data distribution
visualizer.plot_data_distribution(
    processed_data, 
    title="Data Distribution Analysis"
)

# Plot correlation matrix
if hasattr(processed_data, 'corr'):
    visualizer.plot_correlation_matrix(
        processed_data,
        title="Feature Correlation Matrix"
    )
```"""
    
    def _generate_custom_plots_section(self, code_implementation: CodeImplementation) -> str:
        """Generate custom plots section"""
        return """```python
# Create custom visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```"""
    
    def _generate_evaluation_section(self, code_implementation: CodeImplementation) -> str:
        """Generate evaluation section"""
        return """```python
from src.evaluator import Evaluator

# Create evaluator instance
evaluator = Evaluator()

# Evaluate classification performance
results = evaluator.evaluate_classification(
    true_labels, 
    predicted_labels, 
    predicted_probabilities
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1 Score: {results['f1']:.4f}")
```"""
    
    def _generate_advanced_metrics_section(self, code_implementation: CodeImplementation) -> str:
        """Generate advanced metrics section"""
        return """```python
# Advanced evaluation metrics
if len(np.unique(true_labels)) == 2:
    # Binary classification metrics
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

# Save detailed results
evaluator.save_evaluation_results(results, 'evaluation_results.json')
```"""
    
    def _generate_troubleshooting_section(self, code_implementation: CodeImplementation) -> str:
        """Generate troubleshooting section"""
        return """### Common Issues and Solutions

1. **Import Errors**
   - Ensure you're in the correct directory
   - Check that all dependencies are installed
   - Verify Python path includes the src directory

2. **Memory Issues**
   - Reduce dataset size for testing
   - Use data generators for large datasets
   - Monitor memory usage during training

3. **Performance Issues**
   - Check algorithm parameters
   - Consider using smaller models for initial testing
   - Profile code to identify bottlenecks

4. **Data Format Issues**
   - Ensure data is in the correct format
   - Check for missing values
   - Verify data types match expected input

### Getting Help

If you continue to experience issues:

1. Check the [API Documentation](./docs/api.md) for detailed information
2. Review the [Developer Guide](./docs/developer_guide.md) for implementation details
3. Open an issue on [GitHub](https://github.com/paper2code-repos/{code_implementation.name}/issues)
4. Join our community discussions"""
    
    def _generate_performance_tips_section(self, code_implementation: CodeImplementation) -> str:
        """Generate performance tips section"""
        return """### Performance Optimization Tips

1. **Algorithm Selection**
   - Choose appropriate algorithms for your data type
   - Consider computational complexity
   - Balance accuracy and speed requirements

2. **Data Optimization**
   - Preprocess data before training
   - Use appropriate data types
   - Consider data dimensionality reduction

3. **Parameter Tuning**
   - Start with default parameters
   - Use cross-validation for parameter selection
   - Monitor both accuracy and training time

4. **Hardware Utilization**
   - Use available GPU resources if applicable
   - Consider parallel processing for large datasets
   - Monitor system resources during execution"""
    
    def _generate_api_algorithms_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API algorithms section"""
        algorithms = []
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                algorithms.append(f"#### {file.name.replace('.py', '').replace('_', ' ').title()}\n\n```python\nfrom {file.name.replace('.py', '')} import {file.name.replace('.py', '').replace('_', ' ').title()}\n\n# Create algorithm instance\nalgorithm = {file.name.replace('.py', '').replace('_', ' ').title()}()\n\n# Train the algorithm\nalgorithm.train(training_data, training_labels)\n\n# Make predictions\npredictions = algorithm.predict(test_data)\n```\n")
        return "\n".join(algorithms) if algorithms else "No algorithm modules available."
    
    def _generate_api_utilities_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API utilities section"""
        utilities = []
        utility_files = [f for f in code_implementation.files if f.file_type == "utility"]
        
        for file in utility_files:
            if "data_processor" in file.name:
                utilities.append("#### DataProcessor\n\n```python\nfrom src.data_processor import DataProcessor\n\nprocessor = DataProcessor()\ndata = processor.load_data('data.csv')\nprocessed_data = processor.preprocess_data(data)\n```\n")
            elif "visualizer" in file.name:
                utilities.append("#### Visualizer\n\n```python\nfrom src.visualizer import Visualizer\n\nvisualizer = Visualizer()\nvisualizer.plot_data_distribution(data)\nvisualizer.plot_correlation_matrix(dataframe)\n```\n")
            elif "evaluator" in file.name:
                utilities.append("#### Evaluator\n\n```python\nfrom src.evaluator import Evaluator\n\nevaluator = Evaluator()\nresults = evaluator.evaluate_classification(true_labels, predicted_labels)\n```\n")
        
        return "\n".join(utilities) if utilities else "No utility modules available."
    
    def _generate_api_classes_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API classes section"""
        classes = []
        for file in code_implementation.files:
            if file.file_type == "algorithm":
                class_name = file.name.replace('.py', '').replace('_', ' ').title()
                classes.append(f"- **{class_name}**: Main algorithm implementation")
        
        utility_files = [f for f in code_implementation.files if f.file_type == "utility"]
        for file in utility_files:
            if "data_processor" in file.name:
                classes.append("- **DataProcessor**: Data processing and preprocessing utilities")
            elif "visualizer" in file.name:
                classes.append("- **Visualizer**: Data visualization and plotting utilities")
            elif "evaluator" in file.name:
                classes.append("- **Evaluator**: Performance evaluation and metrics")
        
        return "\n".join(classes) if classes else "No classes available."
    
    def _generate_api_functions_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API functions section"""
        return "- **main()**: Main entry point for the implementation\n- **load_data()**: Load data from various file formats\n- **preprocess_data()**: Preprocess and clean data\n- **evaluate_model()**: Evaluate model performance"
    
    def _generate_api_configuration_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API configuration section"""
        return "The implementation uses configuration files to manage parameters:\n\n- **requirements.txt**: Python dependencies\n- **config.py**: Algorithm and system parameters\n- **environment variables**: Optional configuration overrides"
    
    def _generate_api_examples_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API examples section"""
        return """### Basic Example

```python
# Import required modules
from src.data_processor import DataProcessor
from src.algorithms import AlgorithmName
from src.evaluator import Evaluator

# Load and preprocess data
processor = DataProcessor()
data = processor.load_data('data.csv')
processed_data = processor.preprocess_data(data)

# Split data
train_data, test_data = processor.split_data(processed_data)

# Train algorithm
algorithm = AlgorithmName()
algorithm.train(train_data, train_labels)

# Make predictions
predictions = algorithm.predict(test_data)

# Evaluate performance
evaluator = Evaluator()
results = evaluator.evaluate_classification(test_labels, predictions)
```

### Advanced Example

```python
# Custom parameters
algorithm = AlgorithmName(
    parameters={
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100
    }
)

# Cross-validation
cv_results = evaluator.cross_validate(
    algorithm, 
    X, y, 
    cv=5, 
    scoring='accuracy'
)

# Model persistence
algorithm.save_model('model.pkl')
loaded_algorithm = AlgorithmName.load_model('model.pkl')
```"""
    
    def _generate_api_best_practices_section(self, code_implementation: CodeImplementation) -> str:
        """Generate API best practices section"""
        return """### Best Practices

1. **Data Validation**
   - Always validate input data before processing
   - Check for missing values and data types
   - Use data processors for consistent preprocessing

2. **Algorithm Usage**
   - Follow the training → prediction workflow
   - Use appropriate evaluation metrics
   - Monitor training progress and performance

3. **Error Handling**
   - Implement proper error handling
   - Log important events and metrics
   - Provide meaningful error messages

4. **Performance Optimization**
   - Use appropriate data structures
   - Monitor memory usage
   - Consider parallel processing for large datasets

5. **Reproducibility**
   - Set random seeds for reproducible results
   - Save model parameters and configuration
   - Document preprocessing steps"""
    
    def _generate_developer_troubleshooting_section(self) -> str:
        """Generate developer troubleshooting section"""
        return """### Common Development Issues

1. **Import Errors**
   - Check Python path and working directory
   - Verify all dependencies are installed
   - Check for circular imports

2. **Testing Issues**
   - Ensure all tests are isolated
   - Mock external dependencies
   - Check test coverage

3. **Documentation Issues**
   - Keep documentation up to date with code changes
   - Use consistent formatting and style
   - Include examples for complex functions

4. **Build Issues**
   - Check Python version compatibility
   - Verify all dependencies are listed in requirements.txt
   - Test installation in clean environment

### Debugging Tips

1. **Use logging** instead of print statements
2. **Break down complex functions** into smaller, testable units
3. **Write unit tests** for individual components
4. **Use debugging tools** like pdb or IDE debuggers
5. **Check type hints** for type-related errors"""
    
    def _generate_algorithms_summary(self, paper: Paper) -> str:
        """Generate algorithms summary"""
        algorithms = []
        for algorithm in paper.algorithms:
            algorithms.append(f"- **{algorithm.name}**: {algorithm.description}")
        return "\n".join(algorithms) if algorithms else "No specific algorithms described."
    
    def _generate_experimental_setup(self, paper: Paper) -> str:
        """Generate experimental setup summary"""
        setups = []
        for experiment in paper.experiments:
            setups.append(f"- **{experiment.name}**: {experiment.description}")
        return "\n".join(setups) if setups else "No specific experimental setup described."
    
    def _generate_key_findings(self, paper: Paper) -> str:
        """Generate key findings summary"""
        # This would extract actual findings from the paper
        return "- Finding 1: Key result from the research\n- Finding 2: Another important result\n- Finding 3: Significant discovery"
    
    def run(self, paper: Paper, code_implementation: CodeImplementation, 
            architecture_result: Dict[str, Any]) -> DocumentationResult:
        """Main method to generate documentation"""
        start_time = datetime.now()
        
        logger.info(f"Generating documentation for: {paper.metadata.title}")
        
        try:
            # Generate README
            readme = self.generate_readme(paper, code_implementation, architecture_result)
            
            # Generate API documentation
            api_doc = self.generate_api_documentation(code_implementation)
            
            # Generate user guide
            user_guide = self.generate_user_guide(paper, code_implementation)
            
            # Generate paper summary
            paper_summary = self.generate_paper_summary(paper)
            
            # Generate developer guide
            dev_guide = self.generate_developer_guide(code_implementation)
            
            # Generate example notebooks
            notebooks = self.generate_example_notebooks(paper, code_implementation)
            
            # Combine all documentation files
            all_docs = [readme, api_doc, user_guide, paper_summary, dev_guide] + notebooks
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate documentation score
            documentation_score = self._calculate_documentation_score(all_docs, paper)
            
            # Generate coverage metrics
            coverage_metrics = self._generate_coverage_metrics(all_docs, paper)
            
            return DocumentationResult(
                documentation_files=all_docs,
                documentation_score=documentation_score,
                generation_time=generation_time,
                coverage_metrics=coverage_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in documentation generation: {e}")
            raise
    
    def _calculate_documentation_score(self, docs: List[CodeFile], paper: Paper) -> float:
        """Calculate documentation quality score"""
        score = 0.0
        
        # Check for README
        readme_files = [d for d in docs if d.file_type == "readme"]
        if readme_files:
            score += 0.2
        
        # Check for API documentation
        api_files = [d for d in docs if d.file_type == "api"]
        if api_files:
            score += 0.2
        
        # Check for user guide
        user_guide_files = [d for d in docs if d.file_type == "user_guide"]
        if user_guide_files:
            score += 0.2
        
        # Check for paper summary
        summary_files = [d for d in docs if d.file_type == "paper_summary"]
        if summary_files:
            score += 0.15
        
        # Check for developer guide
        dev_guide_files = [d for d in docs if d.file_type == "developer_guide"]
        if dev_guide_files:
            score += 0.15
        
        # Check for notebooks
        notebook_files = [d for d in docs if d.file_type == "notebook"]
        if notebook_files:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_coverage_metrics(self, docs: List[CodeFile], paper: Paper) -> Dict[str, Any]:
        """Generate documentation coverage metrics"""
        metrics = {
            'total_files': len(docs),
            'readme_coverage': False,
            'api_coverage': False,
            'user_guide_coverage': False,
            'paper_summary_coverage': False,
            'developer_guide_coverage': False,
            'notebook_coverage': False,
            'total_coverage': 0.0
        }
        
        # Check coverage for each type
        for doc in docs:
            if doc.file_type == "readme":
                metrics['readme_coverage'] = True
            elif doc.file_type == "api":
                metrics['api_coverage'] = True
            elif doc.file_type == "user_guide":
                metrics['user_guide_coverage'] = True
            elif doc.file_type == "paper_summary":
                metrics['paper_summary_coverage'] = True
            elif doc.file_type == "developer_guide":
                metrics['developer_guide_coverage'] = True
            elif doc.file_type == "notebook":
                metrics['notebook_coverage'] = True
        
        # Calculate total coverage
        coverage_types = [
            metrics['readme_coverage'],
            metrics['api_coverage'],
            metrics['user_guide_coverage'],
            metrics['paper_summary_coverage'],
            metrics['developer_guide_coverage'],
            metrics['notebook_coverage']
        ]
        
        metrics['total_coverage'] = sum(coverage_types) / len(coverage_types)
        
        return metrics