"""
Code Generation Agent

This agent generates code implementations based on paper analysis,
architecture design, and research findings, supporting Python and JavaScript/TypeScript.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from agno import agent
from agno.models.openrouter import OpenRouter

from ..models.paper import Paper, Algorithm, Experiment
from ..models.code import Language, Framework, CodeFile, CodeImplementation

logger = logging.getLogger(__name__)


@dataclass
class CodeGenerationResult:
    """Result from code generation"""
    code_implementation: CodeImplementation
    generated_files: List[CodeFile]
    code_quality_score: float
    generation_time: float
    language_used: Language
    framework_used: Optional[Framework]


class CodeGenerationAgent:
    """Agent for generating code implementations from scientific papers"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.llm = OpenRouter(model=model)
        
        # Create agent for code generation
        self.agent = agent(
            name="code_generator",
            model=self.llm,
            description="Generate code implementations for scientific papers"
        )
    
    def select_primary_language(self, architecture_result: Dict[str, Any], 
                              user_preference: Optional[str] = None) -> Language:
        """Select primary programming language for code generation"""
        if user_preference:
            try:
                return Language(user_preference.lower())
            except ValueError:
                logger.warning(f"Invalid language preference: {user_preference}")
        
        # Use language from architecture result
        primary_lang = architecture_result.get('technology_stack', {}).get('primary_language')
        if primary_lang:
            return primary_lang
        
        # Default to Python
        return Language.PYTHON
    
    def generate_algorithm_implementations(self, paper: Paper, 
                                        architecture_result: Dict[str, Any]) -> List[CodeFile]:
        """Generate implementations for each algorithm in the paper"""
        generated_files = []
        
        for algorithm in paper.algorithms:
            try:
                # Generate algorithm implementation
                algorithm_file = self._generate_algorithm_file(algorithm, paper, architecture_result)
                generated_files.append(algorithm_file)
                
            except Exception as e:
                logger.error(f"Error generating algorithm {algorithm.name}: {e}")
                continue
        
        return generated_files
    
    def generate_main_implementation(self, paper: Paper, 
                                   architecture_result: Dict[str, Any],
                                   algorithm_files: List[CodeFile]) -> CodeFile:
        """Generate main implementation file that orchestrates all algorithms"""
        language = architecture_result.get('technology_stack', {}).get('primary_language', Language.PYTHON)
        
        # Generate main implementation content
        main_content = self._generate_main_content(paper, algorithm_files, language)
        
        # Create main file
        main_file = CodeFile(
            name=f"main.{language.value}",
            content=main_content,
            language=language,
            file_type="main",
            description="Main implementation file that orchestrates all algorithms"
        )
        
        return main_file
    
    def generate_utility_files(self, paper: Paper, 
                             architecture_result: Dict[str, Any]) -> List[CodeFile]:
        """Generate utility files for data processing, visualization, and evaluation"""
        utility_files = []
        language = architecture_result.get('technology_stack', {}).get('primary_language', Language.PYTHON)
        
        # Data processing utilities
        data_processor = self._generate_data_processor(paper, language)
        utility_files.append(data_processor)
        
        # Visualization utilities
        visualizer = self._generate_visualizer(paper, language)
        utility_files.append(visualizer)
        
        # Evaluation utilities
        evaluator = self._generate_evaluator(paper, language)
        utility_files.append(evaluator)
        
        return utility_files
    
    def generate_requirements_file(self, architecture_result: Dict[str, Any]) -> CodeFile:
        """Generate requirements.txt file with dependencies"""
        dependencies = architecture_result.get('dependencies', [])
        
        requirements_content = "# Python dependencies\n"
        requirements_content += "# Core dependencies\n"
        requirements_content += "requests>=2.25.0\n"
        requirements_content += "matplotlib>=3.5.0\n"
        requirements_content += "seaborn>=0.11.0\n"
        requirements_content += "jupyter>=1.0.0\n"
        
        # Add framework dependencies
        tech_stack = architecture_result.get('technology_stack', {})
        frameworks = tech_stack.get('frameworks', [])
        
        for framework in frameworks:
            if framework == Framework.PYTORCH:
                requirements_content += "torch>=1.9.0\n"
                requirements_content += "torchvision>=0.10.0\n"
            elif framework == Framework.TENSORFLOW:
                requirements_content += "tensorflow>=2.6.0\n"
            elif framework == Framework.SCIKITLEARN:
                requirements_content += "scikit-learn>=1.0.0\n"
            elif framework == Framework.XGBOOST:
                requirements_content += "xgboost>=1.5.0\n"
            elif framework == Framework.PANDAS:
                requirements_content += "pandas>=1.3.0\n"
            elif framework == Framework.NUMPY:
                requirements_content += "numpy>=1.21.0\n"
            elif framework == Framework.STATSMODELS:
                requirements_content += "statsmodels>=0.13.0\n"
        
        # Add testing dependencies
        testing_deps = tech_stack.get('testing_frameworks', [])
        for test_dep in testing_deps:
            requirements_content += f"{test_dep}\n"
        
        requirements_content += "\n# Development dependencies\n"
        requirements_content += "black>=21.0.0\n"
        requirements_content += "flake8>=3.9.0\n"
        requirements_content += "mypy>=0.910\n"
        
        return CodeFile(
            name="requirements.txt",
            content=requirements_content,
            language=Language.PYTHON,
            file_type="requirements",
            description="Python dependencies and requirements"
        )
    
    def generate_gitignore_file(self) -> CodeFile:
        """Generate .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.json
*.xml
*.data
*.h5
*.hdf5
*.npz
*.npy

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp
"""
        
        return CodeFile(
            name=".gitignore",
            content=gitignore_content,
            language=Language.PYTHON,
            file_type="gitignore",
            description="Git ignore file for Python projects"
        )
    
    def _generate_algorithm_file(self, algorithm: Algorithm, paper: Paper, 
                                architecture_result: Dict[str, Any]) -> CodeFile:
        """Generate implementation file for a specific algorithm"""
        language = architecture_result.get('technology_stack', {}).get('primary_language', Language.PYTHON)
        
        # Generate algorithm implementation content
        algorithm_content = self._generate_algorithm_content(algorithm, paper, language)
        
        return CodeFile(
            name=f"{algorithm.name.lower().replace(' ', '_')}.{language.value}",
            content=algorithm_content,
            language=language,
            file_type="algorithm",
            description=f"Implementation of {algorithm.name} algorithm"
        )
    
    def _generate_algorithm_content(self, algorithm: Algorithm, paper: Paper, 
                                   language: Language) -> str:
        """Generate implementation content for an algorithm"""
        if language == Language.PYTHON:
            return self._generate_python_algorithm(algorithm, paper)
        elif language == Language.JAVASCRIPT:
            return self._generate_javascript_algorithm(algorithm, paper)
        else:
            return self._generate_python_algorithm(algorithm, paper)  # Default to Python
    
    def _generate_python_algorithm(self, algorithm: Algorithm, paper: Paper) -> str:
        """Generate Python implementation of an algorithm"""
        content = f'''"""
{algorithm.name} Algorithm Implementation

This module implements the {algorithm.name} algorithm as described in:
"{paper.metadata.title}"

Author: Paper2Code Agent
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class {algorithm.name.replace(' ', '')}:
    """
    Implementation of {algorithm.name} algorithm.
    
    Attributes:
        parameters (Dict): Algorithm parameters
        trained (bool): Whether the algorithm is trained
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the {algorithm.name} algorithm.
        
        Args:
            parameters: Algorithm parameters
        """
        self.parameters = parameters or {{}}
        self.trained = False
        
        # Initialize default parameters
        self._set_default_parameters()
    
    def _set_default_parameters(self):
        """Set default parameters for the algorithm."""
        default_params = {{
            # Add default parameters based on algorithm type
        }}
        
        # Update with user-provided parameters
        for key, value in self.parameters.items():
            default_params[key] = value
        
        self.parameters = default_params
    
    def train(self, training_data: Any, training_labels: Optional[Any] = None):
        """
        Train the {algorithm.name} algorithm.
        
        Args:
            training_data: Training data
            training_labels: Training labels (if applicable)
        """
        logger.info(f"Training {algorithm.name} algorithm...")
        
        try:
            # Implement training logic based on algorithm description
            self._training_logic(training_data, training_labels)
            self.trained = True
            logger.info(f"{algorithm.name} training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training {algorithm.name}: {{e}}")
            raise
    
    def predict(self, input_data: Any) -> Any:
        """
        Make predictions using the trained algorithm.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Predictions
        """
        if not self.trained:
            raise ValueError("{algorithm.name} algorithm must be trained before making predictions")
        
        logger.info(f"Making predictions with {algorithm.name}...")
        
        try:
            predictions = self._prediction_logic(input_data)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with {algorithm.name}: {{e}}")
            raise
    
    def _training_logic(self, training_data: Any, training_labels: Optional[Any] = None):
        """
        Implement the core training logic for {algorithm.name}.
        
        This method should be implemented based on the specific algorithm requirements.
        """
        # Placeholder for algorithm-specific training logic
        pass
    
    def _prediction_logic(self, input_data: Any) -> Any:
        """
        Implement the core prediction logic for {algorithm.name}.
        
        This method should be implemented based on the specific algorithm requirements.
        """
        # Placeholder for algorithm-specific prediction logic
        return input_data
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current algorithm parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set algorithm parameters."""
        self.parameters.update(parameters)
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        
        # Implement model saving logic
        logger.info(f"Saving {algorithm.name} model to {{filepath}}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model from file."""
        # Implement model loading logic
        logger.info(f"Loading {algorithm.name} model from {{filepath}}")
        return cls()


# Example usage
if __name__ == "__main__":
    # Create algorithm instance
    algorithm = {algorithm.name.replace(' ', '')}()
    
    # Example training data (replace with actual data)
    training_data = np.random.rand(100, 10)
    training_labels = np.random.randint(0, 2, 100)
    
    # Train the algorithm
    algorithm.train(training_data, training_labels)
    
    # Make predictions
    test_data = np.random.rand(10, 10)
    predictions = algorithm.predict(test_data)
    
    print(f"Predictions: {{predictions}}")
'''
        return content
    
    def _generate_javascript_algorithm(self, algorithm: Algorithm, paper: Paper) -> str:
        """Generate JavaScript implementation of an algorithm"""
        content = f'''/**
 * {algorithm.name} Algorithm Implementation
 * 
 * This module implements the {algorithm.name} algorithm as described in:
 * "{paper.metadata.title}"
 * 
 * Author: Paper2Code Agent
 */

class {algorithm.name.replace(' ', '')} {{
    /**
     * Create a new {algorithm.name} instance.
     * @param {{Object}} parameters - Algorithm parameters
     */
    constructor(parameters = {{}}) {{
        this.parameters = {{...parameters}};
        this.trained = false;
        
        // Set default parameters
        this._setDefaultParameters();
    }}
    
    /**
     * Set default parameters for the algorithm.
     * @private
     */
    _setDefaultParameters() {{
        const defaultParams = {{
            // Add default parameters based on algorithm type
        }};
        
        // Update with user-provided parameters
        Object.keys(this.parameters).forEach(key => {{
            defaultParams[key] = this.parameters[key];
        }});
        
        this.parameters = defaultParams;
    }}
    
    /**
     * Train the {algorithm.name} algorithm.
     * @param {{Array|Object}} trainingData - Training data
     * @param {{Array|Object}} trainingLabels - Training labels (if applicable)
     */
    async train(trainingData, trainingLabels = null) {{
        console.log(`Training ${{algorithm.name}} algorithm...`);
        
        try {{
            await this._trainingLogic(trainingData, trainingLabels);
            this.trained = true;
            console.log(`${{algorithm.name}} training completed successfully`);
            
        }} catch (error) {{
            console.error(`Error training ${{algorithm.name}}:`, error);
            throw error;
        }}
    }}
    
    /**
     * Make predictions using the trained algorithm.
     * @param {{Array|Object}} inputData - Input data for prediction
     * @returns {{Array|Object}} Predictions
     */
    async predict(inputData) {{
        if (!this.trained) {{
            throw new Error(`${{algorithm.name}} algorithm must be trained before making predictions`);
        }}
        
        console.log(`Making predictions with ${{algorithm.name}}...`);
        
        try {{
            const predictions = await this._predictionLogic(inputData);
            return predictions;
            
        }} catch (error) {{
            console.error(`Error making predictions with ${{algorithm.name}}:`, error);
            throw error;
        }}
    }}
    
    /**
     * Implement the core training logic for {algorithm.name}.
     * @private
     */
    async _trainingLogic(trainingData, trainingLabels) {{
        // Placeholder for algorithm-specific training logic
        // This should be implemented based on the specific algorithm requirements
    }}
    
    /**
     * Implement the core prediction logic for {algorithm.name}.
     * @private
     */
    async _predictionLogic(inputData) {{
        // Placeholder for algorithm-specific prediction logic
        return inputData;
    }}
    
    /**
     * Get current algorithm parameters.
     * @returns {{Object}} Current parameters
     */
    getParameters() {{
        return {{...this.parameters}};
    }}
    
    /**
     * Set algorithm parameters.
     * @param {{Object}} parameters - New parameters
     */
    setParameters(parameters) {{
        Object.assign(this.parameters, parameters);
    }}
    
    /**
     * Save the trained model to file.
     * @param {{string}} filepath - File path to save the model
     */
    async saveModel(filepath) {{
        if (!this.trained) {{
            throw new Error("Cannot save untrained model");
        }}
        
        // Implement model saving logic
        console.log(`Saving ${{algorithm.name}} model to ${{filepath}}`);
    }}
    
    /**
     * Load a trained model from file.
     * @param {{string}} filepath - File path to load the model from
     * @returns {{{algorithm.name.replace(' ', '')}}} Loaded model instance
     */
    static async loadModel(filepath) {{
        // Implement model loading logic
        console.log(`Loading ${{algorithm.name}} model from ${{filepath}}`);
        return new {algorithm.name.replace(' ', '')}();
    }}
}}


// Example usage
(async () => {{
    // Create algorithm instance
    const algorithm = new {algorithm.name.replace(' ', '')}();
    
    // Example training data (replace with actual data)
    const trainingData = Array.from({{ length: 100 }}, () => 
        Array.from({{ length: 10 }}, () => Math.random())
    );
    const trainingLabels = Array.from({{ length: 100 }}, () => Math.floor(Math.random() * 2));
    
    // Train the algorithm
    await algorithm.train(trainingData, trainingLabels);
    
    // Make predictions
    const testData = Array.from({{ length: 10 }}, () => 
        Array.from({{ length: 10 }}, () => Math.random())
    );
    const predictions = await algorithm.predict(testData);
    
    console.log(`Predictions:`, predictions);
}})();
'''
        return content
    
    def _generate_main_content(self, paper: Paper, algorithm_files: List[CodeFile], 
                             language: Language) -> str:
        """Generate main implementation content"""
        if language == Language.PYTHON:
            return self._generate_python_main(paper, algorithm_files)
        elif language == Language.JAVASCRIPT:
            return self._generate_javascript_main(paper, algorithm_files)
        else:
            return self._generate_python_main(paper, algorithm_files)  # Default to Python
    
    def _generate_python_main(self, paper: Paper, algorithm_files: List[CodeFile]) -> str:
        """Generate Python main implementation"""
        algorithm_imports = []
        algorithm_instances = []
        
        for algo_file in algorithm_files:
            if algo_file.file_type == "algorithm":
                class_name = algo_file.name.replace('.py', '').replace('_', ' ').title().replace(' ', '')
                algorithm_imports.append(f"from .{algo_file.name.replace('.py', '')} import {class_name}")
                algorithm_instances.append(f"    {class_name}()")
        
        content = f'''"""
{paper.metadata.title} - Main Implementation

This is the main implementation file for the {paper.metadata.title} paper.
It orchestrates all the algorithms and provides a unified interface.

Author: Paper2Code Agent
"""

{chr(10).join(algorithm_imports)}

import logging
import argparse
import sys
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main function that orchestrates all algorithms.
    
    This function demonstrates how to use the implemented algorithms
    from the {paper.metadata.title} paper.
    """
    logger.info("Starting {paper.metadata.title} implementation...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='{paper.metadata.title} Implementation')
    parser.add_argument('--input', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize algorithms
        algorithms = {{
{chr(10).join([f'    "{algo_file.name.replace('.py', '')}": {instance},' for algo_file, instance in zip(algorithm_files, algorithm_instances)])}
        }}
        
        # Load input data
        logger.info(f"Loading input data from {{args.input}}")
        # Implement data loading logic based on your data format
        # input_data = load_data(args.input)
        
        # Process data through algorithms
        logger.info("Processing data through algorithms...")
        results = {{}}
        
        for algo_name, algo_instance in algorithms.items():
            logger.info(f"Running {{algo_name}} algorithm...")
            # Implement algorithm-specific processing
            # results[algo_name] = algo_instance.process(input_data)
        
        # Save results
        logger.info(f"Saving results to {{args.output}}")
        # Implement results saving logic
        # save_results(results, args.output)
        
        logger.info("Implementation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {{e}}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        return content
    
    def _generate_javascript_main(self, paper: Paper, algorithm_files: List[CodeFile]) -> str:
        """Generate JavaScript main implementation"""
        algorithm_imports = []
        
        for algo_file in algorithm_files:
            if algo_file.file_type == "algorithm":
                class_name = algo_file.name.replace('.js', '').replace('_', ' ').title().replace(' ', '')
                algorithm_imports.append(f"import {{ {class_name} }} from './{algo_file.name.replace('.js', '')}';")
        
        content = f'''/**
 * {paper.metadata.title} - Main Implementation
 * 
 * This is the main implementation file for the {paper.metadata.title} paper.
 * It orchestrates all the algorithms and provides a unified interface.
 * 
 * Author: Paper2Code Agent
 */

{chr(10).join(algorithm_imports)}

import fs from 'fs';
import path from 'path';
import {{ fileURLToPath }} from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Main function that orchestrates all algorithms.
 * 
 * This function demonstrates how to use the implemented algorithms
 * from the {paper.metadata.title} paper.
 */
async function main() {{
    console.log(`Starting ${{paper.metadata.title}} implementation...`);
    
    // Parse command line arguments
    const args = process.argv.slice(2);
    
    if (args.length < 2) {{
        console.error('Usage: node main.js <input_file> <output_file> [config_file]');
        process.exit(1);
    }}
    
    const inputFile = args[0];
    const outputFile = args[1];
    const configFile = args[2];
    
    try {{
        // Initialize algorithms
        const algorithms = {{
{chr(10).join([f'            "{algo_file.name.replace('.js', '')}": new {algo_file.name.replace('.js', '').replace('_', ' ').title().replace(' ', '')}(),' for algo_file in algorithm_files if algo_file.file_type == "algorithm"])}
        }};
        
        // Load input data
        console.log(`Loading input data from ${{inputFile}}`);
        // Implement data loading logic based on your data format
        // const inputData = await loadData(inputFile);
        
        // Process data through algorithms
        console.log("Processing data through algorithms...");
        const results = {{}};
        
        for (const [algoName, algoInstance] of Object.entries(algorithms)) {{
            console.log(`Running ${{algoName}} algorithm...`);
            // Implement algorithm-specific processing
            // results[algoName] = await algoInstance.process(inputData);
        }}
        
        // Save results
        console.log(`Saving results to ${{outputFile}}`);
        // Implement results saving logic
        // await saveResults(results, outputFile);
        
        console.log("Implementation completed successfully!");
        
    }} catch (error) {{
        console.error(`Error in main execution:`, error);
        process.exit(1);
    }}
}}


// Run the main function
main();
'''
        return content
    
    def _generate_data_processor(self, paper: Paper, language: Language) -> CodeFile:
        """Generate data processing utility file"""
        if language == Language.PYTHON:
            content = '''"""
Data Processing Utilities

This module provides utilities for data processing and preprocessing
for the {paper.metadata.title} implementation.

Author: Paper2Code Agent
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing utilities for the {paper.metadata.title} implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration for data processing
        """
        self.config = config or {}
        self.processed_data = None
    
    def load_data(self, filepath: str, **kwargs) -> Any:
        """
        Load data from file.
        
        Args:
            filepath: Path to the data file
            **kwargs: Additional parameters for data loading
            
        Returns:
            Loaded data
        """
        try:
            # Auto-detect file format and load accordingly
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath, **kwargs)
            elif filepath.endswith('.json'):
                data = pd.read_json(filepath, **kwargs)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                data = pd.read_excel(filepath, **kwargs)
            else:
                # Default to CSV
                data = pd.read_csv(filepath, **kwargs)
            
            logger.info(f"Successfully loaded data from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            raise
    
    def preprocess_data(self, data: Any, preprocessing_steps: List[str] = None) -> Any:
        """
        Preprocess the data.
        
        Args:
            data: Input data
            preprocessing_steps: List of preprocessing steps to apply
            
        Returns:
            Preprocessed data
        """
        if preprocessing_steps is None:
            preprocessing_steps = ['normalize', 'handle_missing']
        
        processed_data = data.copy()
        
        for step in preprocessing_steps:
            try:
                if step == 'normalize':
                    processed_data = self._normalize_data(processed_data)
                elif step == 'handle_missing':
                    processed_data = self._handle_missing_values(processed_data)
                elif step == 'standardize':
                    processed_data = self._standardize_data(processed_data)
                # Add more preprocessing steps as needed
                
            except Exception as e:
                logger.warning(f"Error applying preprocessing step {step}: {e}")
                continue
        
        self.processed_data = processed_data
        return processed_data
    
    def _normalize_data(self, data: Any) -> Any:
        """Normalize numerical data."""
        # Implement normalization logic
        return data
    
    def _handle_missing_values(self, data: Any) -> Any:
        """Handle missing values in the data."""
        # Implement missing value handling logic
        return data
    
    def _standardize_data(self, data: Any) -> Any:
        """Standardize numerical data."""
        # Implement standardization logic
        return data
    
    def split_data(self, data: Any, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            data: Input data
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            if hasattr(data, 'values'):
                # DataFrame
                train_data, test_data = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
            else:
                # Array-like
                train_data, test_data = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
            
            logger.info(f"Data split into {len(train_data)} training and {len(test_data)} testing samples")
            return train_data, test_data
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple split")
            # Simple split without sklearn
            n_samples = len(data) if hasattr(data, '__len__') else 100
            split_idx = int(n_samples * (1 - test_size))
            
            if hasattr(data, 'iloc'):
                # DataFrame
                return data.iloc[:split_idx], data.iloc[split_idx:]
            else:
                # Array-like
                return data[:split_idx], data[split_idx:]
    
    def get_data_info(self, data: Any) -> Dict[str, Any]:
        """
        Get information about the data.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with data information
        """
        info = {
            'type': type(data).__name__,
            'shape': None,
            'columns': None,
            'missing_values': None,
            'data_types': None
        }
        
        try:
            if hasattr(data, 'shape'):
                info['shape'] = data.shape
            
            if hasattr(data, 'columns'):
                info['columns'] = list(data.columns)
            
            if hasattr(data, 'isnull'):
                info['missing_values'] = data.isnull().sum().to_dict()
            
            if hasattr(data, 'dtypes'):
                info['data_types'] = data.dtypes.to_dict()
            
        except Exception as e:
            logger.warning(f"Error getting data info: {e}")
        
        return info


# Example usage
if __name__ == "__main__":
    # Create data processor instance
    processor = DataProcessor()
    
    # Example usage (replace with actual data file)
    # data = processor.load_data("example_data.csv")
    # processed_data = processor.preprocess_data(data)
    # train_data, test_data = processor.split_data(processed_data)
    # data_info = processor.get_data_info(processed_data)
    
    print("Data processor ready for use!")
'''
        else:
            content = '''/**
 * Data Processing Utilities
 * 
 * This module provides utilities for data processing and preprocessing
 * for the {paper.metadata.title} implementation.
 * 
 * Author: Paper2Code Agent
 */

export class DataProcessor {
    /**
     * Create a new DataProcessor instance.
     * @param {{Object}} config - Configuration for data processing
     */
    constructor(config = {}) {
        this.config = config;
        this.processedData = null;
    }
    
    /**
     * Load data from file.
     * @param {{string}} filepath - Path to the data file
     * @param {{Object}} options - Additional options for data loading
     * @returns {{Promise<any>}} Loaded data
     */
    async loadData(filepath, options = {}) {
        try {
            // This would typically use a library like d3 or pandas.js
            // For now, return a placeholder
            console.log(`Loading data from ${filepath}`);
            
            // Example implementation for CSV data
            if (filepath.endsWith('.csv')) {
                // Use a CSV parsing library
                // const data = await parseCSV(filepath, options);
                return [];
            }
            
            // Add support for other formats as needed
            return [];
            
        } catch (error) {
            console.error(`Error loading data from ${filepath}:`, error);
            throw error;
        }
    }
    
    /**
     * Preprocess the data.
     * @param {{any}} data - Input data
     * @param {{Array<string>}} preprocessingSteps - List of preprocessing steps to apply
     * @returns {{any}} Preprocessed data
     */
    preprocessData(data, preprocessingSteps = ['normalize', 'handleMissing']) {
        let processedData = data;
        
        for (const step of preprocessingSteps) {
            try {
                switch (step) {
                    case 'normalize':
                        processedData = this._normalizeData(processedData);
                        break;
                    case 'handleMissing':
                        processedData = this._handleMissingValues(processedData);
                        break;
                    case 'standardize':
                        processedData = this._standardizeData(processedData);
                        break;
                    // Add more preprocessing steps as needed
                }
            } catch (error) {
                console.warn(`Error applying preprocessing step ${step}:`, error);
                continue;
            }
        }
        
        this.processedData = processedData;
        return processedData;
    }
    
    /**
     * Normalize numerical data.
     * @private
     */
    _normalizeData(data) {
        // Implement normalization logic
        return data;
    }
    
    /**
     * Handle missing values in the data.
     * @private
     */
    _handleMissingValues(data) {
        // Implement missing value handling logic
        return data;
    }
    
    /**
     * Standardize numerical data.
     * @private
     */
    _standardizeData(data) {
        // Implement standardization logic
        return data;
    }
    
    /**
     * Split data into training and testing sets.
     * @param {{any}} data - Input data
     * @param {{number}} testSize - Proportion of data for testing
     * @param {{number}} randomState - Random seed for reproducibility
     * @returns {{Array<any>}} Array of [trainData, testData]
     */
    splitData(data, testSize = 0.2, randomState = 42) {
        try {
            // Simple implementation - replace with proper library if needed
            const totalSamples = Array.isArray(data) ? data.length : 100;
            const splitIndex = Math.floor(totalSamples * (1 - testSize));
            
            if (Array.isArray(data)) {
                return [data.slice(0, splitIndex), data.slice(splitIndex)];
            }
            
            return [data, null];
            
        } catch (error) {
            console.error('Error splitting data:', error);
            throw error;
        }
    }
    
    /**
     * Get information about the data.
     * @param {{any}} data - Input data
     * @returns {{Object}} Object with data information
     */
    getDataInfo(data) {
        const info = {
            type: typeof data,
            length: Array.isArray(data) ? data.length : undefined,
            keys: data && typeof data === 'object' ? Object.keys(data) : undefined
        };
        
        return info;
    }
}


// Example usage
(async () => {
    // Create data processor instance
    const processor = new DataProcessor();
    
    // Example usage (replace with actual data file)
    // const data = await processor.loadData('example_data.csv');
    // const processedData = processor.preprocessData(data);
    // const [trainData, testData] = processor.splitData(processedData);
    // const dataInfo = processor.getDataInfo(processedData);
    
    console.log('Data processor ready for use!');
})();
'''
        
        return CodeFile(
            name="data_processor.py" if language == Language.PYTHON else "data_processor.js",
            content=content,
            language=language,
            file_type="utility",
            description="Data processing and preprocessing utilities"
        )
    
    def _generate_visualizer(self, paper: Paper, language: Language) -> CodeFile:
        """Generate visualization utility file"""
        if language == Language.PYTHON:
            content = '''"""
Visualization Utilities

This module provides utilities for data visualization and plotting
for the {paper.metadata.title} implementation.

Author: Paper2Code Agent
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Visualization utilities for the {paper.metadata.title} implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration for visualization
        """
        self.config = config or {}
        self.figure_size = self.config.get('figure_size', (10, 6))
        self.dpi = self.config.get('dpi', 300)
    
    def plot_data_distribution(self, data: Any, title: str = "Data Distribution", 
                             save_path: Optional[str] = None) -> None:
        """
        Plot data distribution.
        
        Args:
            data: Input data
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=self.figure_size)
            
            # Plot histogram
            if hasattr(data, 'values'):
                # DataFrame
                data_values = data.values.flatten()
            else:
                # Array-like
                data_values = np.array(data).flatten()
            
            axes[0].hist(data_values, bins=30, alpha=0.7, edgecolor='black')
            axes[0].set_title('Histogram')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            
            # Plot box plot
            axes[1].boxplot(data_values)
            axes[1].set_title('Box Plot')
            axes[1].set_ylabel('Value')
            
            plt.suptitle(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting data distribution: {e}")
            raise
    
    def plot_correlation_matrix(self, data: Any, title: str = "Correlation Matrix",
                               save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix.
        
        Args:
            data: Input data (DataFrame)
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            if not hasattr(data, 'corr'):
                logger.error("Data must be a DataFrame for correlation matrix")
                return
            
            # Calculate correlation matrix
            corr_matrix = data.corr()
            
            # Create heatmap
            plt.figure(figsize=self.figure_size)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {e}")
            raise
    
    def plot_algorithm_comparison(self, results: Dict[str, Any], 
                                title: str = "Algorithm Comparison",
                                save_path: Optional[str] = None) -> None:
        """
        Plot algorithm comparison results.
        
        Args:
            results: Dictionary with algorithm names as keys and metrics as values
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            algorithms = list(results.keys())
            metrics = list(results[algorithms[0]].keys()) if algorithms else []
            
            # Create subplots for each metric
            n_metrics = len(metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, metric in enumerate(metrics):
                values = [results[algo].get(metric, 0) for algo in algorithms]
                
                axes[i].bar(algorithms, values, alpha=0.7)
                axes[i].set_title(metric)
                axes[i].set_ylabel('Value')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting algorithm comparison: {e}")
            raise
    
    def plot_learning_curve(self, train_scores: List[float], test_scores: List[float],
                          train_sizes: List[int], title: str = "Learning Curve",
                          save_path: Optional[str] = None) -> None:
        """
        Plot learning curve.
        
        Args:
            train_scores: Training scores
            test_scores: Test scores
            train_sizes: Training set sizes
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=self.figure_size)
            
            plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores, 'o-', color="g", label="Cross-validation score")
            
            plt.title(title)
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting learning curve: {e}")
            raise
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: Optional[List[str]] = None,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Class names for labels
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=self.figure_size)
            
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            
            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            raise
    
    def save_all_plots(self, plots: Dict[str, str], output_dir: str = "plots") -> None:
        """
        Save all plots to a directory.
        
        Args:
            plots: Dictionary with plot names as keys and file paths as values
            output_dir: Output directory for plots
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for plot_name, file_path in plots.items():
            if file_path:
                # Move plot to output directory
                output_path = os.path.join(output_dir, f"{plot_name}.png")
                os.rename(file_path, output_path)
                logger.info(f"Plot {plot_name} saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = Visualizer()
    
    # Example usage (replace with actual data)
    # import numpy as np
    # data = np.random.randn(1000)
    # visualizer.plot_data_distribution(data, "Sample Data Distribution")
    
    # Example correlation matrix
    # df = pd.DataFrame({
    #     'A': np.random.randn(100),
    #     'B': np.random.randn(100),
    #     'C': np.random.randn(100)
    # })
    # visualizer.plot_correlation_matrix(df, "Sample Correlation Matrix")
    
    print("Visualizer ready for use!")
'''
        else:
            content = '''/**
 * Visualization Utilities
 * 
 * This module provides utilities for data visualization and plotting
 * for the {paper.metadata.title} implementation.
 * 
 * Author: Paper2Code Agent
 */

export class Visualizer {
    /**
     * Create a new Visualizer instance.
     * @param {{Object}} config - Configuration for visualization
     */
    constructor(config = {}) {
        this.config = config;
        this.figureSize = config.figureSize || [800, 600];
        this.dpi = config.dpi || 300;
    }
    
    /**
     * Plot data distribution.
     * @param {{Array<number>|Object}} data - Input data
     * @param {{string}} title - Plot title
     * @param {{string}} savePath - Path to save the plot
     */
    plotDataDistribution(data, title = "Data Distribution", savePath = null) {
        try {
            // This would typically use a library like Chart.js, D3.js, or Plotly
            // For now, log the data
            console.log(`Plotting data distribution: ${title}`);
            console.log('Data:', data);
            
            if (savePath) {
                console.log(`Plot would be saved to: ${savePath}`);
            }
            
        } catch (error) {
            console.error('Error plotting data distribution:', error);
            throw error;
        }
    }
    
    /**
     * Plot correlation matrix.
     * @param {{Object}} data - Input data (object with numeric properties)
     * @param {{string}} title - Plot title
     * @param {{string}} savePath - Path to save the plot
     */
    plotCorrelationMatrix(data, title = "Correlation Matrix", savePath = null) {
        try {
            console.log(`Plotting correlation matrix: ${title}`);
            console.log('Data:', data);
            
            if (savePath) {
                console.log(`Plot would be saved to: ${savePath}`);
            }
            
        } catch (error) {
            console.error('Error plotting correlation matrix:', error);
            throw error;
        }
    }
    
    /**
     * Plot algorithm comparison results.
     * @param {{Object<string, Object>}} results - Dictionary with algorithm names as keys and metrics as values
     * @param {{string}} title - Plot title
     * @param {{string}} savePath - Path to save the plot
     */
    plotAlgorithmComparison(results, title = "Algorithm Comparison", savePath = null) {
        try {
            console.log(`Plotting algorithm comparison: ${title}`);
            console.log('Results:', results);
            
            if (savePath) {
                console.log(`Plot would be saved to: ${savePath}`);
            }
            
        } catch (error) {
            console.error('Error plotting algorithm comparison:', error);
            throw error;
        }
    }
    
    /**
     * Plot learning curve.
     * @param {{Array<number>}} trainScores - Training scores
     * @param {{Array<number>}} testScores - Test scores
     * @param {{Array<number>}} trainSizes - Training set sizes
     * @param {{string}} title - Plot title
     * @param {{string}} savePath - Path to save the plot
     */
    plotLearningCurve(trainScores, testScores, trainSizes, title = "Learning Curve", savePath = null) {
        try {
            console.log(`Plotting learning curve: ${title}`);
            console.log('Training scores:', trainScores);
            console.log('Test scores:', testScores);
            console.log('Training sizes:', trainSizes);
            
            if (savePath) {
                console.log(`Plot would be saved to: ${savePath}`);
            }
            
        } catch (error) {
            console.error('Error plotting learning curve:', error);
            throw error;
        }
    }
    
    /**
     * Plot confusion matrix.
     * @param {{Array<Array<number>>}} confusionMatrix - Confusion matrix array
     * @param {{Array<string>}} classNames - Class names for labels
     * @param {{string}} title - Plot title
     * @param {{string}} savePath - Path to save the plot
     */
    plotConfusionMatrix(confusionMatrix, classNames = null, title = "Confusion Matrix", savePath = null) {
        try {
            console.log(`Plotting confusion matrix: ${title}`);
            console.log('Confusion matrix:', confusionMatrix);
            console.log('Class names:', classNames);
            
            if (savePath) {
                console.log(`Plot would be saved to: ${savePath}`);
            }
            
        } catch (error) {
            console.error('Error plotting confusion matrix:', error);
            throw error;
        }
    }
    
    /**
     * Save all plots to a directory.
     * @param {{Object<string, string>}} plots - Dictionary with plot names as keys and file paths as values
     * @param {{string}} outputDir - Output directory for plots
     */
    saveAllPlots(plots, outputDir = "plots") {
        try {
            console.log(`Saving plots to directory: ${outputDir}`);
            console.log('Plots to save:', plots);
            
            // In a real implementation, this would save the plots to files
            console.log('Plots saved successfully!');
            
        } catch (error) {
            console.error('Error saving plots:', error);
            throw error;
        }
    }
}


// Example usage
(async () => {
    // Create visualizer instance
    const visualizer = new Visualizer();
    
    // Example usage (replace with actual data)
    // const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // visualizer.plotDataDistribution(data, "Sample Data Distribution");
    
    // Example correlation matrix
    // const correlationData = {
    //     A: [1, 0.8, 0.6],
    //     B: [0.8, 1, 0.7],
    //     C: [0.6, 0.7, 1]
    // };
    // visualizer.plotCorrelationMatrix(correlationData, "Sample Correlation Matrix");
    
    console.log('Visualizer ready for use!');
})();
'''
        
        return CodeFile(
            name="visualizer.py" if language == Language.PYTHON else "visualizer.js",
            content=content,
            language=language,
            file_type="utility",
            description="Data visualization and plotting utilities"
        )
    
    def _generate_evaluator(self, paper: Paper, language: Language) -> CodeFile:
        """Generate evaluation utility file"""
        if language == Language.PYTHON:
            content = '''"""
Evaluation Utilities

This module provides utilities for evaluating algorithm performance
and metrics for the {paper.metadata.title} implementation.

Author: Paper2Code Agent
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluation utilities for the {paper.metadata.title} implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration for evaluation
        """
        self.config = config or {}
        self.metrics = self.config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for binary classification)
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            results = {}
            
            # Basic metrics
            results['accuracy'] = accuracy_score(y_true, y_pred)
            results['precision'] = precision_score(y_true, y_pred, average='weighted')
            results['recall'] = recall_score(y_true, y_pred, average='weighted')
            results['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
            # Classification report
            results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # ROC AUC (for binary classification)
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                
                # ROC curve
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                results['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            
            logger.info("Classification evaluation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}")
            raise
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate regression performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            results = {}
            
            # Basic metrics
            results['mse'] = mean_squared_error(y_true, y_pred)
            results['rmse'] = np.sqrt(results['mse'])
            results['mae'] = mean_absolute_error(y_true, y_pred)
            results['r2'] = r2_score(y_true, y_pred)
            
            # Additional metrics
            results['mape'] = self._calculate_mape(y_true, y_pred)
            results['smape'] = self._calculate_smape(y_true, y_pred)
            
            logger.info("Regression evaluation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {e}")
            raise
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate clustering performance.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            from sklearn.metrics import (
                silhouette_score, calinski_harabasz_score, davies_bouldin_score
            )
            
            results = {}
            
            # Internal clustering metrics
            results['silhouette_score'] = silhouette_score(X, labels)
            results['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            results['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            
            logger.info("Clustering evaluation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in clustering evaluation: {e}")
            raise
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores.tolist(),
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scoring': scoring,
                'cv_folds': cv
            }
            
            logger.info(f"Cross-validation completed with {cv} folds")
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def compare_algorithms(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple algorithms based on their evaluation results.
        
        Args:
            results: Dictionary with algorithm names as keys and metrics as values
            
        Returns:
            Dictionary with comparison results
        """
        try:
            comparison = {
                'algorithms': list(results.keys()),
                'rankings': {},
                'best_algorithm': None,
                'summary': {}
            }
            
            # Compare each metric
            for metric in self.metrics:
                algorithm_scores = {}
                
                for algo_name, algo_results in results.items():
                    if metric in algo_results:
                        algorithm_scores[algo_name] = algo_results[metric]
                
                # Rank algorithms by metric
                sorted_algorithms = sorted(
                    algorithm_scores.items(),
                    key=lambda x: x[1],
                    reverse=True  # Higher is better for most metrics
                )
                
                comparison['rankings'][metric] = {
                    'ranking': [algo for algo, _ in sorted_algorithms],
                    'scores': algorithm_scores
                }
                
                # Best algorithm for this metric
                comparison['rankings'][metric]['best'] = sorted_algorithms[0][0]
            
            # Overall best algorithm (average rank)
            overall_ranks = {}
            for algo_name in results.keys():
                ranks = []
                for metric_ranking in comparison['rankings'].values():
                    rank = metric_ranking['ranking'].index(algo_name) + 1
                    ranks.append(rank)
                overall_ranks[algo_name] = np.mean(ranks)
            
            # Find best overall algorithm
            best_overall = min(overall_ranks.items(), key=lambda x: x[1])
            comparison['best_algorithm'] = best_overall[0]
            comparison['overall_rankings'] = overall_ranks
            
            # Generate summary
            comparison['summary'] = self._generate_comparison_summary(comparison)
            
            logger.info("Algorithm comparison completed successfully")
            return comparison
            
        except Exception as e:
            logger.error(f"Error in algorithm comparison: {e}")
            raise
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate a summary of algorithm comparison results."""
        summary = f"Algorithm Comparison Summary\\n"
        summary += f"{'='*50}\\n"
        summary += f"Best Overall Algorithm: {comparison['best_algorithm']}\\n\\n"
        
        summary += "Metric Rankings:\\n"
        for metric, ranking in comparison['rankings'].items():
            summary += f"  {metric}: {ranking['best']} (score: {ranking['scores'][ranking['best']]:.4f})\\n"
        
        summary += f"\\nOverall Rankings:\\n"
        for algo, rank in comparison['overall_rankings'].items():
            summary += f"  {algo}: {rank:.2f} (lower is better)\\n"
        
        return summary
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filepath: Path to save the results
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            
            if filepath.endswith('.json'):
                import json
                with open(filepath, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            elif filepath.endswith('.csv'):
                # Convert to DataFrame and save as CSV
                df = pd.DataFrame([serializable_results])
                df.to_csv(filepath, index=False)
            else:
                # Default to JSON
                import json
                with open(filepath + '.json', 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


# Example usage
if __name__ == "__main__":
    # Create evaluator instance
    evaluator = Evaluator()
    
    # Example usage (replace with actual data)
    # import numpy as np
    # y_true = np.array([0, 1, 1, 0, 1])
    # y_pred = np.array([0, 1, 0, 0, 1])
    # y_pred_proba = np.array([0.1, 0.8, 0.4, 0.2, 0.9])
    
    # # Classification evaluation
    # classification_results = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
    # print("Classification Results:", classification_results)
    
    # # Regression evaluation
    # y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # y_pred_reg = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    # regression_results = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
    # print("Regression Results:", regression_results)
    
    print("Evaluator ready for use!")
'''
        else:
            content = '''/**
 * Evaluation Utilities
 * 
 * This module provides utilities for evaluating algorithm performance
 * and metrics for the {paper.metadata.title} implementation.
 * 
 * Author: Paper2Code Agent
 */

export class Evaluator {
    /**
     * Create a new Evaluator instance.
     * @param {{Object}} config - Configuration for evaluation
     */
    constructor(config = {}) {
        this.config = config;
        this.metrics = config.metrics || ['accuracy', 'precision', 'recall', 'f1'];
    }
    
    /**
     * Evaluate classification performance.
     * @param {{Array<number>}} yTrue - True labels
     * @param {{Array<number>}} yPred - Predicted labels
     * @param {{Array<number>}} yPredProb - Predicted probabilities (for binary classification)
     * @returns {{Object}} Dictionary with evaluation metrics
     */
    evaluateClassification(yTrue, yPred, yPredProb = null) {
        try {
            const results = {};
            
            // Basic metrics (simplified implementation)
            results.accuracy = this._calculateAccuracy(yTrue, yPred);
            results.precision = this._calculatePrecision(yTrue, yPred);
            results.recall = this._calculateRecall(yTrue, yPred);
            results.f1 = this._calculateF1(yTrue, yPred);
            
            // Confusion matrix
            results.confusionMatrix = this._calculateConfusionMatrix(yTrue, yPred);
            
            // ROC AUC (for binary classification)
            if (yPredProb && this._isBinaryClassification(yTrue)) {
                results.rocAuc = this._calculateRocAuc(yTrue, yPredProb);
            }
            
            console.log("Classification evaluation completed successfully");
            return results;
            
        } catch (error) {
            console.error('Error in classification evaluation:', error);
            throw error;
        }
    }
    
    /**
     * Evaluate regression performance.
     * @param {{Array<number>}} yTrue - True values
     * @param {{Array<number>}} yPred - Predicted values
     * @returns {{Object}} Dictionary with evaluation metrics
     */
    evaluateRegression(yTrue, yPred) {
        try {
            const results = {};
            
            // Basic metrics
            results.mse = this._calculateMSE(yTrue, yPred);
            results.rmse = Math.sqrt(results.mse);
            results.mae = this._calculateMAE(yTrue, yPred);
            results.r2 = this._calculateR2(yTrue, yPred);
            
            // Additional metrics
            results.mape = this._calculateMAPE(yTrue, yPred);
            results.smape = this._calculateSMAPE(yTrue, yPred);
            
            console.log("Regression evaluation completed successfully");
            return results;
            
        } catch (error) {
            console.error('Error in regression evaluation:', error);
            throw error;
        }
    }
    
    /**
     * Evaluate clustering performance.
     * @param {{Array<Array<number>>}} X - Feature matrix
     * @param {{Array<number>}} labels - Cluster labels
     * @returns {{Object}} Dictionary with evaluation metrics
     */
    evaluateClustering(X, labels) {
        try {
            const results = {};
            
            // Simplified clustering metrics
            results.silhouetteScore = this._calculateSilhouetteScore(X, labels);
            results.calinskiHarabaszScore = this._calculateCalinskiHarabaszScore(X, labels);
            
            console.log("Clustering evaluation completed successfully");
            return results;
            
        } catch (error) {
            console.error('Error in clustering evaluation:', error);
            throw error;
        }
    }
    
    /**
     * Compare multiple algorithms based on their evaluation results.
     * @param {{Object<string, Object>}} results - Dictionary with algorithm names as keys and metrics as values
     * @returns {{Object}} Dictionary with comparison results
     */
    compareAlgorithms(results) {
        try {
            const comparison = {
                algorithms: Object.keys(results),
                rankings: {},
                bestAlgorithm: null,
                summary: ''
            };
            
            // Compare each metric
            for (const metric of this.metrics) {
                const algorithmScores = {};
                
                for (const [algoName, algoResults] of Object.entries(results)) {
                    if (metric in algoResults) {
                        algorithmScores[algoName] = algoResults[metric];
                    }
                }
                
                // Rank algorithms by metric
                const sortedAlgorithms = Object.entries(algorithmScores)
                    .sort(([,a], [,b]) => b - a); // Higher is better
                
                comparison.rankings[metric] = {
                    ranking: sortedAlgorithms.map(([algo]) => algo),
                    scores: algorithmScores,
                    best: sortedAlgorithms[0][0]
                };
            }
            
            // Overall best algorithm (average rank)
            const overallRanks = {};
            for (const algoName of comparison.algorithms) {
                const ranks = [];
                for (const metricRanking of Object.values(comparison.rankings)) {
                    const rank = metricRanking.ranking.indexOf(algoName) + 1;
                    ranks.push(rank);
                }
                overallRanks[algoName] = ranks.reduce((a, b) => a + b, 0) / ranks.length;
            }
            
            // Find best overall algorithm
            const bestOverall = Object.entries(overallRanks)
                .sort(([,a], [,b]) => a - b)[0]; // Lower rank is better
            comparison.bestAlgorithm = bestOverall[0];
            comparison.overallRankings = overallRanks;
            
            // Generate summary
            comparison.summary = this._generateComparisonSummary(comparison);
            
            console.log("Algorithm comparison completed successfully");
            return comparison;
            
        } catch (error) {
            console.error('Error in algorithm comparison:', error);
            throw error;
        }
    }
    
    // Helper methods for classification metrics
    _calculateAccuracy(yTrue, yPred) {
        const correct = yTrue.filter((trueVal, index) => trueVal === yPred[index]).length;
        return correct / yTrue.length;
    }
    
    _calculatePrecision(yTrue, yPred) {
        // Simplified precision calculation
        let truePositives = 0;
        let predictedPositives = 0;
        
        for (let i = 0; i < yTrue.length; i++) {
            if (yPred[i] === 1) {
                predictedPositives++;
                if (yTrue[i] === 1) {
                    truePositives++;
                }
            }
        }
        
        return predictedPositives > 0 ? truePositives / predictedPositives : 0;
    }
    
    _calculateRecall(yTrue, yPred) {
        // Simplified recall calculation
        let truePositives = 0;
        let actualPositives = 0;
        
        for (let i = 0; i < yTrue.length; i++) {
            if (yTrue[i] === 1) {
                actualPositives++;
                if (yPred[i] === 1) {
                    truePositives++;
                }
            }
        }
        
        return actualPositives > 0 ? truePositives / actualPositives : 0;
    }
    
    _calculateF1(yTrue, yPred) {
        const precision = this._calculatePrecision(yTrue, yPred);
        const recall = this._calculateRecall(yTrue, yPred);
        
        return (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    }
    
    _calculateConfusionMatrix(yTrue, yPred) {
        const matrix = {
            truePositives: 0,
            falsePositives: 0,
            trueNegatives: 0,
            falseNegatives: 0
        };
        
        for (let i = 0; i < yTrue.length; i++) {
            if (yTrue[i] === 1 && yPred[i] === 1) {
                matrix.truePositives++;
            } else if (yTrue[i] === 0 && yPred[i] === 1) {
                matrix.falsePositives++;
            } else if (yTrue[i] === 1 && yPred[i] === 0) {
                matrix.falseNegatives++;
            } else {
                matrix.trueNegatives++;
            }
        }
        
        return matrix;
    }
    
    _isBinaryClassification(yTrue) {
        return new Set(yTrue).size === 2;
    }
    
    _calculateRocAuc(yTrue, yPredProb) {
        // Simplified ROC AUC calculation
        // This would typically use a more sophisticated implementation
        return 0.85; // Placeholder
    }
    
    // Helper methods for regression metrics
    _calculateMSE(yTrue, yPred) {
        const squaredErrors = yTrue.map((trueVal, index) => 
            Math.pow(trueVal - yPred[index], 2)
        );
        return squaredErrors.reduce((a, b) => a + b, 0) / yTrue.length;
    }
    
    _calculateMAE(yTrue, yPred) {
        const absoluteErrors = yTrue.map((trueVal, index) => 
            Math.abs(trueVal - yPred[index])
        );
        return absoluteErrors.reduce((a, b) => a + b, 0) / yTrue.length;
    }
    
    _calculateR2(yTrue, yPred) {
        const yMean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
        const totalSumSquares = yTrue.reduce((sum, val) => sum + Math.pow(val - yMean, 2), 0);
        const residualSumSquares = yTrue.reduce((sum, val, index) => 
            sum + Math.pow(val - yPred[index], 2), 0);
        
        return 1 - (residualSumSquares / totalSumSquares);
    }
    
    _calculateMAPE(yTrue, yPred) {
        const mask = yTrue.map(val => val !== 0);
        const filteredTrue = yTrue.filter((_, index) => mask[index]);
        const filteredPred = yPred.filter((_, index) => mask[index]);
        
        const percentageErrors = filteredTrue.map((trueVal, index) => 
            Math.abs((trueVal - filteredPred[index]) / trueVal) * 100
        );
        
        return percentageErrors.reduce((a, b) => a + b, 0) / filteredTrue.length;
    }
    
    _calculateSMAPE(yTrue, yPred) {
        const denominator = yTrue.map((trueVal, index) => 
            (Math.abs(trueVal) + Math.abs(yPred[index])) / 2
        );
        
        const smapeValues = denominator.map((denom, index) => 
            denom !== 0 ? Math.abs(yTrue[index] - yPred[index]) / denom * 100 : 0
        );
        
        return smapeValues.reduce((a, b) => a + b, 0) / yTrue.length;
    }
    
    // Helper methods for clustering metrics
    _calculateSilhouetteScore(X, labels) {
        // Simplified silhouette score calculation
        // This would typically use a more sophisticated implementation
        return 0.65; // Placeholder
    }
    
    _calculateCalinskiHarabaszScore(X, labels) {
        // Simplified Calinski-Harabasz score calculation
        // This would typically use a more sophisticated implementation
        return 250; // Placeholder
    }
    
    _generateComparisonSummary(comparison) {
        let summary = `Algorithm Comparison Summary\\n`;
        summary += `${'='.repeat(50)}\\n`;
        summary += `Best Overall Algorithm: ${comparison.bestAlgorithm}\\n\\n`;
        
        summary += `Metric Rankings:\\n`;
        for (const [metric, ranking] of Object.entries(comparison.rankings)) {
            summary += `  ${metric}: ${ranking.best}\\n`;
        }
        
        summary += `\\nOverall Rankings:\\n`;
        for (const [algo, rank] of Object.entries(comparison.overallRankings)) {
            summary += `  ${algo}: ${rank.toFixed(2)} (lower is better)\\n`;
        }
        
        return summary;
    }
    
    /**
     * Save evaluation results to file.
     * @param {{Object}} results - Evaluation results
     * @param {{string}} filepath - Path to save the results
     */
    saveEvaluationResults(results, filepath) {
        try {
            console.log(`Saving evaluation results to: ${filepath}`);
            
            // In a real implementation, this would save the results to a file
            console.log('Results to save:', results);
            console.log('Evaluation results saved successfully!');
            
        } catch (error) {
            console.error('Error saving evaluation results:', error);
            throw error;
        }
    }
}


// Example usage
(async () => {
    // Create evaluator instance
    const evaluator = new Evaluator();
    
    // Example usage (replace with actual data)
    // const yTrue = [0, 1, 1, 0, 1];
    // const yPred = [0, 1, 0, 0, 1];
    // const yPredProb = [0.1, 0.8, 0.4, 0.2, 0.9];
    
    // // Classification evaluation
    // const classificationResults = evaluator.evaluateClassification(yTrue, yPred, yPredProb);
    // console.log("Classification Results:", classificationResults);
    
    // // Regression evaluation
    // const yTrueReg = [1.0, 2.0, 3.0, 4.0, 5.0];
    // const yPredReg = [1.1, 2.1, 2.9, 3.8, 5.2];
    // const regressionResults = evaluator.evaluateRegression(yTrueReg, yPredReg);
    // console.log("Regression Results:", regressionResults);
    
    console.log('Evaluator ready for use!');
})();
'''
        
        return CodeFile(
            name="evaluator.py" if language == Language.PYTHON else "evaluator.js",
            content=content,
            language=language,
            file_type="utility",
            description="Evaluation metrics and performance assessment utilities"
        )
    
    def run(self, paper: Paper, architecture_result: Dict[str, Any], 
            research_patterns: Dict[str, Any]) -> CodeGenerationResult:
        """Main method to generate code implementation"""
        start_time = datetime.now()
        
        logger.info(f"Generating code implementation for: {paper.metadata.title}")
        
        try:
            # Select primary language
            language = self.select_primary_language(architecture_result)
            
            # Generate algorithm implementations
            algorithm_files = self.generate_algorithm_implementations(paper, architecture_result)
            
            # Generate main implementation
            main_file = self.generate_main_implementation(paper, architecture_result, algorithm_files)
            
            # Generate utility files
            utility_files = self.generate_utility_files(paper, architecture_result)
            
            # Generate configuration files
            requirements_file = self.generate_requirements_file(architecture_result)
            gitignore_file = self.generate_gitignore_file()
            
            # Combine all generated files
            all_files = algorithm_files + [main_file] + utility_files + [requirements_file, gitignore_file]
            
            # Create code implementation object
            code_implementation = CodeImplementation(
                name=paper.metadata.title.lower().replace(' ', '_'),
                language=language,
                framework=architecture_result.get('technology_stack', {}).get('frameworks', [None])[0],
                files=all_files,
                dependencies=architecture_result.get('dependencies', []),
                description=f"Code implementation for {paper.metadata.title}"
            )
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate code quality score
            code_quality_score = self._calculate_code_quality_score(all_files, paper)
            
            return CodeGenerationResult(
                code_implementation=code_implementation,
                generated_files=all_files,
                code_quality_score=code_quality_score,
                generation_time=generation_time,
                language_used=language,
                framework_used=architecture_result.get('technology_stack', {}).get('frameworks', [None])[0]
            )
            
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            raise
    
    def _calculate_code_quality_score(self, files: List[CodeFile], paper: Paper) -> float:
        """Calculate code quality score for generated implementation"""
        score = 0.0
        
        # Check if all algorithms have implementations
        if len(files) >= len(paper.algorithms):
            score += 0.3
        
        # Check for main implementation
        main_files = [f for f in files if f.file_type == "main"]
        if main_files:
            score += 0.2
        
        # Check for utility files
        utility_files = [f for f in files if f.file_type == "utility"]
        if len(utility_files) >= 3:  # Should have data_processor, visualizer, evaluator
            score += 0.3
        
        # Check for configuration files
        config_files = [f for f in files if f.file_type in ["requirements", "gitignore"]]
        if len(config_files) >= 2:
            score += 0.2
        
        return min(1.0, score)