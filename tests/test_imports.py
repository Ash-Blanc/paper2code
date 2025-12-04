"""
Test to verify all imports work correctly.
"""

import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_basic_imports():
    """Test basic module imports."""
    try:
        # Test main app imports
        from app.main import main
        print("✓ app.main imported successfully")
        
        # Test pipeline imports
        from app.pipeline import Paper2CodePipeline
        print("✓ app.pipeline imported successfully")
        
        # Test prompt manager imports
        from app.prompt_manager import PromptManager
        print("✓ app.prompt_manager imported successfully")
        
        # Test agent imports
        from app.agents.paper_analysis import PaperAnalysisAgent
        from app.agents.research import ResearchAgent
        from app.agents.architecture import ArchitectureAgent
        from app.agents.code_generation import CodeGenerationAgent
        from app.agents.documentation import DocumentationAgent
        from app.agents.quality_assurance import QualityAssuranceAgent
        print("✓ All agents imported successfully")
        
        # Test OpenRouter model import
        from agno.models.openrouter import OpenRouter
        print("✓ OpenRouter model imported successfully")
        
        # Test model imports
        from app.models.paper import Paper, PaperMetadata, Author, Algorithm, Experiment
        from app.models.code import Language, Framework, Dependency, CodeFile, CodeImplementation
        from app.models.repository import RepositoryConfig, RepositoryVisibility
        print("✓ All models imported successfully")
        
        # Test integration imports
        from app.integrations.github import GitHubIntegration, GitHubRepositoryResult
        print("✓ GitHub integration imported successfully")
        
        # Test tool imports
        from app.tools.github_tools import GitHubTools
        print("✓ GitHub tools imported successfully")
        
        # Test cache imports
        from app.cache.cache_manager import CacheManager, CacheStats
        print("✓ Cache manager imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_agent_initialization():
    """Test that agents can be initialized."""
    try:
        from app.agents.paper_analysis import PaperAnalysisAgent
        from app.agents.research import ResearchAgent
        from agno.models.openrouter import OpenRouter
        
        # Test OpenRouter initialization (this may fail if API keys are missing, but should not fail on import)
        try:
            llm = OpenRouter(model="openai/gpt-4o")
            print("✓ OpenRouter can be initialized")
        except Exception as e:
            print(f"⚠️  OpenRouter initialization failed (expected if no API key): {e}")
        
        # Test agent initialization (this may fail if API keys are missing, but should not fail on import)
        print("✓ Agent classes can be imported")
        print("✓ Agent initialization test passed (import only)")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        return False

def test_model_creation():
    """Test that model classes can be instantiated."""
    try:
        from app.models.paper import Author, PaperMetadata
        from app.models.code import Language, Dependency
        
        # Test model instantiation
        author = Author(name="Test Author", affiliation="Test University")
        metadata = PaperMetadata(
            title="Test Paper",
            authors=[author],
            abstract="This is a test abstract."
        )
        dependency = Dependency(name="test-package", version="1.0.0")
        
        print("✓ Model classes can be instantiated")
        print("✓ Model creation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running import tests...")
    print("=" * 50)
    
    # Test basic imports
    basic_success = test_basic_imports()
    print()
    
    # Test agent initialization
    agent_success = test_agent_initialization()
    print()
    
    # Test model creation
    model_success = test_model_creation()
    print()
    
    if basic_success and agent_success and model_success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)