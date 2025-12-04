"""
Scenario Test Runner

This script runs all scenario tests for the Paper2Code agent system.
"""

import sys
import os
import logging
from pathlib import Path

# Add the app directory to the Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_scenarios():
    """Run all scenario tests"""
    try:
        # Import pytest
        import pytest
        
        # Get the scenario test directory
        scenario_dir = Path(__file__).parent
        
        # Run all scenario tests
        logger.info("Running Paper2Code scenario tests...")
        
        # Run pytest with verbose output
        exit_code = pytest.main([
            str(scenario_dir),
            "-v",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings"
        ])
        
        if exit_code == 0:
            logger.info("✅ All scenario tests passed!")
        else:
            logger.error("❌ Some scenario tests failed!")
        
        return exit_code
        
    except ImportError as e:
        logger.error(f"Error importing pytest: {e}")
        logger.error("Please install pytest: pip install pytest")
        return 1
    except Exception as e:
        logger.error(f"Error running scenarios: {e}")
        return 1

def run_specific_scenario(scenario_name):
    """Run a specific scenario test"""
    try:
        import pytest
        
        scenario_dir = Path(__file__).parent
        scenario_file = scenario_dir / f"{scenario_name}.test.py"
        
        if not scenario_file.exists():
            logger.error(f"Scenario file not found: {scenario_file}")
            return 1
        
        logger.info(f"Running specific scenario: {scenario_name}")
        
        exit_code = pytest.main([
            str(scenario_file),
            "-v",
            "--tb=short",
            f"-k {scenario_name}"
        ])
        
        if exit_code == 0:
            logger.info(f"✅ Scenario '{scenario_name}' passed!")
        else:
            logger.error(f"❌ Scenario '{scenario_name}' failed!")
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Error running scenario {scenario_name}: {e}")
        return 1

def list_scenarios():
    """List all available scenario tests"""
    scenario_dir = Path(__file__).parent
    
    scenario_files = list(scenario_dir.glob("*.test.py"))
    
    if not scenario_files:
        logger.info("No scenario tests found.")
        return
    
    logger.info("Available scenario tests:")
    for scenario_file in scenario_files:
        scenario_name = scenario_file.stem.replace(".test", "")
        logger.info(f"  - {scenario_name}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Paper2Code scenario tests")
    parser.add_argument("scenario", nargs="?", help="Specific scenario to run")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list:
        list_scenarios()
        return 0
    
    if args.scenario:
        return run_specific_scenario(args.scenario)
    else:
        return run_scenarios()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)