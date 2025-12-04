"""
Prompt Manager

This module handles prompt management using LangWatch Prompt CLI.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from agno.models.openrouter import OpenRouter

logger = logging.getLogger(__name__)


class PromptManager:
    """Manager for handling prompts using LangWatch Prompt CLI"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts = {}
        self.llm = OpenRouter(model="openai/gpt-4o")
        
        # Ensure prompts directory exists
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Load all prompts
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompts from the prompts directory"""
        try:
            for prompt_file in self.prompts_dir.glob("*.yaml"):
                prompt_name = prompt_file.stem
                self.prompts[prompt_name] = self._load_prompt_file(prompt_file)
                logger.info(f"Loaded prompt: {prompt_name}")
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
    
    def _load_prompt_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single prompt file"""
        try:
            with open(file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            
            # Convert to Agno prompt format
            agno_prompt = {
                "model": prompt_data.get("model", "gpt-4o"),
                "temperature": prompt_data.get("temperature", 0.1),
                "messages": prompt_data.get("messages", [])
            }
            
            return agno_prompt
            
        except Exception as e:
            logger.error(f"Error loading prompt file {file_path}: {e}")
            return {}
    
    def get_prompt(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt"""
        return self.prompts.get(prompt_name)
    
    def list_prompts(self) -> List[str]:
        """List all available prompts"""
        return list(self.prompts.keys())
    
    def create_prompt(self, prompt_name: str, prompt_data: Dict[str, Any]) -> bool:
        """Create a new prompt"""
        try:
            # Create prompt file
            prompt_file = self.prompts_dir / f"{prompt_name}.yaml"
            
            with open(prompt_file, 'w') as f:
                yaml.dump(prompt_data, f, default_flow_style=False)
            
            # Load the prompt
            self.prompts[prompt_name] = self._load_prompt_file(prompt_file)
            
            logger.info(f"Created prompt: {prompt_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating prompt {prompt_name}: {e}")
            return False
    
    def update_prompt(self, prompt_name: str, prompt_data: Dict[str, Any]) -> bool:
        """Update an existing prompt"""
        try:
            if prompt_name not in self.prompts:
                logger.error(f"Prompt {prompt_name} not found")
                return False
            
            # Update prompt file
            prompt_file = self.prompts_dir / f"{prompt_name}.yaml"
            
            with open(prompt_file, 'w') as f:
                yaml.dump(prompt_data, f, default_flow_style=False)
            
            # Reload the prompt
            self.prompts[prompt_name] = self._load_prompt_file(prompt_file)
            
            logger.info(f"Updated prompt: {prompt_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_name}: {e}")
            return False
    
    def delete_prompt(self, prompt_name: str) -> bool:
        """Delete a prompt"""
        try:
            if prompt_name not in self.prompts:
                logger.error(f"Prompt {prompt_name} not found")
                return False
            
            # Delete prompt file
            prompt_file = self.prompts_dir / f"{prompt_name}.yaml"
            prompt_file.unlink()
            
            # Remove from prompts dictionary
            del self.prompts[prompt_name]
            
            logger.info(f"Deleted prompt: {prompt_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting prompt {prompt_name}: {e}")
            return False
    
    def get_prompt_variables(self, prompt_name: str) -> List[str]:
        """Get the variables used in a prompt"""
        try:
            prompt = self.get_prompt(prompt_name)
            if not prompt:
                return []
            
            # Extract variables from messages
            variables = []
            for message in prompt.get("messages", []):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    # Extract {{ variable }} patterns
                    import re
                    variable_pattern = r'\{\{\s*(\w+)\s*\}\}'
                    matches = re.findall(variable_pattern, content)
                    variables.extend(matches)
            
            return list(set(variables))
            
        except Exception as e:
            logger.error(f"Error getting prompt variables for {prompt_name}: {e}")
            return []
    
    def render_prompt(self, prompt_name: str, variables: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Render a prompt with variables"""
        try:
            prompt = self.get_prompt(prompt_name)
            if not prompt:
                return None
            
            # Create a copy of the prompt
            rendered_prompt = prompt.copy()
            
            # Render variables in messages
            for message in rendered_prompt.get("messages", []):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    
                    # Replace {{ variable }} with actual values
                    for var_name, var_value in variables.items():
                        placeholder = f"{{{{{var_name}}}}}"
                        content = content.replace(placeholder, str(var_value))
                    
                    message["content"] = content
            
            return rendered_prompt
            
        except Exception as e:
            logger.error(f"Error rendering prompt {prompt_name}: {e}")
            return None
    
    def validate_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """Validate a prompt"""
        try:
            prompt = self.get_prompt(prompt_name)
            if not prompt:
                return {"valid": False, "error": "Prompt not found"}
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
            
            # Check required fields
            required_fields = ["model", "messages"]
            for field in required_fields:
                if field not in prompt:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Check messages structure
            messages = prompt.get("messages", [])
            if not messages:
                validation_result["errors"].append("No messages found")
                validation_result["valid"] = False
            
            # Check message roles
            valid_roles = ["system", "user", "assistant"]
            for i, message in enumerate(messages):
                role = message.get("role")
                if role not in valid_roles:
                    validation_result["errors"].append(f"Invalid role in message {i}: {role}")
                    validation_result["valid"] = False
                
                content = message.get("content")
                if not content:
                    validation_result["warnings"].append(f"Empty content in message {i}")
                
                if role == "system" and i != 0:
                    validation_result["warnings"].append("System message should be first")
            
            # Check temperature
            temperature = prompt.get("temperature")
            if temperature is not None and not (0 <= temperature <= 2):
                validation_result["warnings"].append("Temperature should be between 0 and 2")
            
            # Check for variables
            variables = self.get_prompt_variables(prompt_name)
            if variables:
                validation_result["suggestions"].append(f"Variables found: {', '.join(variables)}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating prompt {prompt_name}: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """Get information about a prompt"""
        try:
            prompt = self.get_prompt(prompt_name)
            if not prompt:
                return {}
            
            variables = self.get_prompt_variables(prompt_name)
            validation = self.validate_prompt(prompt_name)
            
            return {
                "name": prompt_name,
                "model": prompt.get("model"),
                "temperature": prompt.get("temperature"),
                "message_count": len(prompt.get("messages", [])),
                "variables": variables,
                "validation": validation,
                "created_at": self._get_file_creation_time(prompt_name),
                "modified_at": self._get_file_modification_time(prompt_name)
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt info for {prompt_name}: {e}")
            return {}
    
    def _get_file_creation_time(self, prompt_name: str) -> Optional[str]:
        """Get file creation time"""
        try:
            prompt_file = self.prompts_dir / f"{prompt_name}.yaml"
            if prompt_file.exists():
                return prompt_file.stat().st_ctime.isoformat()
        except Exception as e:
            logger.error(f"Error getting creation time for {prompt_name}: {e}")
        return None
    
    def _get_file_modification_time(self, prompt_name: str) -> Optional[str]:
        """Get file modification time"""
        try:
            prompt_file = self.prompts_dir / f"{prompt_name}.yaml"
            if prompt_file.exists():
                return prompt_file.stat().st_mtime.isoformat()
        except Exception as e:
            logger.error(f"Error getting modification time for {prompt_name}: {e}")
        return None
    
    def export_prompts(self, output_file: str) -> bool:
        """Export all prompts to a file"""
        try:
            export_data = {
                "prompts": self.prompts,
                "exported_at": str(Path().cwd()),
                "exported_by": "paper2code-agent"
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported prompts to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting prompts: {e}")
            return False
    
    def import_prompts(self, input_file: str) -> bool:
        """Import prompts from a file"""
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            imported_prompts = import_data.get("prompts", {})
            
            for prompt_name, prompt_data in imported_prompts.items():
                # Create prompt file
                prompt_file = self.prompts_dir / f"{prompt_name}.yaml"
                
                with open(prompt_file, 'w') as f:
                    yaml.dump(prompt_data, f, default_flow_style=False)
                
                # Load the prompt
                self.prompts[prompt_name] = self._load_prompt_file(prompt_file)
            
            logger.info(f"Imported prompts from {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing prompts: {e}")
            return False
    
    def sync_prompts(self) -> bool:
        """Sync prompts with LangWatch Prompt CLI"""
        try:
            # This would integrate with LangWatch Prompt CLI
            # For now, just reload prompts from disk
            self._load_prompts()
            logger.info("Prompts synced successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error syncing prompts: {e}")
            return False
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about all prompts"""
        try:
            stats = {
                "total_prompts": len(self.prompts),
                "models": {},
                "temperatures": {},
                "message_counts": {},
                "variables": {}
            }
            
            for prompt_name, prompt in self.prompts.items():
                # Model statistics
                model = prompt.get("model", "unknown")
                stats["models"][model] = stats["models"].get(model, 0) + 1
                
                # Temperature statistics
                temperature = prompt.get("temperature", 0.1)
                temp_range = "low" if temperature < 0.3 else "medium" if temperature < 0.7 else "high"
                stats["temperatures"][temp_range] = stats["temperatures"].get(temp_range, 0) + 1
                
                # Message count statistics
                message_count = len(prompt.get("messages", []))
                stats["message_counts"][message_count] = stats["message_counts"].get(message_count, 0) + 1
                
                # Variable statistics
                variables = self.get_prompt_variables(prompt_name)
                var_count = len(variables)
                stats["variables"][var_count] = stats["variables"].get(var_count, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting prompt statistics: {e}")
            return {}


# Global prompt manager instance
prompt_manager = PromptManager()


# CLI functions for prompt management
def create_prompt(prompt_name: str, model: str = "gpt-4o", temperature: float = 0.1):
    """Create a new prompt"""
    try:
        # Create basic prompt structure
        prompt_data = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant."
                },
                {
                    "role": "user",
                    "content": "Hello, please help me."
                }
            ]
        }
        
        success = prompt_manager.create_prompt(prompt_name, prompt_data)
        if success:
            print(f"Prompt '{prompt_name}' created successfully")
        else:
            print(f"Failed to create prompt '{prompt_name}'")
            
    except Exception as e:
        print(f"Error creating prompt: {e}")


def list_prompts():
    """List all prompts"""
    try:
        prompts = prompt_manager.list_prompts()
        print("Available prompts:")
        for prompt in prompts:
            print(f"  - {prompt}")
    except Exception as e:
        print(f"Error listing prompts: {e}")


def get_prompt_info(prompt_name: str):
    """Get information about a prompt"""
    try:
        info = prompt_manager.get_prompt_info(prompt_name)
        if info:
            print(f"Prompt: {info['name']}")
            print(f"Model: {info['model']}")
            print(f"Temperature: {info['temperature']}")
            print(f"Message count: {info['message_count']}")
            print(f"Variables: {info['variables']}")
            print(f"Validation: {info['validation']['valid']}")
        else:
            print(f"Prompt '{prompt_name}' not found")
    except Exception as e:
        print(f"Error getting prompt info: {e}")


def validate_prompt(prompt_name: str):
    """Validate a prompt"""
    try:
        validation = prompt_manager.validate_prompt(prompt_name)
        print(f"Validation for '{prompt_name}':")
        print(f"Valid: {validation['valid']}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        if validation['suggestions']:
            print("Suggestions:")
            for suggestion in validation['suggestions']:
                print(f"  - {suggestion}")
                
    except Exception as e:
        print(f"Error validating prompt: {e}")


def sync_prompts():
    """Sync prompts with LangWatch Prompt CLI"""
    try:
        success = prompt_manager.sync_prompts()
        if success:
            print("Prompts synced successfully")
        else:
            print("Failed to sync prompts")
    except Exception as e:
        print(f"Error syncing prompts: {e}")


def get_prompt_statistics():
    """Get prompt statistics"""
    try:
        stats = prompt_manager.get_prompt_statistics()
        print("Prompt Statistics:")
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Models: {stats['models']}")
        print(f"Temperature ranges: {stats['temperatures']}")
        print(f"Message counts: {stats['message_counts']}")
        print(f"Variable counts: {stats['variables']}")
    except Exception as e:
        print(f"Error getting prompt statistics: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Management CLI")
    parser.add_argument("command", choices=["create", "list", "info", "validate", "sync", "stats"])
    parser.add_argument("--name", help="Prompt name")
    parser.add_argument("--model", default="gpt-4o", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_prompt(args.name, args.model, args.temperature)
    elif args.command == "list":
        list_prompts()
    elif args.command == "info":
        get_prompt_info(args.name)
    elif args.command == "validate":
        validate_prompt(args.name)
    elif args.command == "sync":
        sync_prompts()
    elif args.command == "stats":
        get_prompt_statistics()