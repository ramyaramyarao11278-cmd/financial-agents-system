from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """Base class for all financial agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given input data.
        
        Args:
            input_data: Dictionary containing input data for the agent.
            
        Returns:
            Dictionary containing the agent's output.
        """
        pass
    
    def get_info(self) -> Dict[str, str]:
        """
        Get basic information about the agent.
        
        Returns:
            Dictionary containing agent information.
        """
        return {
            "name": self.name,
            "description": self.description
        }
    
    def _validate_input(self, input_data: Dict[str, Any], required_keys: list) -> bool:
        """
        Validate that the input data contains all required keys.
        
        Args:
            input_data: Dictionary containing input data.
            required_keys: List of required keys.
            
        Returns:
            True if all required keys are present, False otherwise.
        """
        return all(key in input_data for key in required_keys)
    
    def _format_output(self, 
                      status: str, 
                      result: Any, 
                      message: Optional[str] = None) -> Dict[str, Any]:
        """
        Format the agent's output in a consistent way.
        
        Args:
            status: Status of the agent's execution ("success" or "error").
            result: The agent's result data.
            message: Optional message.
            
        Returns:
            Formatted output dictionary.
        """
        return {
            "agent": self.name,
            "status": status,
            "result": result,
            "message": message
        }
