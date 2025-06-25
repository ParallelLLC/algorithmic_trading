import abc
import logging
from typing import Dict, Any

class Agent(abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

    @abc.abstractmethod
    def act(self, data: Any) -> Dict[str, Any]:
        """Produce an action given input data"""
        pass
    
    def log_action(self, action: Dict[str, Any]) -> None:
        """Log the action taken by the agent"""
        self.logger.info(f"Action taken: {action}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log errors with context"""
        self.logger.error(f"Error in {self.__class__.__name__}: {error}", 
                         exc_info=True, extra={'context': context})
