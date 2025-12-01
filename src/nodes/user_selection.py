"""User Selection Node - Handles user selections from itineraries"""

from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.utils.logger import setup_logger

logger = setup_logger("user_selection")


class UserSelectionNode:
    """Node for handling user selections"""

    def __init__(self, mem0_manager: Mem0Manager):
        """
        Initialize user selection node

        Args:
            mem0_manager: Mem0 manager instance
        """
        self.mem0_manager = mem0_manager

    def save_selection(self, user_id: str, selection: str) -> str:
        """
        Save user selection to history

        Args:
            user_id: User identifier
            selection: User's selection

        Returns:
            Confirmation message
        """
        try:
            logger.info(f"Saving selection for user {user_id}")

            # Save to Mem0
            self.mem0_manager.add_memory(
                user_id=user_id, message=selection, metadata={"type": "selection"}
            )

            response = (
                f"Great choice! I've saved your selection: {selection}\n\n"
                "This will be used for your future travel planning. "
                "Would you like to proceed with booking or make any changes?"
            )

            logger.info(f"Successfully saved selection for user {user_id}")
            return response

        except Exception as e:
            logger.error(f"Error saving selection: {e}")
            return "I encountered an error saving your selection. Please try again."
