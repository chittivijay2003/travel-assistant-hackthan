"""User Input Node - Loads user history and prepares state for intent classification"""

from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.utils.logger import setup_logger

logger = setup_logger("user_input")


class UserInputNode:
    """Node for handling user input and loading history"""

    def __init__(self, mem0_manager: Mem0Manager):
        """
        Initialize user input node

        Args:
            mem0_manager: Mem0 manager instance for history retrieval
        """
        self.mem0_manager = mem0_manager

    def __call__(self, state: GraphState) -> GraphState:
        """
        Load user history from database and prepare state

        Args:
            state: Current graph state with user_input and user_id

        Returns:
            Updated state with user_history loaded
        """
        try:
            user_id = state["user_id"]
            logger.info(f"Loading user history for user {user_id}")

            # Load user history from Mem0
            memories = self.mem0_manager.get_memories(user_id, limit=20)

            # Format user history for easy access
            user_history = []
            for mem in memories:
                history_item = {
                    "content": mem.get("memory", mem.get("content", "")),
                    "metadata": mem.get("metadata", {}),
                    "timestamp": mem.get("created_at", ""),
                }
                user_history.append(history_item)

            state["user_history"] = user_history

            logger.info(f"Loaded {len(user_history)} history items for user {user_id}")

            # Log user input
            logger.info(f"User input: {state['user_input']}")

            state["error"] = None

        except Exception as e:
            logger.error(f"Error loading user history: {e}")
            state["user_history"] = []
            state["error"] = f"User history loading error: {str(e)}"

        return state
