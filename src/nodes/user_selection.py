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

    def __call__(self, state: GraphState) -> GraphState:
        """
        Process user selection from itinerary

        Args:
            state: Current graph state

        Returns:
            Updated state with selection saved
        """
        try:
            user_id = state["user_id"]
            user_input = state["user_input"]

            # Get conversation history
            conversation_history = state.get("conversation_history", [])

            # Find the last assistant message (should be the itinerary)
            last_itinerary = None
            if conversation_history:
                for msg in reversed(conversation_history):
                    if msg["role"] == "assistant":
                        last_itinerary = msg["content"]
                        break

            # Check if user is making a selection
            confirmation_words = [
                "yes",
                "sure",
                "okay",
                "go ahead",
                "proceed",
                "i want this",
                "sounds good",
                "perfect",
                "i'll take",
            ]

            is_confirmation = any(
                word in user_input.lower() for word in confirmation_words
            )

            if is_confirmation and last_itinerary:
                # Save the selection
                logger.info(f"Saving itinerary selection for user {user_id}")
                self.mem0_manager.add_memory(
                    user_id=user_id,
                    message=f"Selected itinerary: {last_itinerary[:500]}",
                    metadata={"type": "selection", "source": "itinerary"},
                )

                # Mark that user made a selection
                state["user_selections"] = last_itinerary[:500]

                response = (
                    "✅ Great! I've saved your itinerary selection.\n\n"
                    "What would you like to do next?\n\n"
                    "**Option 1:** Create a detailed travel plan with flights and cabs\n"
                    "**Option 2:** Get in-trip support (lounges, food, accessories)\n"
                    "**Option 3:** Modify the itinerary\n\n"
                    "Just tell me what you'd like!"
                )
            else:
                # User might be asking a question or making a comment
                response = (
                    "I've noted your feedback. Would you like to:\n"
                    "- Proceed with this itinerary?\n"
                    "- Request changes to the itinerary?\n"
                    "- Create a complete travel plan?"
                )

            state["response"] = response
            state["error"] = None

            logger.info(f"User selection processed for user {user_id}")

        except Exception as e:
            logger.error(f"Error in user selection node: {e}")
            state[
                "response"
            ] = "I encountered an error saving your selection. Please try again."
            state["error"] = str(e)

        return state
