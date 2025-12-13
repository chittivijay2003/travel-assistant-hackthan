"""User Input Node - Loads user history and prepares state for intent classification"""

from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.utils.guardrails import Guardrails
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import LangFuseTracer, is_langfuse_enabled

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
            txnid = state.get("txnid")

            with LangFuseTracer("user_input_validation", session_id=txnid) as tracer:
                # Add trace metadata
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["user_id"] = user_id
                    tracer.metadata["input_length"] = len(state["user_input"])

                # LAYER 1: Validate user input with guardrails (PII + Injection + NeMo)
                # NeMo response is used ONLY for safety validation, not for final output
                # If NeMo responds normally → input is safe, continue to workflow
                # If NeMo refuses → input is unsafe, block and return error
                logger.info(f"Validating user input for user {user_id}")
                validation_result = Guardrails.validate(
                    text=state["user_input"],
                    use_nemo=True,  # Enable NeMo for testing (set to False in production)
                    check_injection=True,  # Check for prompt injection patterns
                )

                # Add validation result to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["validation_safe"] = validation_result["safe"]
                    tracer.metadata["validation_blocked"] = validation_result["blocked"]
                    tracer.metadata["validation_reasons"] = validation_result["reasons"]

                # If input is blocked, set error response and end workflow
                if not validation_result["safe"]:
                    logger.warning(
                        f"User input blocked: {validation_result['reasons']}"
                    )
                    state["response"] = validation_result["text"]  # Error message
                    state["next_node"] = "end"  # Skip to end
                    state[
                        "error"
                    ] = f"Input validation failed: {validation_result['details']}"

                    if tracer.trace and is_langfuse_enabled():
                        tracer.metadata["blocked_message"] = validation_result["text"]

                    return state

                # Input is safe - continue with normal flow
                logger.info(f"User input validated successfully for user {user_id}")

                # Load user history from Mem0
                logger.info(f"Loading user history for user {user_id}")
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

                logger.info(
                    f"Loaded {len(user_history)} history items for user {user_id}"
                )

                # Log user input
                logger.info(f"User input: {state['user_input']}")

                # Add history count to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["history_count"] = len(user_history)

                state["error"] = None

        except Exception as e:
            logger.error(f"Error in user input node: {e}")
            state["user_history"] = []
            state["error"] = f"User input processing error: {str(e)}"
            # On error, allow continuation to avoid breaking the app
            state["next_node"] = "intent_classification"

        return state
