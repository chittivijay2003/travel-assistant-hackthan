"""Information Node - Handles user preference storage"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import (
    LangFuseTracer,
    is_langfuse_enabled,
    get_langfuse_callback_handler,
)
from src.utils.multimodel_selector import MultiModelSelector

logger = setup_logger("information_node")


class InformationNode:
    """Node for handling user information and preferences"""

    def __init__(self, mem0_manager: Mem0Manager):
        """
        Initialize information node

        Args:
            mem0_manager: Mem0 manager instance
        """
        self.mem0_manager = mem0_manager
        # Use multi-model selector: gemini-2.5-flash for simple information extraction
        self.llm = MultiModelSelector.get_model_for_information()

        self.itinerary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert travel planner. Update the previous itinerary based on the user's new preferences.

Previous Itinerary:
{previous_itinerary}

User's New Preferences:
{new_preferences}

All User Preferences:
{all_preferences}

Generate an UPDATED comprehensive itinerary that incorporates the new preferences while keeping the same destination and duration.
Include:
- Daily activities and attractions
- Food recommendations (especially considering dietary preferences)
- Suggested timings
- Travel tips
- Estimated costs (if applicable)

Format the response in a clear, day-by-day structure.""",
                ),
                (
                    "user",
                    "Please update the itinerary with my new preference: {user_input}",
                ),
            ]
        )

    def __call__(self, state: GraphState) -> GraphState:
        """
        Process user information and store in history

        Args:
            state: Current graph state

        Returns:
            Updated state with response
        """
        # Start LangFuse tracing
        with LangFuseTracer(
            name="information_node",
            trace_type="trace",
            metadata={"node": "information", "model": Config.GEMINI_MODEL},
            user_id=state.get("user_id"),
            session_id=state.get("user_id"),
        ) as tracer:
            try:
                logger.info(f"Processing information for user {state['user_id']}")

                user_input = state["user_input"]
                user_id = state["user_id"]

                # Save to Mem0
                self.mem0_manager.add_memory(
                    user_id=user_id, message=user_input, metadata={"type": "preference"}
                )

                # Check if user just received an itinerary and is adding preferences
                conversation_history = state.get("conversation_history", [])
                should_update_itinerary = False
                previous_itinerary = None

                if conversation_history and len(conversation_history) >= 2:
                    # Find the last assistant message
                    for msg in reversed(conversation_history):
                        if msg["role"] == "assistant":
                            last_msg = msg["content"]
                            # Check if it's an itinerary (contains day-by-day info)
                            if (
                                "day 1" in last_msg.lower()
                                or "day 2" in last_msg.lower()
                            ):
                                should_update_itinerary = True
                                previous_itinerary = last_msg
                                logger.info(
                                    "Detected previous itinerary, will update with new preferences"
                                )
                            break

                if should_update_itinerary and previous_itinerary:
                    # Get all user preferences
                    memories = self.mem0_manager.get_memories(user_id, limit=20)
                    all_preferences = "\n".join(
                        [
                            f"- {mem.get('memory', mem.get('content', ''))}"
                            for mem in memories
                            if mem.get("metadata", {}).get("type") == "preference"
                        ]
                    )

                    # Regenerate itinerary with new preferences
                    chain = self.itinerary_prompt | self.llm

                    # Get LangFuse callback handler for token tracking
                    langfuse_handler = get_langfuse_callback_handler(
                        trace_name=f"information_update_itinerary_{user_id}",
                        user_id=user_id,
                        session_id=user_id,
                    )

                    if langfuse_handler:
                        result = chain.invoke(
                            {
                                "previous_itinerary": previous_itinerary,
                                "new_preferences": user_input,
                                "all_preferences": all_preferences,
                                "user_input": user_input,
                            },
                            config={"callbacks": [langfuse_handler]},
                        )
                        # Flush to ensure trace is sent
                        langfuse_handler.flush()
                    else:
                        result = chain.invoke(
                            {
                                "previous_itinerary": previous_itinerary,
                                "new_preferences": user_input,
                                "all_preferences": all_preferences,
                                "user_input": user_input,
                            }
                        )

                    updated_itinerary = result.content

                    response = (
                        f"Great! I've noted your preference: {user_input}\n\n"
                        f"Here's your updated itinerary incorporating this preference:\n\n"
                        f"{updated_itinerary}"
                    )

                    state["itinerary_data"] = {
                        "request": "Updated itinerary with new preferences",
                        "itinerary": updated_itinerary,
                    }
                    state["response"] = response
                    state["error"] = None
                else:
                    # Just save preference without updating itinerary
                    response = (
                        f"Thank you for sharing! I've noted that: {user_input}\n\n"
                        "This information will help me provide better recommendations for your future travels. "
                        "Feel free to share more preferences or ask me for travel suggestions!"
                    )

                    state["response"] = response
                    state["error"] = None

                    logger.info(f"Successfully saved preference for user {user_id}")

                    # Add metadata to trace
                    if tracer.trace and is_langfuse_enabled():
                        tracer.metadata["preference_saved"] = True
                        tracer.metadata["itinerary_updated"] = should_update_itinerary

            except Exception as e:
                logger.error(f"Error in information node: {e}")
                state[
                    "response"
                ] = "I encountered an error saving your information. Please try again."
                state["error"] = str(e)

                # Add error to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["error"] = str(e)

        return state
