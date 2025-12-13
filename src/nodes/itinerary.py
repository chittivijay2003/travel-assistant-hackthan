"""Itinerary Node - Generates travel itineraries using LLM"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import LangFuseTracer, is_langfuse_enabled

logger = setup_logger("itinerary_node")


class ItineraryNode:
    """Node for generating travel itineraries"""

    def __init__(self, mem0_manager: Mem0Manager):
        """
        Initialize itinerary node

        Args:
            mem0_manager: Mem0 manager instance
        """
        self.mem0_manager = mem0_manager
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert travel planner. Create detailed and personalized travel itineraries.

CRITICAL CONTEXT ANALYSIS RULES:
1. Read the ENTIRE conversation history carefully to understand the complete context
2. If user says "I want a 4-day trip" but mentioned "Paris" in previous messages, the destination is PARIS
3. If user is modifying a previous request (e.g., 3-day to 4-day), keep the SAME destination unless explicitly changed
4. Extract trip information from ALL messages in the conversation, not just the current one
5. The most recent travel-related context takes precedence for details like duration
6. If user previously discussed Paris and now just says "plan my trip", assume Paris

User Preferences:
{user_preferences}

Conversation Context (contains previous destination mentions):
{conversation_context}

Generate a comprehensive itinerary that includes:
- Daily activities and attractions
- Suggested timings
- Food recommendations
- Travel tips
- Estimated costs (if applicable)

Format the response in a clear, day-by-day structure.""",
                ),
                ("user", "{user_request}"),
            ]
        )

    def __call__(self, state: GraphState) -> GraphState:
        """
        Generate itinerary based on user request

        Args:
            state: Current graph state

        Returns:
            Updated state with itinerary response
        """
        # Start LangFuse tracing
        with LangFuseTracer(
            name="itinerary_generation",
            trace_type="trace",
            metadata={"node": "itinerary", "model": Config.GEMINI_MODEL},
            user_id=state.get("user_id"),
            session_id=state.get("user_id"),
        ) as tracer:
            try:
                logger.info(f"Generating itinerary for user {state['user_id']}")

                user_id = state["user_id"]
                user_input = state["user_input"]
                conversation_history = state.get("conversation_history", [])

                # Get user preferences from Mem0
                memories = self.mem0_manager.get_memories(user_id, limit=10)
                user_preferences = (
                    "\n".join(
                        [
                            f"- {mem.get('memory', mem.get('content', ''))}"
                            for mem in memories
                        ]
                    )
                    if memories
                    else "No specific preferences recorded yet."
                )

                # Build conversation context string
                conversation_context_str = ""
                if conversation_history:
                    conversation_context_str = (
                        "Recent Conversation (check for destinations mentioned):\n"
                    )
                    for msg in conversation_history[-6:]:  # Last 3 exchanges
                        role = "User" if msg["role"] == "user" else "Assistant"
                        conversation_context_str += (
                            f"{role}: {msg['content'][:300]}...\n\n"
                            if len(msg["content"]) > 300
                            else f"{role}: {msg['content']}\n\n"
                        )
                else:
                    conversation_context_str = "No previous conversation."

                # Add input to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["user_request"] = user_input[:200]
                    tracer.metadata["preferences_count"] = len(memories)
                    tracer.metadata["conversation_history_length"] = len(
                        conversation_history
                    )

                # Generate itinerary
                chain = self.prompt | self.llm
                result = chain.invoke(
                    {
                        "user_preferences": user_preferences,
                        "user_request": user_input,
                        "conversation_context": conversation_context_str,
                    }
                )

                itinerary_response = result.content

                state["response"] = itinerary_response
                state["itinerary_data"] = {
                    "request": user_input,
                    "itinerary": itinerary_response,
                }
                state["error"] = None

                logger.info(f"Successfully generated itinerary for user {user_id}")

                # Add success to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["success"] = True
                    tracer.metadata["response_length"] = len(itinerary_response)

            except Exception as e:
                logger.error(f"Error in itinerary node: {e}")
                state[
                    "response"
                ] = "I encountered an error generating the itinerary. Please try again."
                state["error"] = str(e)

                # Add error to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["error"] = str(e)

        return state
