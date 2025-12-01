"""Itinerary Node - Generates travel itineraries using LLM"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.config import Config
from src.utils.logger import setup_logger

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

Consider the user's preferences and history when creating itineraries.

User Preferences:
{user_preferences}

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
        try:
            logger.info(f"Generating itinerary for user {state['user_id']}")

            user_id = state["user_id"]
            user_input = state["user_input"]

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

            # Generate itinerary
            chain = self.prompt | self.llm
            result = chain.invoke(
                {"user_preferences": user_preferences, "user_request": user_input}
            )

            itinerary_response = result.content

            state["response"] = itinerary_response
            state["itinerary_data"] = {
                "request": user_input,
                "itinerary": itinerary_response,
            }
            state["error"] = None

            logger.info(f"Successfully generated itinerary for user {user_id}")

        except Exception as e:
            logger.error(f"Error in itinerary node: {e}")
            state["response"] = (
                "I encountered an error generating the itinerary. Please try again."
            )
            state["error"] = str(e)

        return state
