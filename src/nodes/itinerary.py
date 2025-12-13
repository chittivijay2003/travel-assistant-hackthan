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

CRITICAL CONTEXT ANALYSIS (HIGHEST PRIORITY):
1. **Analyze conversation history FIRST**: Before generating any itinerary, carefully review all previous messages
2. **Extract destination from history**: Look for any location names mentioned in previous user messages (cities, countries, regions)
3. **Destination persistence rule**: Once a destination is mentioned, it remains the active destination UNLESS:
   - User explicitly names a different destination ("I want to go to Rome instead")
   - User says "change destination" or "somewhere else"
   - This is the very first conversation with no previous context

UNDERSTANDING USER INTENT:
4. **Duration changes preserve destination**: 
   - If history shows "Paris" and user now says "4-day trip" → Still Paris, just different duration
   - The destination doesn't change when only trip duration/length is modified

5. **Activity preferences preserve destination**:
   - If history shows a destination was discussed, and user now expresses preferences/interests/activities
   - Interpret as: User wants to experience those activities AT THE PREVIOUSLY MENTIONED DESTINATION
   - Adapt the itinerary to incorporate those activities within/around the existing destination
   - Example semantic patterns to recognize:
     * "I love [activity]" → Add [activity] to existing destination
     * "I prefer [interest]" → Incorporate [interest] into existing destination's itinerary
     * "I enjoy [hobby]" → Find [hobby] opportunities at existing destination
     * "I'm interested in [theme]" → Theme the existing destination's itinerary around this

6. **Flexible activity adaptation**:
   - Be creative in finding ways to incorporate user's interests at the current destination
   - Every destination has nature/hiking spots, cultural activities, adventure options nearby
   - Don't suggest new destinations just because an activity seems uncommon for the location
   - Research and include day trips, nearby areas, or creative alternatives

DECISION LOGIC:
- Has destination been mentioned in history? → YES: Keep that destination, adapt activities
- Has destination been mentioned in history? → NO: User is free to suggest destinations or activities for new trip planning

User Preferences:
{user_preferences}

Conversation Context:
{conversation_context}

OUTPUT FORMAT:
Generate a comprehensive itinerary including:
- Daily activities matching user's interests
- Suggested timings
- Food recommendations
- Travel tips
- Estimated costs (if applicable)

Use clear day-by-day structure.""",
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
