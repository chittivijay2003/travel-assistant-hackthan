"""Intent Classification Node - Classifies user intent using LLM"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState, IntentType
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import LangFuseTracer, is_langfuse_enabled
from src.utils.multimodel_selector import MultiModelSelector

logger = setup_logger("intent_classification")


class IntentClassificationNode:
    """Node for classifying user intent"""

    def __init__(self):
        """Initialize the intent classification node"""
        # Use multi-model selector: gemini-2.5-flash for fast intent classification
        self.llm = MultiModelSelector.get_model_for_intent_classification()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intent classifier for a travel assistant system.

**CRITICAL CONTEXT-AWARENESS RULE**: 
If conversation history shows ANY previous trip-related discussions (itineraries, trip planning, destination mentions), 
then activity/preference statements like "I love trekking in monsoon" should be classified as **itinerary** 
because the user is implicitly requesting trip suggestions for that activity.
            
Classify the user's input into ONE of these categories:

1. **information**: User sharing personal preferences/information ONLY in initial conversation with NO trip context
   - Examples when NO previous trip discussion exists: "I love trekking", "I prefer vegetarian food"
   - NEVER use this if conversation history contains itineraries, trip planning, or destination discussions
   
2. **itinerary**: User asking for travel itinerary suggestions (explicit OR implicit)
   - Explicit examples: "Suggest a 3-day itinerary in Japan", "Plan my trip to Paris"
   - **IMPLICIT ACTIVITY-BASED REQUESTS (classify as itinerary)**:
     * "I love trekking in monsoon" (after previous trip discussions) → itinerary for trekking destinations
     * "I enjoy beaches" (after trip context) → itinerary for beach destinations
     * "I like adventure sports" (after trip context) → itinerary for adventure destinations
   - CONTEXT-AWARE: If user previously asked for trips (Paris, 2-day, etc.) and now mentions activity/season/preferences, 
     classify as **itinerary** (they want suggestions for that activity)
   - Key indicators: Day count/duration, destinations, OR activity/preferences mentioned AFTER trip context
   
3. **travel_plan**: User asking for complete travel plan OR providing trip details OR confirming a previous itinerary
   - Explicit requests: "Suggest travel plan with cabs and flights", "Book my trip to London"
   - Providing details in response to assistant's questions: "jan 5th to 8th, morning, hyderabad, 3 travelers, $2000"
   - Route details: "tokyo to osaka", "mumbai to goa", "delhi to jaipur" (when travel plan was discussed)
   - Selection examples: "yes", "yes I want this plan", "I like this itinerary", "go ahead with this"
   - **CRITICAL CONTEXT CHECK**: Check if assistant's LAST message asked for travel plan or mentioned "cabs", "flights", "travel plan", "booking"
     * If YES and user now provides: destinations/routes/cities → classify as **travel_plan**
     * Example: Assistant says "Suggest travel plan" → User says "tokyo to osaka" → **travel_plan** (providing route)
   - CRITICAL: If user's input contains dates, numbers of travelers, budget amounts, origin/destination cities - likely travel_plan
   - Pattern recognition: "city to city" format often indicates travel plan route specification
   
4. **support_trip**: User asking for in-trip support/queries
   - Examples: "Suggest lounges at airport", "Food places for day 1", "Travel accessories needed"

Conversation History (REVIEW THIS CAREFULLY):
{conversation_history}

DECISION LOGIC (Apply in order):
Step 1: Check assistant's LAST message for context:
  - Did assistant ask for travel plan / mention "cabs and flights" / "booking"?
    → If YES and user provides route/cities ("tokyo to osaka") → **travel_plan**
  - Did assistant ask for trip details (dates, travelers, budget)?
    → If YES and user provides those details → **travel_plan**

Step 2: Check user input format:
  - "city to city" format (tokyo to osaka, paris to london)?
    → If previous context has travel plan discussion → **travel_plan**
  - Contains dates, traveler count, budget?
    → **travel_plan**

Step 3: For activity/preference statements without travel plan context:
  - Does history show previous trip discussions (itineraries, destinations)?
    → YES: Classify as **itinerary** (adding activity to existing trip context)
    → NO: Classify as **information** (just sharing preferences)

Step 4: Duration modifications:
  - "4-day trip" after "3-day Paris" → **itinerary** (modifying existing trip)

Respond with ONLY the category name: information, itinerary, travel_plan, or support_trip""",
                ),
                ("user", "{user_input}"),
            ]
        )

    def __call__(self, state: GraphState) -> GraphState:
        """
        Classify user intent

        Args:
            state: Current graph state

        Returns:
            Updated state with intent
        """
        # Start LangFuse tracing for intent classification
        with LangFuseTracer(
            name="intent_classification",
            trace_type="trace",
            metadata={
                "node": "intent_classification",
                "model": Config.GEMINI_MODEL,
                "temperature": 0.3,
            },
            user_id=state.get("user_id"),
            session_id=state.get("user_id"),
        ) as tracer:
            try:
                logger.info(f"Classifying intent for user {state['user_id']}")

                # Build conversation history context
                conversation_history = state.get("conversation_history", [])
                history_text = (
                    "\n".join(
                        [
                            f"{msg['role']}: {msg['content'][:200]}..."
                            if len(msg["content"]) > 200
                            else f"{msg['role']}: {msg['content']}"
                            for msg in conversation_history[-4:]  # Last 2 exchanges
                        ]
                    )
                    if conversation_history
                    else "No previous conversation."
                )

                # Add input to trace metadata
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["user_input"] = state["user_input"][:200]
                    tracer.metadata["history_length"] = len(conversation_history)

                # Get classification from LLM using LangChain chain (preserves context)
                chain = self.prompt | self.llm
                result = chain.invoke(
                    {
                        "user_input": state["user_input"],
                        "conversation_history": history_text,
                    }
                )

                # Extract intent from response
                result_text = result.content
                intent_text = result_text.strip().lower()  # Map to IntentType
                intent_mapping = {
                    "information": IntentType.INFORMATION,
                    "itinerary": IntentType.ITINERARY,
                    "travel_plan": IntentType.TRAVEL_PLAN,
                    "support_trip": IntentType.SUPPORT_TRIP,
                }

                intent = intent_mapping.get(intent_text, IntentType.UNKNOWN)

                logger.info(f"Classified intent: {intent.value}")

                # Add result to trace metadata
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["classified_intent"] = intent.value
                    tracer.metadata["raw_response"] = intent_text

                state["intent"] = intent.value
                state["error"] = None

            except Exception as e:
                logger.error(f"Error in intent classification: {e}")
                state["intent"] = IntentType.UNKNOWN.value
                state["error"] = f"Intent classification error: {str(e)}"

                # Add error to trace metadata
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["error"] = str(e)

        return state
