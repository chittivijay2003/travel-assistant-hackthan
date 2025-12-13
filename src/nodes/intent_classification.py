"""Intent Classification Node - Classifies user intent using LLM"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState, IntentType
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import LangFuseTracer, is_langfuse_enabled

logger = setup_logger("intent_classification")


class IntentClassificationNode:
    """Node for classifying user intent"""

    def __init__(self):
        """Initialize the intent classification node"""
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=0.3,  # Lower temperature for consistent classification
            google_api_key=Config.GOOGLE_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intent classifier for a travel assistant system.
            
Classify the user's input into ONE of these categories:

1. **information**: User sharing personal preferences/information about themselves
   - Examples: "I love trekking in monsoon", "I prefer vegetarian food", "I like mountains"
   - NOT this: Providing trip details in response to a request
   
2. **itinerary**: User asking for travel itinerary suggestions
   - Examples: "Suggest a 3-day itinerary in Japan", "Plan my trip to Paris"
   
3. **travel_plan**: User asking for complete travel plan OR providing trip details OR confirming a previous itinerary
   - Examples: "Suggest travel plan with cabs and flights", "Book my trip to London"
   - Providing details: "jan 5th to 8th, morning, hyderabad, 3 travelers, $2000"
   - Selection examples: "yes", "yes I want this plan", "I like this itinerary", "go ahead with this"
   - CRITICAL: If conversation shows the assistant just asked for trip details (destination, dates, origin, travelers, budget), 
     and user is providing those details, classify as travel_plan
   - CRITICAL: If user's input contains dates, numbers of travelers, budget amounts, origin cities - likely travel_plan
   
4. **support_trip**: User asking for in-trip support/queries
   - Examples: "Suggest lounges at airport", "Food places for day 1", "Travel accessories needed"

Conversation History:
{conversation_history}

IMPORTANT: Check if the last assistant message was asking for trip information. If yes, classify current user input as travel_plan.

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

                # Get classification from LLM
                chain = self.prompt | self.llm
                result = chain.invoke(
                    {
                        "user_input": state["user_input"],
                        "conversation_history": history_text,
                    }
                )

                # Extract intent from response
                intent_text = result.content.strip().lower()

                # Map to IntentType
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
