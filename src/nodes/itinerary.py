"""Itinerary Node - Generates travel itineraries using LLM"""

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
        # Use multi-model selector: gemini-2.5-pro for creative itinerary generation
        self.llm = MultiModelSelector.get_model_for_itinerary()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert travel planner. Create detailed and personalized travel itineraries.

**CRITICAL: USER PREFERENCES ARE MANDATORY** - If user preferences exist, they MUST be incorporated into the itinerary.

STEP 1 - ANALYZE USER PREFERENCES (HIGHEST PRIORITY):
Review the "User Preferences" section below carefully:
{user_preferences}

IF preferences mention activities (trekking, food preferences, etc.):
→ Your itinerary MUST center around those activities
→ Example: If "trekking" or "mountains" mentioned → Include hiking trails, mountain destinations, scenic viewpoints
→ Example: If "vegetarian food" mentioned → Highlight vegetarian restaurants and food options
→ Example: If "beaches" mentioned → Focus on coastal areas and water activities
→ Example: If "cultural" mentioned → Include museums, historical sites, local experiences

STEP 2 - DESTINATION CONTEXT ANALYSIS:
1. **Analyze conversation history**: Review all previous messages for location names
2. **Extract destination from history**: Look for cities, countries, regions mentioned earlier
3. **Destination persistence rule**: Once mentioned, destination stays active UNLESS:
   - User explicitly names a different destination
   - User says "change destination" or "somewhere else"

STEP 3 - UNDERSTANDING USER INTENT:
4. **Duration changes preserve destination**: 
   - History shows "Paris" + user says "4-day trip" → Still Paris, just different duration

5. **Activity preferences preserve destination**:
   - History shows destination + user expresses interests/activities
   - Interpret as: User wants those activities AT THE DESTINATION
   - Adapt the itinerary to incorporate activities within/around the existing destination

6. **Flexible activity adaptation**:
   - Be creative finding ways to incorporate interests at the destination
   - Research nearby areas, day trips, creative alternatives

Conversation Context (check for previously mentioned destinations):
{conversation_context}

**MANDATORY OUTPUT REQUIREMENTS**:
1. **If user preferences exist**: Your itinerary MUST reflect them prominently
2. **Match destination to preferences**: Choose destinations/activities that align with user's stated interests
3. **Be specific**: Don't give generic itineraries - personalize based on what you know about the user

Generate a comprehensive itinerary including:
- Daily activities **matching user's stated preferences**
- Suggested timings
- Food recommendations (**aligned with dietary preferences if mentioned**)
- Travel tips
- Estimated costs (if applicable)

Use clear day-by-day structure with **preference-aligned activities**.""",
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
                        content = msg["content"]
                        conversation_context_str += (
                            f"{role}: {content[:300]}...\n\n"
                            if len(content) > 300
                            else f"{role}: {content}\n\n"
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

                # Get LangFuse callback handler for token tracking
                langfuse_handler = get_langfuse_callback_handler(
                    trace_name=f"itinerary_{state.get('user_id', 'unknown')}",
                    user_id=state.get("user_id"),
                    session_id=state.get("user_id"),
                )

                # Generate itinerary using ENSEMBLE pattern (highest quality)
                # Get responses from multiple models and select the best one
                ensemble_models = MultiModelSelector.get_ensemble_models()

                responses = []
                for idx, model in enumerate(ensemble_models):
                    model_name = "gemini-2.5-flash" if idx == 0 else "gemini-2.5-pro"
                    logger.info(
                        f"Getting response from ensemble model {idx+1}/{len(ensemble_models)}: {model_name}"
                    )

                    chain = self.prompt | model

                    if langfuse_handler:
                        result = chain.invoke(
                            {
                                "user_preferences": user_preferences,
                                "user_request": user_input,
                                "conversation_context": conversation_context_str,
                            },
                            config={"callbacks": [langfuse_handler]},
                        )
                        langfuse_handler.flush()
                    else:
                        result = chain.invoke(
                            {
                                "user_preferences": user_preferences,
                                "user_request": user_input,
                                "conversation_context": conversation_context_str,
                            }
                        )

                    responses.append(
                        {
                            "content": result.content,
                            "model": model_name,
                            "length": len(result.content),
                        }
                    )

                # Select best response (using length as a simple heuristic - more detailed is better)
                best_response = max(responses, key=lambda x: x["length"])
                itinerary_response = best_response["content"]

                logger.info(
                    f"Selected itinerary from {best_response['model']} ({best_response['length']} chars)"
                )
                ensemble_summary = [
                    f"{r['model']}: {r['length']} chars" for r in responses
                ]
                logger.info(f"All ensemble responses: {ensemble_summary}")

                # Add ensemble metadata to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["ensemble_models"] = [r["model"] for r in responses]
                    tracer.metadata["selected_model"] = best_response["model"]
                    tracer.metadata["response_lengths"] = [
                        r["length"] for r in responses
                    ]

                # Extract token usage from response metadata (if available)
                if tracer.trace and is_langfuse_enabled():
                    if (
                        hasattr(result, "response_metadata")
                        and result.response_metadata
                    ):
                        usage = result.response_metadata.get("usage_metadata", {})
                        if usage:
                            tracer.metadata["input_tokens"] = usage.get(
                                "prompt_token_count", 0
                            )
                            tracer.metadata["output_tokens"] = usage.get(
                                "candidates_token_count", 0
                            )
                            tracer.metadata["total_tokens"] = usage.get(
                                "total_token_count", 0
                            )

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
