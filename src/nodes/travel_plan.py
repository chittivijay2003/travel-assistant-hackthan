"""Travel Plan Node - Generates complete travel plans with flights and cabs"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.utils.rag_manager import RAGManager
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.multimodel_selector import MultiModelSelector

logger = setup_logger("travel_plan")


class TravelPlanNode:
    """Node for generating complete travel plans"""

    def __init__(self, mem0_manager: Mem0Manager, rag_manager: RAGManager):
        """
        Initialize travel plan node

        Args:
            mem0_manager: Mem0 manager instance
            rag_manager: RAG manager instance
        """
        self.mem0_manager = mem0_manager
        self.rag_manager = rag_manager
        # Use multi-model selector: gemini-2.5-pro for complex travel planning with RAG
        self.llm = MultiModelSelector.get_model_for_travel_plan()

        self.info_gathering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a travel planning assistant. Analyze the ENTIRE conversation context to identify what information is missing.

Required information for a complete travel plan:
- Destination
- Travel dates (start and end) - formats like "feb11th to 13th", "jan 5-8", "10th to 12th" are valid
- Start time of the day - recognize: morning, afternoon, evening, night, early morning, late night, OR specific times like "9am", "2pm"
- Origin/departure city
- Number of travelers - single number like "2", "3" is valid
- Budget preferences - formats like "$1500", "1500", "$2000" are valid

IMPORTANT: 
- Check BOTH the current request AND the conversation history for all required information
- The user may have provided information across multiple messages
- If the conversation shows the destination was mentioned earlier, don't ask for it again
- Be FLEXIBLE in recognizing information formats:
  * "morning" = valid start time
  * "2" = valid number of travelers
  * "$1500" or "1500" = valid budget
  * "feb11th to 13th" = valid date range
  * "tokyo" and "osaka" together = origin and destination
- If ALL information is available (either in current message or conversation history), respond with "COMPLETE"
- Don't be overly strict - if the user provides time as "morning", that's sufficient

Based on ALL available context, list ONLY the truly missing information.
If all information is available, respond with "COMPLETE".""",
                ),
                (
                    "user",
                    """Current User Input: {user_request}
            
Complete Context (History + Recent Conversation):
{user_history}

What information is still missing? If everything is available, say "COMPLETE".""",
                ),
            ]
        )

        self.travel_plan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert travel planner. Create a comprehensive travel plan with flights and cab options.

IMPORTANT: Analyze the COMPLETE conversation context to understand the full travel request.
- The destination may have been mentioned in an earlier message
- The user may have provided details across multiple messages
- Extract ALL trip information from the entire conversation, not just the current message

Company Travel Policy:
{policy_context}

User Preferences:
{user_preferences}

User Selections:
{user_selections}

Conversation Context:
{conversation_context}

Current Request:
{user_request}

Create a detailed travel plan that includes:
1. Flight options (within company budget as per policy)
2. Cab/transportation options (compliant with policy)
3. Accommodation recommendations
4. Daily itinerary
5. Estimated total costs
6. Policy compliance notes

Ensure all recommendations fall within the company's travel budget as specified in the policy.
Make sure you understand the complete trip details from the ENTIRE conversation before generating the plan.""",
                ),
                (
                    "user",
                    "Generate a complete travel plan based on all the information provided above.",
                ),
            ]
        )

    def _prepare_travel_request(self, state: GraphState) -> dict:
        """
        Prepare travel request with user selections and all required information

        Args:
            state: Current graph state

        Returns:
            Prepared request dictionary
        """
        user_id = state["user_id"]
        user_input = state["user_input"]

        # Extract information from conversation and history
        request = {
            "user_id": user_id,
            "current_input": user_input,
            "destination": None,
            "origin": None,
            "dates": None,
            "start_time": None,
            "travelers": None,
            "budget": None,
            "user_selections": state.get("user_selections"),
        }

        # Get user memories for selections and preferences
        memories = self.mem0_manager.get_memories(user_id, limit=20)
        for mem in memories:
            content = mem.get("memory", mem.get("content", "")).lower()
            metadata = mem.get("metadata", {})

            # Extract selected itinerary
            if "selected itinerary" in content or metadata.get("type") == "selection":
                request["user_selections"] = content

        logger.info(f"Prepared travel request: {request}")
        return request

    def _validate_travel_request(self, request: dict) -> tuple[bool, str]:
        """
        Validate travel request has all required information

        Args:
            request: Prepared request dictionary

        Returns:
            Tuple of (is_valid, missing_info_message)
        """
        required_fields = {
            "destination": "destination",
            "dates": "travel dates",
            "start_time": "start time of the day",
            "origin": "origin/departure city",
            "travelers": "number of travelers",
            "budget": "budget",
        }

        missing = []
        for field, display_name in required_fields.items():
            if not request.get(field):
                missing.append(display_name)

        if missing:
            missing_msg = "To create a complete travel plan, I need:\n" + "\n".join(
                [f"- {item}" for item in missing]
            )
            return False, missing_msg

        logger.info("Travel request validation passed")
        return True, ""

    def __call__(self, state: GraphState) -> GraphState:
        """
        Generate complete travel plan

        Args:
            state: Current graph state

        Returns:
            Updated state with travel plan
        """
        try:
            logger.info(f"Generating travel plan for user {state['user_id']}")

            user_id = state["user_id"]
            user_input = state["user_input"]
            conversation_history = state.get("conversation_history", [])

            # Step 1: Prepare request with user selections and all information
            logger.info("Step 1: Preparing travel request...")
            self._prepare_travel_request(state)  # Logs prepared request details

            # Check if user is confirming a previous itinerary
            logger.info(f"Conversation history length: {len(conversation_history)}")

            if conversation_history and len(conversation_history) >= 2:
                last_assistant_msg = None
                for msg in reversed(conversation_history):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg["content"]
                        break

                logger.info(
                    f"Last assistant message preview: {last_assistant_msg[:100] if last_assistant_msg else 'None'}..."
                )

                # If last message looks like an itinerary and user is confirming
                if last_assistant_msg and (
                    "day 1" in last_assistant_msg.lower()
                    or "itinerary" in last_assistant_msg.lower()
                ):
                    logger.info("Detected itinerary in conversation history")
                    # Check if current input is a confirmation
                    confirmation_words = [
                        "yes",
                        "sure",
                        "okay",
                        "go ahead",
                        "proceed",
                        "i want this",
                        "sounds good",
                    ]
                    if any(word in user_input.lower() for word in confirmation_words):
                        # Save the selected itinerary
                        logger.info(
                            "User confirmed previous itinerary, saving selection"
                        )
                        self.mem0_manager.add_memory(
                            user_id=user_id,
                            message=f"Selected itinerary: {last_assistant_msg[:500]}",
                            metadata={"type": "selection"},
                        )
                        logger.info("Selection saved successfully")
                    else:
                        logger.info(f"No confirmation word found in: {user_input}")
                else:
                    logger.info("No itinerary detected in last assistant message")
            else:
                logger.info("Not enough conversation history")  # Get user history
            memories = self.mem0_manager.get_memories(user_id, limit=15)
            user_history = (
                "\n".join(
                    [
                        f"- {mem.get('memory', mem.get('content', ''))}"
                        for mem in memories
                    ]
                )
                if memories
                else "No history available."
            )

            # Build complete context including conversation history
            context_parts = [user_history]

            # Add recent conversation context
            if conversation_history:
                conversation_text = "\n\nRecent Conversation:\n"
                for msg in conversation_history[-6:]:  # Last 3 exchanges
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_text += f"{role}: {msg['content']}\n"
                context_parts.append(conversation_text)

            full_context = "\n".join(context_parts)

            # Check if user is asking for city suggestions
            asking_for_suggestions = any(
                phrase in user_input.lower()
                for phrase in [
                    "give city names",
                    "suggest cities",
                    "which cities",
                    "what cities",
                    "city suggestions",
                    "suggest some cities",
                    "list cities",
                    "city options",
                    "what are the cities",
                ]
            )

            if asking_for_suggestions:
                # User is asking for city suggestions - use LLM to intelligently analyze context
                logger.info(
                    "Using LLM to generate smart, context-aware city suggestions"
                )

                city_suggestion_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """You are a travel assistant. The user is asking for city suggestions for their travel plan.

Analyze the conversation history to identify which destination/region the user has been discussing, then provide relevant city suggestions.

Your response should include:
1. Cities/districts WITHIN the main destination
2. Nearby cities for day trips or regional travel (with distances)
3. Clear guidance on how to specify origin and destination

Format:
**Cities in/around [Destination] region:**

**Within [Destination]:**
- List 3-5 areas/districts

**Near [Destination] (for regional travel):**
- List 4-6 nearby cities with distances

**For your travel plan:**
- If traveling TO [Destination]: 'Your City to [Destination]'
- If traveling WITHIN region: '[City1] to [City2]'

Please specify your origin and destination.""",
                        ),
                        (
                            "user",
                            """Conversation History:
{conversation_context}

Current Request: {user_request}

Provide city suggestions based on the destination discussed in the conversation.""",
                        ),
                    ]
                )

                # Build conversation context
                conversation_context_str = ""
                if conversation_history:
                    for msg in conversation_history:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        conversation_context_str += f"{role}: {msg['content']}\n\n"
                else:
                    conversation_context_str = "No previous conversation."

                try:
                    city_chain = city_suggestion_prompt | self.llm
                    city_result = city_chain.invoke(
                        {
                            "conversation_context": conversation_context_str,
                            "user_request": user_input,
                        }
                    )

                    state["response"] = city_result.content.strip()
                    state["error"] = None
                    logger.info("Smart city suggestions generated successfully")
                    return state
                except Exception as e:
                    logger.error(f"Error generating city suggestions: {e}")
                    state["response"] = (
                        "I'd be happy to suggest cities! "
                        "Could you please specify which destination region you're interested in? "
                        "For example: 'Cities in Paris region' or 'Cities near Tokyo'"
                    )
                    state["error"] = None
                    return state

            # Build comprehensive context including ALL user messages for information extraction
            all_user_inputs = []
            for msg in conversation_history:
                if msg["role"] == "user":
                    all_user_inputs.append(msg["content"])
            all_user_inputs.append(user_input)  # Add current input

            combined_user_input = " | ".join(all_user_inputs)

            logger.info(f"Combined user inputs for analysis: {combined_user_input}")

            # Check for missing information
            info_chain = self.info_gathering_prompt | self.llm
            info_result = info_chain.invoke(
                {"user_request": combined_user_input, "user_history": full_context}
            )

            missing_info = info_result.content.strip()

            logger.info(f"Missing info check result: {missing_info}")

            if missing_info != "COMPLETE" and "COMPLETE" not in missing_info:
                # Request missing information
                state["response"] = (
                    f"To create the best travel plan for you, I need some additional information:\n\n"
                    f"{missing_info}\n\n"
                    "Please provide these details so I can help you better."
                )
                state["error"] = None
                return state

            # Get policy context using RAG
            policy_query = f"travel budget cab flight policy {user_input}"
            try:
                policy_context = self.rag_manager.get_policy_context(policy_query)
            except Exception as e:
                logger.warning(f"RAG not available: {e}")
                policy_context = "Note: Policy compliance system is currently unavailable. Using general best practices."

            # Get user preferences
            preference_memories = [
                mem
                for mem in memories
                if mem.get("metadata", {}).get("type") == "preference"
            ]
            user_preferences = (
                "\n".join(
                    [
                        f"- {mem.get('memory', mem.get('content', ''))}"
                        for mem in preference_memories
                    ]
                )
                if preference_memories
                else "No specific preferences."
            )

            # Get user selections
            selection_memories = [
                mem
                for mem in memories
                if mem.get("metadata", {}).get("type") == "selection"
            ]
            user_selections = (
                "\n".join(
                    [
                        f"- {mem.get('memory', mem.get('content', ''))}"
                        for mem in selection_memories
                    ]
                )
                if selection_memories
                else "No previous selections."
            )

            # Build conversation context string
            conversation_context_str = ""
            if conversation_history:
                conversation_context_str = "Recent Conversation:\n"
                for msg in conversation_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_context_str += f"{role}: {msg['content']}\n\n"
            else:
                conversation_context_str = "No previous conversation."

            # Generate travel plan
            plan_chain = self.travel_plan_prompt | self.llm
            plan_result = plan_chain.invoke(
                {
                    "policy_context": policy_context,
                    "user_preferences": user_preferences,
                    "user_selections": user_selections,
                    "conversation_context": conversation_context_str,
                    "user_request": user_input,
                }
            )

            # DEBUG: Check what LLM returned
            logger.info(f"[DEBUG] plan_result type: {type(plan_result)}")
            logger.info(f"[DEBUG] plan_result: {plan_result}")
            logger.info(
                f"[DEBUG] plan_result.content: {plan_result.content[:100] if plan_result.content else 'EMPTY'}"
            )

            travel_plan = plan_result.content

            # DEBUG: Log what we're setting
            logger.info(
                f"[DEBUG] Setting response in state: {travel_plan[:100] if travel_plan else 'EMPTY'}..."
            )
            logger.info(
                f"[DEBUG] Response length being set: {len(travel_plan) if travel_plan else 0}"
            )

            state["response"] = travel_plan
            state["travel_plan_data"] = {
                "request": user_input,
                "plan": travel_plan,
                "policy_compliant": True,
            }
            state["policy_context"] = policy_context
            state["error"] = None

            # Save the travel plan request to history
            self.mem0_manager.add_memory(
                user_id=user_id,
                message=f"Requested travel plan: {user_input}",
                metadata={"type": "travel_plan_request"},
            )

            logger.info(f"Successfully generated travel plan for user {user_id}")

        except Exception as e:
            logger.error(f"Error in travel plan node: {e}")
            state[
                "response"
            ] = "I encountered an error creating the travel plan. Please try again."
            state["error"] = str(e)

        return state
