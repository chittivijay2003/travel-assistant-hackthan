"""Support Trip Node - Handles in-trip support queries"""

from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.utils.rag_manager import RAGManager
from src.config import Config
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import (
    LangFuseTracer,
    is_langfuse_enabled,
    get_langfuse_callback_handler,
)
from src.utils.multimodel_selector import MultiModelSelector

logger = setup_logger("support_trip")


class SupportTripNode:
    """Node for handling in-trip support queries"""

    def __init__(self, mem0_manager: Mem0Manager, rag_manager: RAGManager):
        """
        Initialize support trip node

        Args:
            mem0_manager: Mem0 manager instance
            rag_manager: RAG manager instance for policy compliance
        """
        self.mem0_manager = mem0_manager
        self.rag_manager = rag_manager
        # Use multi-model selector: gemini-2.5-flash for fast support queries
        self.llm = MultiModelSelector.get_model_for_support_trip()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful travel support assistant. Provide in-trip support and recommendations.

CRITICAL CONTEXT RULES:
1. Check the conversation history for trip details (destination, dates, etc.)
2. If user asks about "Day 2" or specific days, reference their current travel plan
3. Use conversation context to understand which trip they're referring to
4. If user mentions a destination earlier, assume queries are about that destination

Company Travel Policy (for compliance):
{policy_context}

User's Travel History:
{user_travel_history}

User's Selected Plans:
{user_selections}

Recent Conversation Context:
{conversation_context}

Based on the user's query, provide detailed recommendations for:
- Airport lounge facilities (check policy for lounge access entitlements)
- Food places and restaurants (consider meal allowances from policy)
- Travel accessories (within company guidelines)
- Local attractions
- Emergency contacts
- Any other travel-related support

IMPORTANT: Ensure all recommendations comply with the company travel policy.
Be specific, practical, and consider the user's preferences and current travel plans.""",
                ),
                ("user", "{user_query}"),
            ]
        )

    def __call__(self, state: GraphState) -> GraphState:
        """
        Handle in-trip support query

        Args:
            state: Current graph state

        Returns:
            Updated state with support response
        """
        # Start LangFuse tracing
        with LangFuseTracer(
            name="support_trip",
            trace_type="trace",
            metadata={"node": "support_trip", "model": Config.GEMINI_MODEL},
            user_id=state.get("user_id"),
            session_id=state.get("user_id"),
        ) as tracer:
            try:
                logger.info(f"Processing support query for user {state['user_id']}")

                # Add query to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["user_query"] = state["user_input"][:200]

                user_id = state["user_id"]
                user_input = state["user_input"]
                conversation_history = state.get("conversation_history", [])

                # Build conversation context
                conversation_context_str = ""
                if conversation_history:
                    conversation_context_str = (
                        "Recent Conversation (for trip context):\n"
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

                # Get policy context from RAG for compliance
                policy_context = ""
                try:
                    policy_docs = self.rag_manager.query(user_input, top_k=3)
                    if policy_docs:
                        policy_context = "\n\n".join(
                            [doc.page_content for doc in policy_docs]
                        )
                        logger.info(
                            f"Retrieved {len(policy_docs)} policy documents for support query"
                        )
                    else:
                        policy_context = "No specific policy guidelines available."
                except Exception as e:
                    logger.warning(f"Could not retrieve policy context: {e}")
                    policy_context = "No specific policy guidelines available."

                # Get user's travel history and selections
                memories = self.mem0_manager.get_memories(user_id, limit=20)

                # Filter travel-related memories
                travel_memories = [
                    mem
                    for mem in memories
                    if mem.get("metadata", {}).get("type")
                    in ["travel_plan_request", "selection"]
                ]
                user_travel_history = (
                    "\n".join(
                        [
                            f"- {mem.get('memory', mem.get('content', ''))}"
                            for mem in travel_memories
                        ]
                    )
                    if travel_memories
                    else "No travel history available."
                )

                # Get selections
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
                    else "No selections available."
                )

                # Get LangFuse callback handler for token tracking
                langfuse_handler = get_langfuse_callback_handler(
                    trace_name=f"support_trip_{state.get('user_id', 'unknown')}",
                    user_id=state.get("user_id"),
                    session_id=state.get("user_id"),
                )

                # Generate support response using CASCADE pattern (cost optimization)
                # Try cheap model first, fallback to powerful if response is insufficient
                cheap_model, powerful_model = MultiModelSelector.get_cascade_models()

                # Track which model was used
                model_used = "gemini-2.0-flash"  # Default to cheap

                try:
                    # Try cheap model first
                    chain_cheap = self.prompt | cheap_model

                    if langfuse_handler:
                        result = chain_cheap.invoke(
                            {
                                "policy_context": policy_context,
                                "user_travel_history": user_travel_history,
                                "user_selections": user_selections,
                                "conversation_context": conversation_context_str,
                                "user_query": user_input,
                            },
                            config={"callbacks": [langfuse_handler]},
                        )
                        langfuse_handler.flush()
                    else:
                        result = chain_cheap.invoke(
                            {
                                "policy_context": policy_context,
                                "user_travel_history": user_travel_history,
                                "user_selections": user_selections,
                                "conversation_context": conversation_context_str,
                                "user_query": user_input,
                            }
                        )

                    support_response = result.content

                    # Check if response is insufficient (too short or generic)
                    if len(support_response) < 100:
                        logger.info(
                            "Cheap model response insufficient, falling back to powerful model"
                        )

                        # Fallback to powerful model
                        chain_powerful = self.prompt | powerful_model
                        model_used = "gemini-2.5-pro"  # Update to powerful

                        if langfuse_handler:
                            result = chain_powerful.invoke(
                                {
                                    "policy_context": policy_context,
                                    "user_travel_history": user_travel_history,
                                    "user_selections": user_selections,
                                    "conversation_context": conversation_context_str,
                                    "user_query": user_input,
                                },
                                config={"callbacks": [langfuse_handler]},
                            )
                            langfuse_handler.flush()
                        else:
                            result = chain_powerful.invoke(
                                {
                                    "policy_context": policy_context,
                                    "user_travel_history": user_travel_history,
                                    "user_selections": user_selections,
                                    "conversation_context": conversation_context_str,
                                    "user_query": user_input,
                                }
                            )

                        support_response = result.content
                    else:
                        logger.info(
                            f"Cheap model response sufficient ({len(support_response)} chars), no fallback needed"
                        )

                except Exception as e:
                    logger.warning(f"Error with cheap model, falling back: {e}")
                    # Fallback to powerful model on error
                    chain_powerful = self.prompt | powerful_model
                    model_used = "gemini-2.5-pro"  # Update to powerful

                    if langfuse_handler:
                        result = chain_powerful.invoke(
                            {
                                "policy_context": policy_context,
                                "user_travel_history": user_travel_history,
                                "user_selections": user_selections,
                                "conversation_context": conversation_context_str,
                                "user_query": user_input,
                            },
                            config={"callbacks": [langfuse_handler]},
                        )
                        langfuse_handler.flush()
                    else:
                        result = chain_powerful.invoke(
                            {
                                "policy_context": policy_context,
                                "user_travel_history": user_travel_history,
                                "user_selections": user_selections,
                                "conversation_context": conversation_context_str,
                                "user_query": user_input,
                            }
                        )

                    support_response = result.content

                # Log which model was used for tracking
                logger.info(f"Support query answered using: {model_used}")
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["model_used"] = model_used

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

                state["response"] = support_response
                state["error"] = None

                # Save the support query to history
                self.mem0_manager.add_memory(
                    user_id=user_id,
                    message=f"Support query: {user_input}",
                    metadata={"type": "support_query"},
                )

                logger.info(f"Successfully handled support query for user {user_id}")

            except Exception as e:
                logger.error(f"Error in support trip node: {e}")
                state[
                    "response"
                ] = "I encountered an error processing your support request. Please try again."
                state["error"] = str(e)

        return state
