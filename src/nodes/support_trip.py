"""Support Trip Ancillaries Node - Handles in-trip support queries"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.nodes.state import GraphState
from src.utils.mem0_manager import Mem0Manager
from src.utils.rag_manager import RAGManager
from src.config import Config
from src.utils.logger import setup_logger

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
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            temperature=Config.TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful travel support assistant. Provide in-trip support and recommendations.

Company Travel Policy (for compliance):
{policy_context}

User's Travel History:
{user_travel_history}

User's Selected Plans:
{user_selections}

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
        try:
            logger.info(f"Processing support query for user {state['user_id']}")

            user_id = state["user_id"]
            user_input = state["user_input"]

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

            # Generate support response
            chain = self.prompt | self.llm
            result = chain.invoke(
                {
                    "policy_context": policy_context,
                    "user_travel_history": user_travel_history,
                    "user_selections": user_selections,
                    "user_query": user_input,
                }
            )

            support_response = result.content

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
