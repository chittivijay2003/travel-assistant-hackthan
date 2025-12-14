"""LangGraph workflow for Travel Assistant"""

from typing import Literal
from langgraph.graph import StateGraph, END
from src.nodes.state import GraphState, IntentType
from src.nodes.user_input import UserInputNode
from src.nodes.intent_classification import IntentClassificationNode
from src.nodes.information import InformationNode
from src.nodes.itinerary import ItineraryNode
from src.nodes.travel_plan import TravelPlanNode
from src.nodes.support_trip import SupportTripNode
from src.nodes.user_selection import UserSelectionNode
from src.utils.mem0_manager import Mem0Manager
from src.utils.rag_manager import RAGManager
from src.utils.logger import setup_logger
from src.utils.langfuse_manager import (
    LangFuseTracer,
    is_langfuse_enabled,
    flush_langfuse,
)

logger = setup_logger("workflow")


class TravelAssistantGraph:
    """LangGraph workflow for travel assistant"""

    def __init__(self):
        """Initialize the travel assistant graph"""
        # Initialize managers
        self.mem0_manager = Mem0Manager()
        self.rag_manager = RAGManager()

        # Initialize nodes
        self.user_input_node = UserInputNode(self.mem0_manager)
        self.intent_node = IntentClassificationNode()
        self.information_node = InformationNode(self.mem0_manager)
        self.itinerary_node = ItineraryNode(self.mem0_manager)
        self.user_selection_node = UserSelectionNode(self.mem0_manager)
        self.travel_plan_node = TravelPlanNode(self.mem0_manager, self.rag_manager)
        self.support_trip_node = SupportTripNode(self.mem0_manager, self.rag_manager)

        # Build graph
        self.graph = self._build_graph()

    def _route_after_validation(
        self, state: GraphState
    ) -> Literal["intent_classification", "end"]:
        """
        Route after user input validation - check if input was blocked

        Args:
            state: Current graph state

        Returns:
            Next node name
        """
        # Check if validation blocked the input
        next_node = state.get("next_node")

        if next_node == "end":
            logger.warning("Input validation failed - routing to END")
            return "end"

        logger.info("Input validation passed - routing to intent_classification")
        return "intent_classification"

    def _route_intent(
        self, state: GraphState
    ) -> Literal["information", "itinerary", "travel_plan", "support_trip", "end"]:
        """
        Route to appropriate node based on intent

        Args:
            state: Current graph state

        Returns:
            Next node name
        """
        intent = state.get("intent", IntentType.UNKNOWN.value)

        logger.info(f"Routing to node based on intent: {intent}")

        routing_map = {
            IntentType.INFORMATION.value: "information",
            IntentType.ITINERARY.value: "itinerary",
            IntentType.TRAVEL_PLAN.value: "travel_plan",
            IntentType.SUPPORT_TRIP.value: "support_trip",
        }

        return routing_map.get(intent, "end")

    def _route_after_selection(
        self, state: GraphState
    ) -> Literal["travel_plan", "end"]:
        """
        Route after user selection - check if user wants travel plan

        Args:
            state: Current graph state

        Returns:
            Next node name
        """
        user_input = state.get("user_input", "").lower()

        # Check if user wants to proceed with travel plan
        travel_plan_keywords = [
            "travel plan",
            "complete plan",
            "flights",
            "cabs",
            "book",
            "transportation",
            "detailed plan",
        ]

        wants_travel_plan = any(
            keyword in user_input for keyword in travel_plan_keywords
        )

        if wants_travel_plan:
            logger.info("Routing from user_selection to travel_plan")
            # Update intent for travel plan processing
            state["intent"] = IntentType.TRAVEL_PLAN.value
            return "travel_plan"

        logger.info("Routing from user_selection to END")
        return "end"

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow

        Returns:
            Compiled graph
        """
        # Create graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("load_history", self.user_input_node)
        workflow.add_node("intent_classification", self.intent_node)
        workflow.add_node("information", self.information_node)
        workflow.add_node("itinerary", self.itinerary_node)
        workflow.add_node("user_selection", self.user_selection_node)
        workflow.add_node("travel_plan", self.travel_plan_node)
        workflow.add_node("support_trip", self.support_trip_node)

        # Set entry point - starts with load_history node
        workflow.set_entry_point("load_history")

        # Load history node routes based on validation result
        workflow.add_conditional_edges(
            "load_history",
            self._route_after_validation,
            {
                "intent_classification": "intent_classification",
                "end": END,
            },
        )

        # Add conditional edges from intent classification
        workflow.add_conditional_edges(
            "intent_classification",
            self._route_intent,
            {
                "information": "information",
                "itinerary": "itinerary",
                "travel_plan": "travel_plan",
                "support_trip": "support_trip",
                "end": END,
            },
        )

        # Information and support_trip nodes end directly
        workflow.add_edge("information", END)
        workflow.add_edge("support_trip", END)

        # Itinerary ends directly (user can respond with new message for selection)
        workflow.add_edge("itinerary", END)

        # User selection can route to travel_plan if user wants complete plan, otherwise END
        workflow.add_conditional_edges(
            "user_selection",
            self._route_after_selection,
            {
                "travel_plan": "travel_plan",
                "end": END,
            },
        )

        # Travel plan ends after execution
        workflow.add_edge("travel_plan", END)

        # Compile graph
        return workflow.compile()

    def process_message(
        self, user_id: str, user_input: str, conversation_history: list = None
    ) -> str:
        """
        Process a user message through the graph

        Args:
            user_id: User identifier
            user_input: User's input message
            conversation_history: Previous conversation messages

        Returns:
            Response message
        """
        # Start LangFuse tracing for the entire pipeline
        with LangFuseTracer(
            name="travel_assistant_pipeline",
            trace_type="trace",
            metadata={
                "operation": "full_pipeline",
                "user_input": user_input[:200],
                "conversation_length": len(conversation_history)
                if conversation_history
                else 0,
            },
            user_id=user_id,
            session_id=user_id,
        ) as tracer:
            try:
                # Log transaction ID for audit trail
                logger.info(f"[AUDIT] Transaction ID: {tracer.txnid} - User: {user_id}")
                logger.info(f"Processing message for user {user_id}: {user_input}")

                # Initialize state
                initial_state = {
                    "user_id": user_id,
                    "user_input": user_input,
                    "intent": None,
                    "user_history": [],
                    "response": "",
                    "itinerary_data": None,
                    "travel_plan_data": None,
                    "user_selections": None,
                    "policy_context": None,
                    "error": None,
                    "metadata": {"txnid": tracer.txnid},  # Include txnid in metadata
                    "conversation_history": conversation_history or [],
                    "txnid": tracer.txnid,  # Include txnid at top level for easy access
                }

                # Run graph
                result = self.graph.invoke(initial_state)

                # Extract response
                response = result.get(
                    "response", "I'm sorry, I couldn't process your request."
                )

                # DEBUG: Log what we're returning
                logger.info(
                    f"[DEBUG] Response extracted from state: {response[:100] if response else 'NONE'}..."
                )
                logger.info(
                    f"[DEBUG] Response length: {len(response) if response else 0}"
                )
                logger.info(f"[DEBUG] Result keys: {list(result.keys())}")

                # Add result metadata to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["intent_classified"] = result.get("intent")
                    tracer.metadata["response_length"] = len(response)
                    tracer.metadata["success"] = not result.get("error")
                    if result.get("itinerary_data"):
                        tracer.metadata["generated_itinerary"] = True
                    if result.get("travel_plan_data"):
                        tracer.metadata["generated_travel_plan"] = True

                if result.get("error"):
                    logger.error(f"Error in graph execution: {result['error']}")
                    if tracer.trace and is_langfuse_enabled():
                        tracer.metadata["error"] = result["error"]

                # Flush traces to ensure they're sent
                flush_langfuse()

                return response

            except Exception as e:
                logger.error(f"Error processing message: {e}")

                # Add error to trace
                if tracer.trace and is_langfuse_enabled():
                    tracer.metadata["error"] = str(e)
                    tracer.metadata["success"] = False

                # Flush traces even on error
                flush_langfuse()

                return (
                    "I encountered an error processing your message. Please try again."
                )

    def ingest_policies(self):
        """Ingest policy documents into RAG system"""
        try:
            logger.info("Ingesting policy documents")
            self.rag_manager.ingest_all_policies()
            logger.info("Policy ingestion complete")
        except Exception as e:
            logger.error(f"Error ingesting policies: {e}")
            raise
