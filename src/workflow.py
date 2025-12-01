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
from src.utils.mem0_manager import Mem0Manager
from src.utils.rag_manager import RAGManager
from src.utils.logger import setup_logger

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
        self.travel_plan_node = TravelPlanNode(self.mem0_manager, self.rag_manager)
        self.support_trip_node = SupportTripNode(self.mem0_manager)

        # Build graph
        self.graph = self._build_graph()

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

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow

        Returns:
            Compiled graph
        """
        # Create graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("user_input", self.user_input_node)
        workflow.add_node("intent_classification", self.intent_node)
        workflow.add_node("information", self.information_node)
        workflow.add_node("itinerary", self.itinerary_node)
        workflow.add_node("travel_plan", self.travel_plan_node)
        workflow.add_node("support_trip", self.support_trip_node)

        # Set entry point - starts with user_input node
        workflow.set_entry_point("user_input")

        # User input node always goes to intent classification
        workflow.add_edge("user_input", "intent_classification")

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

        # All nodes end after execution
        workflow.add_edge("information", END)
        workflow.add_edge("itinerary", END)
        workflow.add_edge("travel_plan", END)
        workflow.add_edge("support_trip", END)

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
        try:
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
                "metadata": {},
                "conversation_history": conversation_history or [],
            }

            # Run graph
            result = self.graph.invoke(initial_state)

            # Extract response
            response = result.get(
                "response", "I'm sorry, I couldn't process your request."
            )

            if result.get("error"):
                logger.error(f"Error in graph execution: {result['error']}")

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I encountered an error processing your message. Please try again."

    def ingest_policies(self):
        """Ingest policy documents into RAG system"""
        try:
            logger.info("Ingesting policy documents")
            self.rag_manager.ingest_all_policies()
            logger.info("Policy ingestion complete")
        except Exception as e:
            logger.error(f"Error ingesting policies: {e}")
            raise
