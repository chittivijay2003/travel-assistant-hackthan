"""State definition for LangGraph workflow"""

from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum


class IntentType(str, Enum):
    """User intent types"""

    INFORMATION = "information"
    ITINERARY = "itinerary"
    TRAVEL_PLAN = "travel_plan"
    SUPPORT_TRIP = "support_trip"
    UNKNOWN = "unknown"


class GraphState(TypedDict):
    """State object for the travel assistant graph"""

    # User information
    user_id: str
    user_input: str

    # Intent classification
    intent: Optional[str]

    # User history and context
    user_history: List[Dict[str, Any]]
    conversation_history: List[
        Dict[str, str]
    ]  # Chat history for context-aware classification

    # Response
    response: str

    # Additional data
    itinerary_data: Optional[Dict[str, Any]]
    travel_plan_data: Optional[Dict[str, Any]]
    user_selections: Optional[Dict[str, Any]]

    # Policy context for RAG
    policy_context: Optional[str]

    # Error handling
    error: Optional[str]

    # Metadata
    metadata: Dict[str, Any]
