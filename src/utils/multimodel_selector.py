"""Multi-Model Selector - Integrates multi-model orchestration with LangChain chains"""

from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config

# Import multi-model instances and routing functions from multimodel_manager
from src.utils.multimodel_manager import (
    gemini_25_flash,
    gemini_25_pro,
    gemini_20_flash,
    is_simple_query,
    is_complex_query,
    is_technical_query,
)


class MultiModelSelector:
    """
    Selects appropriate model based on task complexity while preserving
    LangChain chain compatibility (ChatPromptTemplate context)
    """

    @staticmethod
    def get_model_for_intent_classification(
        query: str = None,
    ) -> ChatGoogleGenerativeAI:
        """
        Intent classification - routes by query complexity if provided

        Args:
            query: Optional user query to route by complexity

        Returns:
            gemini-2.5-flash (default) or routed model based on query
        """
        if query:
            return MultiModelSelector.route_query(query)
        return gemini_25_flash

    @staticmethod
    def get_model_for_information() -> ChatGoogleGenerativeAI:
        """
        Information extraction is simple
        Use: gemini-2.5-flash (fast & cheap)
        """
        return gemini_25_flash

    @staticmethod
    def get_model_for_itinerary() -> ChatGoogleGenerativeAI:
        """
        Itinerary generation requires creativity and reasoning
        Use: gemini-2.5-pro (better quality)
        """
        return gemini_25_pro

    @staticmethod
    def get_model_for_travel_plan() -> ChatGoogleGenerativeAI:
        """
        Travel plan is complex with RAG, policy checks, multiple requirements
        Use: gemini-2.5-pro (strongest reasoning)
        """
        return gemini_25_pro

    @staticmethod
    def get_model_for_support_trip() -> ChatGoogleGenerativeAI:
        """
        Support queries are typically simple
        Use: gemini-2.5-flash (fast responses)
        """
        return gemini_25_flash

    @staticmethod
    def route_query(query: str) -> ChatGoogleGenerativeAI:
        """
        Route query to best model based on complexity.

        Args:
            query: The user query to analyze

        Returns:
            ChatGoogleGenerativeAI: The best model for this query

        Routing logic:
        - Technical queries → gemini-2.5-pro
        - Complex queries → gemini-2.5-pro
        - Simple queries → gemini-2.5-flash
        - Default → gemini-2.5-flash
        """
        if is_technical_query(query):
            return gemini_25_pro
        elif is_complex_query(query):
            return gemini_25_pro
        elif is_simple_query(query):
            return gemini_25_flash
        else:
            return gemini_25_flash

    @staticmethod
    def get_cascade_models() -> tuple:
        """
        Return cascade pattern: (cheap_model, fallback_model)
        For cost optimization: Try cheap first, fallback to strong
        """
        return (gemini_20_flash, gemini_25_pro)

    @staticmethod
    def get_ensemble_models() -> list:
        """
        Return models for ensemble pattern
        For highest quality: Combine multiple model outputs
        """
        return [gemini_25_flash, gemini_25_pro]


# Usage example in nodes:
#
# Instead of:
#   self.llm = ChatGoogleGenerativeAI(model=Config.GEMINI_MODEL, ...)
#
# Use:
#   self.llm = MultiModelSelector.get_model_for_itinerary()
#
# Then continue using chain invocation as before:
#   chain = self.prompt | self.llm
#   result = chain.invoke({...})
