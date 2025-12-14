"""Multi-Model Selector - Integrates multi-model orchestration with LangChain chains"""

from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config

# Import multi-model instances from multimodel_manager
from src.utils.multimodel_manager import (
    gemini_25_flash,
    gemini_25_pro,
    gemini_20_flash,
)


class MultiModelSelector:
    """
    Selects appropriate model based on task complexity while preserving
    LangChain chain compatibility (ChatPromptTemplate context)
    """

    @staticmethod
    def get_model_for_intent_classification() -> ChatGoogleGenerativeAI:
        """
        Intent classification is a simple, fast task
        Use: gemini-2.5-flash (fast & cheap)
        """
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
