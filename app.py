"""Streamlit UI for Travel Assistant"""

import streamlit as st
import uuid
from src.workflow import TravelAssistantGraph
from src.utils.logger import setup_logger

logger = setup_logger("ui")

# Page configuration
st.set_page_config(page_title="AI Travel Assistant", page_icon="✈️", layout="wide")

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    logger.info(f"New session started: {st.session_state.user_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    try:
        st.session_state.graph = TravelAssistantGraph()
        logger.info("Travel Assistant Graph initialized")
    except Exception as e:
        logger.error(f"Error initializing graph: {e}")
        st.error(
            "Failed to initialize the travel assistant. Please check your configuration."
        )
        st.stop()

# App header
st.title("✈️ AI Travel Assistant")
st.markdown(
    """
Welcome to your intelligent travel companion! I can help you with:
- 📝 **Save Preferences**: Tell me about your travel preferences
- 🗺️ **Itinerary Planning**: Get customized travel itineraries
- 🛫 **Complete Travel Plans**: Book flights, cabs, and accommodations
- 🆘 **Trip Support**: Get help during your trip (lounges, food, accessories)
"""
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
    This AI-powered travel assistant uses:
    - **LangGraph** for workflow orchestration
    - **LangChain** for LLM interactions
    - **Multi-Model Orchestration** (Gemini 2.5 Flash/Pro)
    - **LangFuse** for LLM observability & tracing
    - **NeMo Guardrails** for input validation & safety
    - **Mem0** with all-MiniLM-L6-v2 for semantic memory
    - **RAG (FAISS)** for policy compliance retrieval
    - **Google Gemini** for intelligent responses
    """
    )

    st.divider()

    st.header("🔧 Session Info")
    st.text(f"Session ID: {st.session_state.user_id[:8]}...")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    st.header("📚 Examples")
    st.markdown(
        """
    **Preferences:**
    - "I love trekking in the monsoon season"
    - "I prefer vegetarian food"
    
    **Itinerary:**
    - "Suggest a 3-day itinerary in Japan"
    
    **Travel Plan:**
    - "Plan a trip to London with flights and cabs"
    
    **Support:**
    - "Suggest lounges at Tokyo airport"
    - "Food places for day 1"
    """
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input(
    "Ask me about travel plans, itineraries, or share your preferences..."
):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            try:
                response = st.session_state.graph.process_message(
                    user_id=st.session_state.user_id,
                    user_input=prompt,
                    conversation_history=st.session_state.messages,
                )

                # DEBUG: Log what we received
                logger.info(
                    f"[APP DEBUG] Received response from graph: {response[:100] if response else 'NONE'}..."
                )
                logger.info(
                    f"[APP DEBUG] Response length: {len(response) if response else 0}"
                )
                logger.info(f"[APP DEBUG] Response type: {type(response)}")

                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error processing message: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p><strong>Built with:</strong> LangGraph • LangChain • Streamlit • LangFuse • NeMo Guardrails • Mem0 • FAISS</p>
        <p><strong>Powered by:</strong> Google Gemini 2.5 (Multi-Model Orchestration)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
