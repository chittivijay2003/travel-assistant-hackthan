"""
Travel Assistant Architecture Visualization

Run this to see the LangGraph workflow structure
"""


def print_architecture():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    AI TRAVEL ASSISTANT ARCHITECTURE                      ║
╚══════════════════════════════════════════════════════════════════════════╝

                              USER INPUT
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Streamlit/CLI UI      │
                    └──────────┬──────────────┘
                               │
                               ▼
        ╔══════════════════════════════════════════════════╗
        ║          LANGGRAPH WORKFLOW (State-Based)        ║
        ╚══════════════════════════════════════════════════╝
                               │
                               ▼
                ┌──────────────────────────────┐
                │  INTENT CLASSIFICATION NODE  │
                │      (LLM - GPT-4)          │
                │  Temperature: 0.3           │
                └──────────┬───────────────────┘
                           │
            ┌──────────────┼──────────────┬─────────────┐
            │              │              │             │
            ▼              ▼              ▼             ▼
    ┌─────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐
    │INFORMATION  │  │ITINERARY │  │  TRAVEL   │  │ SUPPORT  │
    │   NODE      │  │  NODE    │  │   PLAN    │  │   TRIP   │
    │             │  │  (LLM)   │  │NODE(LLM+  │  │NODE(LLM) │
    │             │  │          │  │   RAG)    │  │          │
    └──────┬──────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘
           │              │              │             │
           │              │              │             │
    Save to Mem0    Get from Mem0  Get from Mem0  Get from Mem0
                    + Generate      + RAG Query    + Generate
                      with LLM      + Generate      Support with
                                     with LLM         LLM
           │              │              │             │
           └──────────────┴──────────────┴─────────────┘
                               │
                               ▼
                          RESPONSE
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Save to Mem0       │
                    │  (if applicable)    │
                    └─────────────────────┘

╔══════════════════════════════════════════════════════════════════════════╗
║                         SUPPORTING SYSTEMS                               ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│  MEM0 (User History Manager)                                            │
│  ├─ Stores: preferences, selections, travel plans, support queries     │
│  ├─ Metadata tagging for type classification                           │
│  └─ Semantic search for context retrieval                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  RAG SYSTEM (Policy Compliance)                                         │
│  ├─ Vector Store: FAISS                                                │
│  ├─ Embeddings: OpenAI                                                 │
│  ├─ Documents: Company travel policy PDFs                              │
│  └─ Purpose: Ensure budget compliance in travel plans                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  LOGGING SYSTEM                                                         │
│  ├─ File logging per module                                            │
│  ├─ Console logging for real-time monitoring                           │
│  └─ Error tracking and debugging                                       │
└─────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════╗
║                            DATA FLOW                                     ║
╚══════════════════════════════════════════════════════════════════════════╝

Example 1: "I love trekking in monsoon"
    → Intent: INFORMATION
    → Action: Save to Mem0
    → Response: Acknowledgment

Example 2: "Suggest 3-day itinerary in Japan"
    → Intent: ITINERARY
    → Action: Retrieve preferences from Mem0 → LLM generates itinerary
    → Response: Detailed itinerary considering user preferences

Example 3: "Plan trip to London with flights and cabs"
    → Intent: TRAVEL_PLAN
    → Action: 
        1. Check missing info (dates, origin, etc.)
        2. RAG query for policy rules
        3. Retrieve user preferences from Mem0
        4. LLM generates compliant plan
    → Response: Complete travel plan with flights, cabs, hotels

Example 4: "Suggest lounges at Tokyo airport"
    → Intent: SUPPORT_TRIP
    → Action: Retrieve travel history from Mem0 → LLM generates suggestions
    → Response: Lounge recommendations

╔══════════════════════════════════════════════════════════════════════════╗
║                        TECHNOLOGY STACK                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

┌────────────────┬─────────────────────────────────────────────────────────┐
│ LangGraph      │ Workflow orchestration, state management, routing      │
│ LangChain      │ LLM chains, prompts, document processing               │
│ OpenAI GPT     │ Intent classification, itinerary/plan generation       │
│ Mem0           │ User history, preferences, selections storage          │
│ FAISS          │ Vector store for policy documents                      │
│ Streamlit      │ Interactive chat UI                                    │
│ Python logging │ Comprehensive logging and error tracking               │
└────────────────┴─────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print_architecture()
