<div align="center">

# ✈️ AI Travel Assistant

### Enterprise-Grade Multi-Modal Travel Planning Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-green.svg)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-2.5-orange.svg)](https://ai.google.dev/)
[![LiveKit](https://img.shields.io/badge/LiveKit-Voice-purple.svg)](https://livekit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Chat via Web UI** 💬 | **Speak with Voice Agent** 🎤 | **Enterprise Policy Compliance** 📋

</div>

---

## 🌟 Overview

An intelligent travel assistant powered by **LangGraph**, **Google Gemini**, **Local Embeddings**, and **RAG** with dual interfaces:
- 💬 **Streamlit Chat UI**: Interactive web-based conversation
- 🎤 **LiveKit Voice Agent**: Natural speech interaction with STT/TTS

Both interfaces share the same powerful backend featuring:
- 🤖 **Multi-Model Orchestration**: Gemini 2.5 Flash (fast) & Pro (complex reasoning)
- 💾 **Semantic Memory**: 64 cached embeddings with all-MiniLM-L6-v2
- 🔍 **RAG Pipeline**: FAISS-based policy compliance checking
- 🛡️ **Enterprise Safety**: NeMo Guardrails for input validation
- 📊 **Full Observability**: 20+ Langfuse trace points with transaction IDs

> 🎯 **One Backend, Two Interfaces**: Both chat and voice use `TravelAssistantGraph` - zero code duplication!

## 🌟 Features

### 🎯 Intent-Based Routing
The system intelligently classifies user queries into four categories using Google Gemini:

1. **Information Node** 📝
   - Saves user preferences and travel information
   - Smart itinerary regeneration when new preferences are added
   - Examples: "I love mountains", "I prefer vegetarian food"

2. **Itinerary Node** 🗺️
   - Generates personalized travel itineraries
   - Uses user history for customization
   - Examples: "Suggest a 3-day itinerary in Japan"

3. **Travel Plan Node** 🛫
   - Creates complete travel plans with flights, hotels, and cabs
   - Validates against company policy using RAG
   - Policy-compliant cost breakdowns
   - Conversation-aware (remembers context across messages)
   - Examples: "Plan a 5-day trip to Tokyo with flights"

4. **Support Trip Node** 🆘
   - Provides trip modifications and support
   - Policy-compliant suggestions
   - Examples: "Change my hotel", "Add cab service for day 3"

## 🏗️ Architecture

```
User Input → Load History → Intent Classification (Gemini) → Route to Node
                                                             ↓
                  ┌──────────────────────────────────────────────────┐
                  │                                                  │
            Information    Itinerary    Travel Plan    Support Trip
            Node           Node         Node           Node
                  │                                                  │
                  └──────────────────────────────────────────────────┘
                                        ↓
                          Response + Save to History
```

### 🔑 Technology Stack

**🤖 AI & LLM:**
- **LangGraph**: Workflow orchestration with state management and conditional routing
- **LangChain**: LLM interaction, prompt engineering, and chain composition
- **Google Gemini**: Multi-model orchestration
  - `gemini-2.5-flash` (4096 tokens): Fast responses for intent, information, support
  - `gemini-2.5-pro` (8192 tokens): Complex reasoning for itineraries and travel plans
  - Automatic routing based on query complexity

**💾 Memory & Retrieval:**
- **Mem0**: Semantic memory management with persistent storage (64 cached embeddings, local JSON)
- **all-MiniLM-L6-v2**: Local embeddings (384 dimensions) for user history and policies
- **FAISS**: Vector store for policy document retrieval (no external database needed)
- **RAG Pipeline**: Policy compliance with semantic search and chunking

**🔒 Safety & Observability:**
- **NeMo Guardrails**: Input validation and safety checks
- **Langfuse**: LLM observability platform tracking tokens, costs, latency, and performance
  - 20+ trace points across all workflow nodes
  - Automatic token/cost tracking via LangChain callbacks
  - Transaction ID audit trails for compliance
  - Real-time dashboard at https://cloud.langfuse.com
- **Structured Logging**: Audit trails with transaction IDs for compliance

**🎤 User Interfaces:**
- **Streamlit**: Interactive web-based chat UI (port 8501)
- **LiveKit**: Real-time voice agent with full STT/TTS pipeline
  - **STT**: OpenAI Whisper (whisper-1, English)
  - **TTS**: OpenAI TTS (alloy voice, tts-1 model)
  - **VAD**: Silero voice activity detection
  - **Integration**: Seamlessly uses same TravelAssistantGraph backend

### ✅ Key Benefits
- 🎤 **Dual Interface** - Chat UI and Voice Agent using same backend
- 🚀 **100% Local Embeddings** - No API calls for embeddings, zero quota issues
- 💾 **Simple Storage** - JSON files + FAISS (no external databases required)
- 🔄 **Conversation Context** - Full history awareness across multi-turn interactions
- 📋 **Policy Compliance** - RAG ensures all plans follow company travel rules
- ⚡ **Smart Routing** - Multi-model orchestration (Flash for speed, Pro for complexity)
- 🔍 **Full Observability** - 20+ Langfuse trace points with transaction ID audit trails
- 🛡️ **Enterprise Safety** - NeMo Guardrails for input validation and content safety

## 📋 Prerequisites

**Required:**
- Python 3.10 or higher
- Google API key (for Gemini models)

**Optional (for Voice Agent):**
- OpenAI API key (for Whisper STT and TTS)
- LiveKit account and credentials

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/chittivijay2003/travel-assistant-hackthan.git
   cd travel-assistant-hackthan
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and configure:
   ```env
   # Required
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=models/gemini-pro-latest
   
   # Optional: Cloud Memory (leave empty for local JSON storage)
   MEM0_API_KEY=
   
   # Optional: Voice Agent
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: LLM Observability (tracks tokens, costs, latency)
   # Sign up free at https://cloud.langfuse.com
   LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
   LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
   LANGFUSE_HOST=https://us.cloud.langfuse.com
   ```

4. **Create and ingest policy document**
   ```bash
   python3 create_policy.py
   ```
   This creates a company travel policy PDF and ingests it into FAISS vector store.

5. **Run the application**
   
   **Option A: Chat UI Only** (No voice setup needed)
   ```bash
   streamlit run app.py
   ```
   Access at: `http://localhost:8501`
   
   **Option B: Both Chat UI + Voice Agent**
   ```bash
   python3 start_servers.py
   ```
   - Starts Streamlit UI immediately on port 8501
   - Waits 10 seconds, then starts LiveKit voice agent
   - Both interfaces share the same backend (no code duplication)
   
   **Option C: Voice Agent Only** (For testing)
   ```bash
   python3 agent.py dev
   ```

**To generate LiveKit access token for voice client:**
```bash
python3 generate_token.py
```

## 💬 Example Interactions

**1. Share Preferences (Information Node)**
```
User: I love trekking in the mountains
Assistant: Thank you for sharing! I've noted that: I love trekking in the mountains
```

**2. Request Itinerary (Itinerary Node)**
```
User: Suggest a 3-day itinerary in Tokyo
Assistant: [Generates detailed 3-day Tokyo itinerary based on your preferences]

User: yes, proceed with this
Assistant: [Moves to travel plan creation]
```

**3. Plan Complete Trip (Travel Plan Node)**
```
User: Plan a trip to London with flights and cabs
Assistant: I need: Travel dates, Origin, Number of travelers, Budget

User: Jan 15-20, from NYC, 1 person, $5000
Assistant: [Generates complete plan with flights, hotel, cabs - all policy compliant]
```

**4. Get Trip Support (Support Trip Node)**
```
User: Change my hotel to a cheaper option
Assistant: [Provides cheaper hotel options within company policy]
```

## 🎤 Voice Agent Setup & Usage

### Architecture
The LiveKit voice agent provides hands-free interaction using the same backend as the chat UI:

```
Voice Input → Silero VAD → Whisper STT → TravelAssistantGraph → OpenAI TTS → Voice Output
                                              ↓
                                    Multi-Model Routing
                                    (Gemini Flash/Pro)
                                              ↓
                                    All Integrations:
                                    - Mem0 Memory
                                    - RAG/FAISS
                                    - NeMo Guardrails
                                    - Langfuse Tracing
```

### Quick Start

**1. Configure LiveKit credentials** in `.env`:
```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
OPENAI_API_KEY=your_openai_api_key_here
```

**2. Start the voice agent**:
```bash
python3 start_servers.py  # Starts both UI and voice agent
# Or voice only:
python3 agent.py dev
```

**3. Generate access token**:
```bash
python3 generate_token.py
```
Copy the JWT token and use it with LiveKit Playground or your client app.

**4. Connect and speak**:
- Use [LiveKit Playground](https://agents-playground.livekit.io/)
- Paste your token
- Join room: `travel-demo-room`
- Agent name: `test-assistant-travel`
- Start speaking naturally!

### Voice Commands
All chat commands work via voice (same backend):
- "I love beach vacations and water sports"
- "Suggest a 5-day trip to Paris"
- "Plan a trip from New York to London, January 15 to 20, 2 people, budget 5000 dollars"
- "Change my hotel to something cheaper"
- "Add cab service for day 3"

### Technical Details

**Voice Pipeline Components:**
- **VAD (Voice Activity Detection)**: Silero - detects when you start/stop speaking
- **STT (Speech-to-Text)**: OpenAI Whisper (whisper-1) - converts speech to text
- **LLM Backend**: TravelAssistantGraph (same as chat UI)
- **TTS (Text-to-Speech)**: OpenAI TTS (alloy voice, tts-1 model)

**Configuration in `agent.py`:**
- User ID format: `voice_{room_name}`
- Conversation history: Last 20 messages (10 exchanges)
- All backend integrations automatically active
- No code duplication with chat UI

## 📁 Project Structure

```
travel-assistant-hackthan/
├── src/
│   ├── nodes/                        # LangGraph workflow nodes
│   │   ├── __init__.py
│   │   ├── state.py                  # GraphState definition
│   │   ├── user_input.py             # Load user history (with NeMo validation)
│   │   ├── intent_classification.py  # Gemini-based intent routing
│   │   ├── information.py            # Save preferences + smart updates
│   │   ├── itinerary.py              # Generate creative itineraries
│   │   ├── travel_plan.py            # Policy-compliant travel plans
│   │   └── support_trip.py           # Trip modifications & support
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                 # Structured logging with txnid
│   │   ├── mem0_manager.py           # User memory (JSON + embeddings)
│   │   └── rag_manager.py            # Policy RAG (FAISS + semantic search)
│   ├── config.py                     # Environment configuration
│   └── workflow.py                   # LangGraph orchestration (20+ traces)
├── data/
│   ├── user_memories.json            # User history (64 cached embeddings)
│   ├── user_embeddings.npy           # all-MiniLM-L6-v2 embeddings (384-dim)
│   ├── policies/                     # Policy PDF documents
│   │   └── company_travel_policy.pdf
│   └── vector_store/                 # FAISS index
│       └── faiss_index/
│           └── index.faiss
├── logs/                             # Application logs (audit trails)
├── .env                              # Environment variables (gitignored)
├── .env.example                      # Environment template
├── app.py                            # Streamlit chat UI (port 8501)
├── agent.py                          # LiveKit voice agent (360 lines)
├── generate_token.py                 # LiveKit JWT token generator
├── start_servers.py                  # Unified startup script (48 lines)
├── create_policy.py                  # Policy generator & FAISS ingestion
├── requirements.txt                  # Python dependencies (33 packages)
└── README.md                         # This file
```

### Key Files

**User Interfaces:**
- `app.py` - Streamlit chat UI, calls `graph.process_message()`
- `agent.py` - LiveKit voice agent, calls same `graph.process_message()`
- `start_servers.py` - Starts both UI and voice agent with 10s delay

**Backend Core:**
- `src/workflow.py` - TravelAssistantGraph with all integrations
- `src/nodes/` - 7 workflow nodes with Langfuse tracing
- `src/utils/mem0_manager.py` - 64 cached embeddings, all-MiniLM-L6-v2
- `src/utils/rag_manager.py` - FAISS policy retrieval

**Configuration:**
- `.env.example` - Template with LiveKit, LangFuse, Redis settings
- `requirements.txt` - Minimal dependencies (langgraph, livekit-agents, etc.)

**Utilities:**
- `generate_token.py` - Creates JWT for LiveKit Playground
- `create_policy.py` - Generates sample policy and ingests to FAISS

## 🔧 Configuration

### Environment Variables

The `.env.example` file contains all configuration options. Copy it to `.env` and update:

```env
# =============================================================================
# GOOGLE GEMINI (REQUIRED)
# =============================================================================
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=models/gemini-pro-latest

# =============================================================================
# MEM0 MEMORY (OPTIONAL - for Cloud Memory)
# =============================================================================
# Leave empty to use local JSON storage (default)
MEM0_API_KEY=

# =============================================================================
# LIVEKIT VOICE AGENT (REQUIRED for Voice)
# =============================================================================
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
OPENAI_API_KEY=your_openai_api_key_here

# =============================================================================
# LANGFUSE (OPTIONAL - LLM Observability & Tracing)
# =============================================================================
# Tracks: Token usage, costs, latency, model performance, user sessions
# Get free account at: https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_HOST=https://us.cloud.langfuse.com

# =============================================================================
# REDIS (OPTIONAL - for Caching)
# =============================================================================
USE_REDIS=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
LOG_LEVEL=INFO
TEMPERATURE=0.7
```

### Getting API Keys

**Google Gemini API** (Required):
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy and add to `.env` as `GOOGLE_API_KEY`

**OpenAI API** (Required for Voice):
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create new secret key
3. Copy and add to `.env` as `OPENAI_API_KEY`
4. Ensure account has credits for Whisper & TTS

**LiveKit Credentials** (Required for Voice):
1. Sign up at [LiveKit Cloud](https://livekit.io/)
2. Create a new project
3. Copy API Key, API Secret, and WebSocket URL
4. Add to `.env` as `LIVEKIT_*` variables

**Langfuse** (Optional - for Observability):
1. Sign up at [Langfuse Cloud](https://cloud.langfuse.com)
2. Create a new project
3. Copy public and secret keys from project settings
4. Add to `.env` as `LANGFUSE_*` variables
5. View traces at your dashboard

### Langfuse Observability Features

When enabled, Langfuse provides:
- **20+ Trace Points**: All LLM calls, RAG queries, memory operations
- **Transaction IDs**: Unique `txnid` for each request (audit trails)
- **Cost Tracking**: Monitor Gemini API usage and costs
- **Latency Metrics**: Response time for each component
- **Error Tracking**: Failed operations with stack traces
- **Complete Flow**: Visualize entire request pipeline

Search logs with `grep '[AUDIT]' logs/workflow.log` to find transaction IDs.

### Policy Documents

The system uses company travel policies for validation:
- **Default Policy**: Auto-created by `create_policy.py`
- **Custom Policies**: Add PDFs to `data/policies/` directory
- **Ingestion**: Run `python3 create_policy.py` to update FAISS index
- **Retrieval**: Semantic search with all-MiniLM-L6-v2 embeddings

**Policy Content Includes:**
- Flight budgets (domestic/international)
- Hotel per-night rates
- Cab/transportation allowances
- Total trip budget limits by duration

## 🔍 How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
│  Streamlit UI (app.py)   |   LiveKit Voice (agent.py)      │
└────────────────┬────────────────────┬───────────────────────┘
                 │                    │
                 └──────────┬─────────┘
                            │ graph.process_message()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TravelAssistantGraph (workflow.py)             │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │  1. User Input Node (NeMo Guardrails validation)   │   │
│  └────────────────┬───────────────────────────────────┘   │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐   │
│  │  2. Intent Classification (Gemini Flash)           │   │
│  │     → information | itinerary | travel_plan |      │   │
│  │       support_trip | user_selection                │   │
│  └────────────────┬───────────────────────────────────┘   │
│                   │                                         │
│         ┌─────────┼─────────┬──────────┬──────────┐       │
│         ▼         ▼         ▼          ▼          ▼       │
│    Information Itinerary Travel_Plan Support  Selection   │
│       Node       Node       Node      Node      Node      │
│    (Flash)    (Pro 8K)   (Pro 8K)  (Flash)   (Flash)     │
│         │         │         │          │          │       │
│         └─────────┴─────────┴──────────┴──────────┘       │
│                            │                               │
│                            ▼                               │
│              Response + Save to History                    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    Mem0 Memory         RAG/FAISS          Langfuse
   (64 embeddings)   (Policy compliance)   (20+ traces)
```

### Workflow Components

**1. User Input Node** (`src/nodes/user_input.py`)
- **NeMo Guardrails**: Validates input for safety and content policy
- **Memory Loading**: Retrieves last 20 user memories from Mem0
- **Context Formatting**: Prepares conversation history for downstream nodes
- **Langfuse Trace**: Logs input processing

**2. Intent Classification** (`src/nodes/intent_classification.py`)
- **Model**: Gemini 2.5 Flash (fast, 4096 tokens)
- **Temperature**: 0.3 (for accurate routing)
- **Analysis**: Examines user input + conversation history
- **Routes**: 
  - `information` - Saving preferences
  - `itinerary` - Generate trip suggestions
  - `travel_plan` - Book complete trips
  - `support_trip` - Modify existing trips
  - `user_selection` - Confirm itinerary choices

**3. Node Processing**

**Information Node** (`src/nodes/information.py`)
- **Model**: Gemini 2.5 Flash
- **Function**: Saves user preferences with semantic embeddings
- **Smart Feature**: Auto-regenerates itineraries when new preferences added
- **Storage**: `data/user_memories.json` + `user_embeddings.npy`
- **Metadata**: `type=preference` for filtering

**Itinerary Node** (`src/nodes/itinerary.py`)
- **Model**: Gemini 2.5 Pro (8192 tokens for detailed plans)
- **Temperature**: 0.8 (creative, varied suggestions)
- **Personalization**: Uses user history from Mem0
- **Output**: Day-by-day itineraries with activities, meals, attractions

**Travel Plan Node** (`src/nodes/travel_plan.py`)
- **Model**: Gemini 2.5 Pro (8192 tokens for complex reasoning)
- **Temperature**: 0.7 (balanced creativity and accuracy)
- **Validation**: Checks for dates, origin, travelers, budget
- **RAG Integration**: Queries FAISS for policy compliance
- **Output**: Complete plan with flights, hotels, cabs, cost breakdown
- **Policy Checks**: Ensures all costs within company limits

**Support Trip Node** (`src/nodes/support_trip.py`)
- **Model**: Gemini 2.5 Flash (fast support responses)
- **Function**: Retrieves current trip from conversation history
- **RAG Integration**: Validates modifications against policies
- **Output**: Suggestions for hotel changes, cab additions, etc.

**User Selection Node** (`src/nodes/user_selection.py`)
- **Model**: Gemini 2.5 Flash
- **Function**: Confirms itinerary choices, moves to travel planning
- **Storage**: Saves selection with `type=selection` metadata

### Memory Management (Mem0)

**File**: `src/utils/mem0_manager.py`

**Architecture**:
- **Storage**: JSON file (`user_memories.json`) + NumPy embeddings
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)
- **Current State**: 64 cached embeddings loaded
- **Benefits**: 100% local, no API calls, instant retrieval

**Operations**:
- `add_memory()`: Saves new memory with metadata and embedding
- `get_memories()`: Retrieves relevant memories via semantic search
- `search_similar()`: Finds top-k similar memories by cosine similarity
- `get_all()`: Returns all memories (for context loading)

**Metadata Types**:
```python
{
    "preference": "User likes/dislikes/interests",
    "selection": "Confirmed itinerary choice",
    "travel_plan_request": "Trip booking details",
    "trip_modification": "Changes to existing trip"
}
```

### RAG for Policy Compliance

**File**: `src/utils/rag_manager.py`

**Architecture**:
- **Vector Store**: FAISS (local, no external database)
- **Embeddings**: all-MiniLM-L6-v2 (same model as Mem0)
- **Chunking**: 1000 characters with 200 character overlap
- **Documents**: PDF policies in `data/policies/`

**Policy Rules** (Auto-generated):
- **Flights**: $500 domestic, $2,500 international
- **Hotels**: $200-300/night depending on city tier
- **Cabs**: $100/day for business travel
- **Total Budgets**: Varies by trip duration (3-7+ days)

**Query Process**:
1. User requests travel plan
2. System generates embedding for query
3. FAISS retrieves top-3 relevant policy chunks
4. Policy context injected into Gemini prompt
5. Plan generated within policy constraints

### Multi-Model Orchestration

**Configuration** (`src/utils/multimodel_manager.py`):

```python
models = {
    "gemini-2.5-flash": {
        "max_tokens": 4096,
        "use_cases": ["intent", "information", "support", "selection"]
    },
    "gemini-2.5-pro": {
        "max_tokens": 8192,
        "use_cases": ["itinerary", "travel_plan"]
    }
}
```

**Routing Logic**:
- **Fast queries** → Gemini 2.5 Flash (lower cost, faster)
- **Complex reasoning** → Gemini 2.5 Pro (higher quality)
- **Cost optimization**: ~70% of requests use Flash
- **Quality assurance**: Complex plans use Pro to avoid truncation

### Conversation Context

**Implementation**:
- Full conversation history passed through all nodes
- State maintained in `GraphState` (LangGraph)
- Enables multi-turn interactions:
  - "I want to visit Japan" → "Suggest 5-day itinerary" → "yes, proceed"
- Remembers previous itineraries, confirmations, modifications
- Transaction IDs (txnid) link all operations in a conversation

## 🧪 Testing

Test different scenarios:

```python
# Information Node
"I love mountains and trekking"

# Itinerary Node  
"Suggest a 5-day itinerary in Switzerland"

# Confirm itinerary and move to travel plan
"yes, I want this plan"
"proceed with this itinerary"

# Travel Plan Node
"Plan a trip to Tokyo"
# System asks for: dates, origin, travelers, budget
"Jan 5-8, from Hyderabad, 3 travelers, $2000"
# Generates complete plan

# Support Trip Node
"Change my hotel to a cheaper option"
```

## 📊 Logging & Observability

### Local Logs

All logs stored in `logs/` directory with structured format:

```
logs/
├── workflow.log                 # Graph execution, [AUDIT] transaction IDs
├── intent_classification.log    # Intent routing decisions
├── information.log              # Preference saving operations
├── itinerary.log                # Itinerary generation
├── travel_plan.log              # Travel planning operations
├── support_trip.log             # Trip modification requests
├── rag_manager.log              # Policy retrieval queries
├── mem0_manager.log             # Memory operations (64 embeddings)
├── langfuse_manager.log         # Tracing status and errors
└── ui.log                       # Streamlit UI events
```

### Transaction ID Tracking

Every request gets a unique `txnid` for complete audit trails:

```bash
# View audit logs with transaction IDs
grep '[AUDIT]' logs/workflow.log

# Example output:
[AUDIT] Transaction ID: txn_a1b2c3d4 - User: user_123 - Intent: travel_plan
[AUDIT] Transaction ID: txn_x9y8z7w6 - User: voice_travel-demo - Intent: itinerary

# Trace specific transaction
grep 'txn_a1b2c3d4' logs/*.log
```

**Transaction Flow**:
1. User input → txnid generated
2. All nodes log operations with txnid
3. Memory, RAG, LLM calls tagged with txnid
4. Response returned with txnid in logs
5. Langfuse traces linked by txnid

### Langfuse Observability

**20+ Trace Points** throughout the system:

```
Request
├─ user_input_node (txnid, guardrails validation)
├─ intent_classification_node (Gemini Flash call)
├─ [Routed Node]
│  ├─ mem0_retrieval (semantic search)
│  ├─ rag_query (policy compliance)
│  ├─ gemini_generation (Flash or Pro)
│  └─ mem0_save (new memory with embedding)
└─ response (complete output)
```

**Dashboard Features** (https://cloud.langfuse.com):
- **Traces**: View complete request pipeline
- **Generations**: All Gemini API calls with input/output
- **Costs**: Track spending by model (Flash vs Pro)
- **Latency**: Response times for each component
- **Errors**: Failed operations with stack traces
- **Search**: Filter by txnid, user_id, intent, or model

**Metrics Tracked**:
- Total requests by interface (chat UI vs voice)
- Intent distribution (information, itinerary, travel_plan, support)
- Model usage (Flash vs Pro split)
- Average response time per intent
- Memory operations (add, search, retrieve)
- RAG queries (policy retrievals)
- Error rates and types

### Monitoring Commands

```bash
# Real-time monitoring
tail -f logs/workflow.log

# Count requests by intent
grep 'Intent:' logs/intent_classification.log | sort | uniq -c

# Check error rates
grep 'ERROR' logs/*.log | wc -l

# View policy retrievals
grep 'Retrieved policy' logs/rag_manager.log

# Monitor memory operations
grep 'Added memory' logs/mem0_manager.log

# Track transaction flow
txnid="txn_a1b2c3d4"
grep "$txnid" logs/*.log
```

## 🛠️ Troubleshooting

### Common Issues

**1. Google API Key Errors**
```bash
# Verify your key is set
python3 -c "from src.config import Config; print(Config.GOOGLE_API_KEY[:20])"

# Test Gemini access
python3 -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print('API key valid!')"
```

**Error**: `google.generativeai.types.generation_types.BlockedPromptException`
- **Cause**: Gemini safety filters blocked the prompt
- **Solution**: Rephrase input, avoid policy violations

**2. Import/Dependency Errors**
```bash
# Reinstall all dependencies
pip3 install --upgrade -r requirements.txt

# Check versions
pip3 list | grep -E "langgraph|langchain|livekit"
```

**3. FAISS Index Not Found**
```bash
# Error: "FAISS index not found at data/vector_store/faiss_index/"
python3 create_policy.py  # Creates policy PDF and FAISS index

# Verify index created
ls -la data/vector_store/faiss_index/
```

**4. Slow First Run**
- **Expected**: First run downloads all-MiniLM-L6-v2 model (~90MB)
- **Location**: `~/.cache/torch/sentence_transformers/`
- **Subsequent Runs**: Fast (model cached locally)

**5. LiveKit Voice Agent Issues**

**Error**: `LIVEKIT_URL not found in environment`
```bash
# Check .env file exists and has LiveKit config
cat .env | grep LIVEKIT

# Verify all required variables
python3 -c "from src.config import Config; print(Config.LIVEKIT_URL)"
```

**Error**: `Failed to connect to LiveKit server`
- **Check**: LiveKit URL format (must be `wss://...`)
- **Verify**: API keys are correct
- **Test**: Generate token with `python3 generate_token.py`

**Error**: `OpenAI API key not found`
- **Required**: For Whisper STT and TTS in voice agent
- **Add**: `OPENAI_API_KEY` to `.env` file

**6. Streamlit UI Issues**

**Port Already in Use**:
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

**MPS Tensor Error** (PyTorch on macOS):
- **Cause**: PyTorch MPS backend issue
- **Solution**: Already handled - embeddings use CPU fallback if needed

**7. Memory/Embedding Issues**

**Error**: `user_memories.json not found`
```bash
# Create empty memory file
mkdir -p data
echo '[]' > data/user_memories.json
echo '[]' > data/user_embeddings.npy
```

**Error**: `Shape mismatch in embeddings`
```bash
# Reset memory database
rm data/user_memories.json data/user_embeddings.npy
python3 -c "from src.utils.mem0_manager import Mem0Manager; m = Mem0Manager(); print('Memory reset')"
```

**8. Langfuse Tracing Issues**

**Traces Not Appearing**:
- **Check**: Langfuse keys in `.env` are correct
- **Verify**: `LANGFUSE_HOST` is set (default: `https://us.cloud.langfuse.com`)
- **Test**: Check `logs/langfuse_manager.log` for errors
- **Note**: Tracing continues to work locally even if Langfuse is down

**9. Policy Compliance Issues**

**Plans Exceed Budget**:
- **Check**: Policy document in `data/policies/company_travel_policy.pdf`
- **Regenerate**: Run `python3 create_policy.py` to reset policy
- **Modify**: Edit policy PDF and re-run ingestion

**RAG Not Retrieving Policies**:
```bash
# Check FAISS index
python3 -c "from src.utils.rag_manager import RAGManager; r = RAGManager(); print(r.query('flight budget'))"
```

### Debug Mode

Enable verbose logging:
```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or run with debug flag
export LOG_LEVEL=DEBUG && streamlit run app.py
```

### Getting Help

If issues persist:
1. Check `logs/` directory for error details
2. Search logs by transaction ID: `grep 'txn_...' logs/*.log`
3. Review Langfuse dashboard for failed traces
4. Check GitHub Issues: https://github.com/chittivijay2003/travel-assistant-hackthan/issues

## 📖 Additional Resources

### Documentation
- **This README**: Complete setup and usage guide
- **Code Comments**: Inline documentation throughout codebase
- **Logs**: Detailed execution traces in `logs/` directory

### External Links
- **LangGraph**: https://python.langchain.com/docs/langgraph
- **Google Gemini**: https://ai.google.dev/
- **LiveKit**: https://docs.livekit.io/
- **Langfuse**: https://langfuse.com/docs
- **NeMo Guardrails**: https://github.com/NVIDIA/NeMo-Guardrails

### Community & Support
- **Issues**: https://github.com/chittivijay2003/travel-assistant-hackthan/issues
- **Discussions**: GitHub Discussions tab
- **Email**: Contact repository owner via GitHub profile

## 🚀 Deployment

### Local Development
```bash
# Chat UI only
streamlit run app.py

# Both UI + Voice Agent
python3 start_servers.py
```

### Production Deployment Options

#### 1. Streamlit Cloud (Chat UI Only - Easiest)

**Steps**:
1. Push repository to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add secrets in dashboard:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key"
   GEMINI_MODEL = "models/gemini-pro-latest"
   ```
5. Click "Deploy"

**Limitations**:
- Chat UI only (no voice agent support)
- Free tier has resource limits
- Public URL generated automatically

#### 2. Docker Deployment (Full System)

**Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/policies data/vector_store/faiss_index logs

# Expose ports
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and Run**:
```bash
# Build image
docker build -t travel-assistant .

# Run container
docker run -d \
  -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e OPENAI_API_KEY=your_openai_key \
  -e LIVEKIT_URL=your_livekit_url \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  travel-assistant
```

**Docker Compose** (UI + Voice Agent):
```yaml
version: '3.8'

services:
  streamlit-ui:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0

  voice-agent:
    build: .
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LIVEKIT_URL=${LIVEKIT_URL}
      - LIVEKIT_API_KEY=${LIVEKIT_API_KEY}
      - LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: python3 agent.py dev
```

Run with:
```bash
docker-compose up -d
```

#### 3. Cloud Platform Deployment

**AWS (EC2 + ECS)**:
```bash
# Launch EC2 instance (Ubuntu)
# SSH into instance
ssh -i key.pem ubuntu@your-instance-ip

# Clone repository
git clone https://github.com/chittivijay2003/travel-assistant-hackthan.git
cd travel-assistant-hackthan

# Install dependencies
pip3 install -r requirements.txt

# Set environment variables
nano .env  # Add your keys

# Run with PM2 (process manager)
pip3 install pm2
pm2 start start_servers.py --name travel-assistant
pm2 save
pm2 startup
```

**Google Cloud (Cloud Run)**:
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project/travel-assistant

# Deploy to Cloud Run
gcloud run deploy travel-assistant \
  --image gcr.io/your-project/travel-assistant \
  --platform managed \
  --region us-central1 \
  --set-env-vars GOOGLE_API_KEY=your_key
```

**Azure (Container Instances)**:
```bash
# Create container registry
az acr create --resource-group myResourceGroup --name myregistry --sku Basic

# Build and push
az acr build --registry myregistry --image travel-assistant .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name travel-assistant \
  --image myregistry.azurecr.io/travel-assistant \
  --cpu 2 --memory 4 \
  --environment-variables GOOGLE_API_KEY=your_key \
  --ports 8501
```

#### 4. LiveKit Cloud (Voice Agent)

For voice agent in production:

1. **Deploy to LiveKit Cloud**:
   - Use LiveKit Cloud's agent deployment
   - Upload `agent.py` as worker
   - Configure environment variables in dashboard

2. **Self-Hosted LiveKit**:
   ```bash
   # Install LiveKit server
   curl -sSL https://get.livekit.io | bash
   
   # Run server
   livekit-server --config config.yaml
   
   # Deploy agent
   python3 agent.py prod
   ```

### Environment Variables for Production

**Security Best Practices**:
- ✅ Store secrets in cloud provider's secrets manager
- ✅ Never commit `.env` to git (already in `.gitignore`)
- ✅ Use `.env.example` as template
- ✅ Rotate API keys regularly
- ✅ Use separate keys for dev/staging/prod

**Required Variables**:
```env
# Minimum for Chat UI
GOOGLE_API_KEY=your_google_api_key

# Additional for Voice Agent
OPENAI_API_KEY=your_openai_key
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret

# Recommended for Observability
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

### Scaling Considerations

**Chat UI Scaling**:
- Streamlit is single-threaded, use multiple instances behind load balancer
- Session state stored in memory (consider Redis for multi-instance)
- Static files served directly by nginx/CDN

**Voice Agent Scaling**:
- LiveKit handles agent distribution automatically
- Each agent instance handles one conversation
- Scale by increasing agent worker pool

**Database Scaling**:
- Current: JSON files (suitable for demo/small-scale)
- Production: Migrate to PostgreSQL/MongoDB for Mem0
- FAISS: Consider Pinecone/Weaviate for distributed vector search

## 🔐 Security

- ✅ `.env` is in `.gitignore`
- ✅ API keys stored in environment variables
- ✅ No hardcoded credentials
- ✅ All data stored locally

## 📝 License

Educational project - MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature-name`
5. Open pull request

## 📧 Contact

- **GitHub**: [@chittivijay2003](https://github.com/chittivijay2003)
- **Repository**: [travel-assistant-hackthan](https://github.com/chittivijay2003/travel-assistant-hackthan)

## 🎯 Key Features Summary

| Feature | Technology | Description |
|---------|-----------|-------------|
| **Chat Interface** | Streamlit | Web-based chat UI on port 8501 |
| **Voice Interface** | LiveKit + OpenAI | Real-time voice agent with STT/TTS |
| **Multi-Model LLM** | Gemini 2.5 Flash/Pro | Smart routing: Flash (fast) & Pro (complex) |
| **Memory** | Mem0 + all-MiniLM-L6-v2 | 64 cached embeddings, 384-dim vectors |
| **Policy Compliance** | FAISS + RAG | Semantic search over company policies |
| **Safety** | NeMo Guardrails | Input validation and content filtering |
| **Observability** | Langfuse | 20+ trace points with transaction IDs |
| **Workflow** | LangGraph | Conditional routing with state management |
| **Storage** | JSON + NumPy | No external databases required |

## 📈 Performance Metrics

- **Response Time**: 2-4 seconds (Flash), 4-8 seconds (Pro)
- **Memory Footprint**: ~2GB RAM with all-MiniLM-L6-v2 loaded
- **Token Limits**: 4096 (Flash), 8192 (Pro)
- **Embeddings**: 100% local, zero API calls
- **Concurrent Users**: Single-threaded (scale with instances)
- **Voice Latency**: <500ms STT, <1s TTS

## 🔄 Version History

**Current Version**: 2.0.0
- ✅ Dual interface: Chat UI + Voice Agent
- ✅ Multi-model orchestration (Gemini Flash/Pro)
- ✅ 20+ Langfuse trace points
- ✅ NeMo Guardrails integration
- ✅ Simplified startup system
- ✅ Complete observability with transaction IDs

**v1.0.0** (Initial Release):
- Basic chat UI with LangGraph
- Single Gemini model
- JSON-based memory
- FAISS policy retrieval

---

## 🌟 Built With

<div align="center">

**LangGraph** • **Google Gemini 2.5** • **LiveKit** • **FAISS** • **all-MiniLM-L6-v2** • **Streamlit** • **Langfuse** • **NeMo Guardrails**

Made with ❤️ for intelligent travel assistance

</div>

---

### 📝 Important Notes

- **Gemini Models**: Uses `gemini-2.5-flash` and `gemini-2.5-pro` - ensure API key has access
- **Voice Agent**: Requires OpenAI API key for Whisper STT and TTS (separate from Gemini)
- **Local Storage**: All data in `data/` directory - backup regularly in production
- **Observability**: Langfuse optional but recommended for production monitoring
- **Scaling**: Single-threaded design - deploy multiple instances for high traffic

### 🔒 Security Notes

- ✅ All API keys in `.env` (gitignored)
- ✅ No hardcoded credentials
- ✅ NeMo Guardrails for input validation
- ✅ Policy compliance enforced via RAG
- ✅ Transaction IDs for audit trails
- ⚠️ Langfuse traces may contain user data - review compliance requirements
