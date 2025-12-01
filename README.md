# ✈️ AI Travel Assistant with LangGraph

An intelligent travel assistant powered by **LangGraph**, **Google Gemini**, **Local Embeddings**, and **RAG** that helps users plan trips, manage itineraries, and get in-trip support while ensuring compliance with company travel policies.

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
- **LangGraph**: Workflow orchestration with state management
- **LangChain**: LLM interaction and prompt engineering  
- **Google Gemini**: Language model (models/gemini-2.5-flash)
- **all-MiniLM-L6-v2**: Local embeddings for user history and policies
- **FAISS**: Vector store for policy document retrieval (no Docker needed)
- **Streamlit**: Interactive chat UI
- **JSON + NumPy**: User history storage (Mem0 alternative)

### ✅ Key Benefits
- 🚀 **100% Local Embeddings** - No API calls for embeddings, zero quota issues
- 💾 **Simple Storage** - JSON files + FAISS (no external databases)
- 🔄 **Conversation Context** - Remembers full conversation history
- 📋 **Policy Compliance** - RAG ensures all plans follow company rules
- ⚡ **Fast & Efficient** - Local embeddings + Gemini Flash

## 📋 Prerequisites

- Python 3.10+
- Google API key (for Gemini)

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
   
   Edit `.env` and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=models/gemini-2.5-flash
   ```

4. **Create and ingest policy document**
   ```bash
   python3 create_policy.py
   ```
   This creates a company travel policy PDF and ingests it into FAISS vector store.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`

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
```

**3. Update Itinerary with New Preference**
```
User: I also love vegetarian food
Assistant: [Automatically regenerates Tokyo itinerary with vegetarian restaurants]
```

**4. Plan Complete Trip (Travel Plan Node)**
```
User: Plan a trip to London with flights and cabs
Assistant: I need: Travel dates, Origin, Number of travelers, Budget

User: Jan 15-20, from NYC, 1 person, $5000
Assistant: [Generates complete plan with flights, hotel, cabs - all policy compliant]
```

**5. Get Trip Support (Support Trip Node)**
```
User: Change my hotel to a cheaper option
Assistant: [Provides cheaper hotel options within company policy]
```

## 📁 Project Structure

```
travel-assistant-hackthan/
├── src/
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── state.py                 # GraphState definition
│   │   ├── user_input.py            # Load user history from database
│   │   ├── intent_classification.py # Gemini-based intent classifier
│   │   ├── information.py           # Save preferences + smart updates
│   │   ├── itinerary.py             # Generate creative itineraries
│   │   ├── travel_plan.py           # Policy-compliant travel plans
│   │   └── support_trip.py          # Trip modifications & support
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                # Logging configuration
│   │   ├── mem0_manager.py          # User history (JSON + all-MiniLM-L6-v2)
│   │   └── rag_manager.py           # Policy RAG (FAISS + all-MiniLM-L6-v2)
│   ├── config.py                    # Environment configuration
│   └── workflow.py                  # LangGraph workflow orchestration
├── data/
│   ├── user_memories.json           # User history database
│   ├── user_embeddings.npy          # Cached embeddings
│   ├── policies/                    # Policy PDF documents
│   │   └── company_travel_policy.pdf
│   └── vector_store/                # FAISS vector store
│       └── faiss_index/
├── logs/                            # Application logs
├── .env                             # Environment variables (not in git)
├── .env.example                     # Environment template
├── app.py                           # Streamlit UI
├── create_policy.py                 # Policy generator & ingestion
├── requirements.txt                 # Python dependencies
├── ARCHITECTURE_GUIDE.md            # Detailed implementation guide
└── README.md                        # This file
```

## 🔧 Configuration

### Environment Variables
Edit `.env` file:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (defaults provided)
GEMINI_MODEL=models/gemini-2.5-flash
TEMPERATURE=0.7
```

### Get Google API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` file

### Policy Documents
- Default policy is auto-created by `create_policy.py`
- Add custom policies to `data/policies/` directory
- Policies are automatically ingested into FAISS

## 🔍 How It Works

### 1. User Input Node
- **File**: `src/nodes/user_input.py`
- Loads last 20 user memories from JSON database
- Formats as readable history for context

### 2. Intent Classification (Gemini)
- **File**: `src/nodes/intent_classification.py`
- Uses Google Gemini (temperature=0.3 for accuracy)
- Analyzes input + conversation history
- Routes to: information | itinerary | travel_plan | support_trip

### 3. Node Processing

**Information Node** (`src/nodes/information.py`)
- Saves preferences with all-MiniLM-L6-v2 embeddings
- Smart feature: Auto-regenerates itineraries when new preferences added
- Storage: `data/user_memories.json` + `data/user_embeddings.npy`

**Itinerary Node** (`src/nodes/itinerary.py`)
- Gemini generates creative day-by-day plans (temperature=0.8)
- Uses user preferences for personalization

**Travel Plan Node** (`src/nodes/travel_plan.py`)
- Validates required info (dates, origin, travelers, budget)
- Queries RAG for policy compliance
- Gemini generates complete plan (temperature=0.7)
- Includes flights, hotels, cabs with cost breakdown

**Support Trip Node** (`src/nodes/support_trip.py`)
- Retrieves current trip from history
- Queries RAG for policy compliance
- Suggests modifications within budget

### 4. User History (Mem0 Alternative)
- **File**: `src/utils/mem0_manager.py`
- **Storage**: JSON file + NumPy embeddings
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Benefits**: Local, fast, no API calls

**Metadata Types**:
- `preference` - User likes/dislikes
- `selection` - Confirmed itinerary choice
- `travel_plan_request` - Trip booking
- `trip_modification` - Changes to trip

### 5. RAG for Policy Compliance
- **File**: `src/utils/rag_manager.py`
- **Vector Store**: FAISS (local, no Docker)
- **Embeddings**: all-MiniLM-L6-v2 (same as Mem0)
- **Chunking**: 1000 chars with 200 char overlap

**Policy Content**:
- Flight budgets: $500 domestic, $2,500 international
- Hotel budgets: $200-300/night
- Cab allowances: $100/day
- Total trip budgets by duration

### 6. Conversation Context
- Full conversation history passed through all nodes
- Enables multi-turn interactions
- Remembers previous itineraries, confirmations, etc.

## 🧪 Testing

Test different scenarios:

```python
# Information Node
"I love mountains and trekking"

# Itinerary Node  
"Suggest a 5-day itinerary in Switzerland"

# Smart Update (after itinerary shown)
"I also prefer vegetarian food"  # Auto-regenerates with veggie options

# Travel Plan Node
"Plan a trip to Tokyo"
# System asks for: dates, origin, travelers, budget
"Jan 5-8, from Hyderabad, 3 travelers, $2000"
# Generates complete plan

# Support Trip Node
"Change my hotel to a cheaper option"
```

## 📊 Logging

Logs in `logs/` directory:
- `workflow.log` - Graph execution
- `intent_classification.log` - Intent decisions
- `travel_plan.log` - Travel planning
- `rag_manager.log` - Policy queries
- `mem0_manager.log` - User history operations
- `ui.log` - Streamlit UI

## 🛠️ Troubleshooting

### Common Issues

1. **Google API key errors**
   ```bash
   # Verify your key is set
   python3 -c "from src.config import Config; Config.validate()"
   ```

2. **Import errors**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **FAISS index not found**
   ```bash
   python3 create_policy.py  # Creates policy and FAISS index
   ```

4. **Slow first run**
   - First run downloads all-MiniLM-L6-v2 model (~90MB)
   - Subsequent runs are fast (model cached)

## 📖 Documentation

- **ARCHITECTURE_GUIDE.md** - Complete implementation details
  - File structure and logic placement
  - Technology choices explained
  - Code examples for each component
  - Data flow diagrams

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

**Streamlit Cloud** (Easiest):
1. Push to GitHub
2. Connect at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Add `GOOGLE_API_KEY` in secrets
4. Deploy!

**Docker**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

**Environment Variables for Production**:
- Store `GOOGLE_API_KEY` in secrets manager
- Never commit `.env` to git
- Use `.env.example` as template

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

---

Built with ❤️ using **LangGraph** • **Google Gemini** • **FAISS** • **all-MiniLM-L6-v2** • **Streamlit**
