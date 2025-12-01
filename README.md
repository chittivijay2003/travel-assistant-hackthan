# ✈️ AI Travel Assistant with LangGraph

An intelligent travel assistant powered by LangGraph, LangChain, Mem0, and RAG that helps users plan trips, manage itineraries, and get in-trip support while ensuring compliance with company travel policies.

## 🌟 Features

### 🎯 Intent-Based Routing
The system intelligently classifies user queries into four categories:

1. **Information Node** 📝
   - Saves user preferences and travel information
   - Examples: "I love trekking in monsoon", "I prefer vegetarian food"

2. **Itinerary Node** 🗺️
   - Generates personalized travel itineraries
   - Uses user history for customization
   - Examples: "Suggest a 3-day itinerary in Japan"

3. **Travel Plan Node** 🛫
   - Creates complete travel plans with flights and cabs
   - Validates against company policy using RAG
   - Gathers missing information before planning
   - Examples: "Plan a trip to London with flights and cabs"

4. **Support Trip Ancillaries Node** 🆘
   - Provides in-trip support and recommendations
   - Examples: "Suggest lounges at Tokyo airport", "Food places for day 1"

## 🏗️ Architecture

```
User Input → Intent Classification (LLM) → Route to Appropriate Node
                                          ↓
         ┌──────────────────────────────────────────────────┐
         │                                                  │
    Information    Itinerary    Travel Plan    Support Trip
    Node           Node         Node           Node
         │                                                  │
         └──────────────────────────────────────────────────┘
                              ↓
                    Response Generation
                              ↓
                    Save to User History (Mem0)
```

### Technology Stack
- **LangGraph**: Workflow orchestration with state management
- **LangChain**: LLM interaction and prompt engineering
- **Mem0**: User history and preference management
- **RAG (FAISS)**: Policy document retrieval for compliance
- **OpenAI GPT**: Intelligent response generation
- **Streamlit**: Interactive chat UI
- **Redis**: Vector storage support

## 📋 Prerequisites

- Python 3.13+
- Google API key (for Gemini)
- Qdrant instance (local or cloud)
- Mem0 (uses local mode by default)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   cd travel-assistant-hackthan
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-1.5-flash  # or gemini-1.5-pro
   QDRANT_URL=http://localhost:6333
   ```

4. **Start Qdrant (if running locally)**
   ```bash
   # Using Docker (recommended)
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install Qdrant locally
   # See: https://qdrant.tech/documentation/quick-start/
   ```

5. **Create sample policy document**
   ```bash
   python create_policy.py
   ```
   This creates a sample company travel policy PDF in `data/policies/`

5. **Ingest policy documents**
   ```bash
   python setup.py
   ```
   This processes the policy PDF and creates the vector store for RAG

6. **Run the Streamlit UI**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Example Interactions

**1. Share Preferences (Information Node)**
```
User: I love trekking in the monsoon season
Assistant: Thank you for sharing! I've noted that: I love trekking in the monsoon season
          This information will help me provide better recommendations...
```

**2. Request Itinerary (Itinerary Node)**
```
User: Suggest a 3-day itinerary in Japan
Assistant: [Generates detailed 3-day itinerary based on user preferences]
```

**3. Plan Complete Trip (Travel Plan Node)**
```
User: Suggest a travel plan with flights and cabs to Tokyo
Assistant: [May ask for missing information like dates, start time]
          [Then generates complete plan with flights, cabs within policy budget]
```

**4. Get Trip Support (Support Trip Node)**
```
User: Suggest lounge facilities at Tokyo Narita airport
Assistant: [Provides lounge recommendations based on travel history]
```

## 📁 Project Structure

```
travel-assistant-hackthan/
├── src/
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── state.py                 # GraphState definition
│   │   ├── intent_classification.py # LLM-based intent classifier
│   │   ├── information.py           # User preference storage
│   │   ├── itinerary.py             # Itinerary generation
│   │   ├── travel_plan.py           # Complete travel planning
│   │   ├── support_trip.py          # In-trip support
│   │   └── user_selection.py        # User selection handling
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                # Logging utility
│   │   ├── mem0_manager.py          # Mem0 integration
│   │   └── rag_manager.py           # RAG system with FAISS
│   ├── config.py                    # Configuration
│   └── workflow.py                  # LangGraph workflow
├── data/
│   ├── policies/                    # Policy PDF documents
│   └── vector_store/                # FAISS vector store
├── logs/                            # Application logs
├── app.py                           # Streamlit UI
├── setup.py                         # Policy ingestion script
├── create_policy.py                 # Sample policy generator
├── pyproject.toml                   # Dependencies
└── README.md                        # This file
```

## 🔧 Configuration

### Environment Variables
Edit `.env` file to configure:

- `GOOGLE_API_KEY`: Your Google API key (required)
- `GEMINI_MODEL`: Model to use (default: gemini-1.5-flash, or use gemini-1.5-pro)
- `TEMPERATURE`: LLM temperature (default: 0.7)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `QDRANT_API_KEY`: Qdrant API key (optional, for cloud)
- `MEM0_API_KEY`: Mem0 API key (optional, uses local mode if not set)
- `LOG_LEVEL`: Logging level (default: INFO)

### Policy Documents
Place your company travel policy PDFs in `data/policies/` and run:
```bash
python setup.py
```

## 🔍 How It Works

### 1. Intent Classification
Uses LLM to classify user intent with low temperature (0.3) for consistent categorization.

### 2. User History (Mem0)
- Stores user preferences, selections, and travel history
- Retrieved contextually for personalized responses
- Tagged with metadata (preference, selection, travel_plan_request, etc.)

### 3. RAG for Policy Compliance
- PDF policies chunked and embedded using OpenAI embeddings
- Stored in FAISS vector store
- Retrieved contextually during travel planning
- Ensures recommendations comply with company budgets

### 4. LangGraph Workflow
- State-based execution with conditional routing
- Each node processes state and returns updated state
- Parallel-safe design with immutable state updates

## 🧪 Testing

Test different scenarios:

```python
# Test intent classification
"I prefer window seats"  # → Information Node
"Plan a 5-day trip to Paris"  # → Itinerary Node
"Book flights and cabs to London"  # → Travel Plan Node
"Food recommendations for day 2"  # → Support Trip Node
```

## 📊 Logging

Logs are stored in `logs/` directory:
- `ui.log` - Streamlit UI logs
- `workflow.log` - Graph execution logs
- `intent_classification.log` - Intent classification logs
- `rag_manager.log` - RAG system logs
- `mem0_manager.log` - Mem0 operations logs
- And more...

## 🛠️ Troubleshooting

### Common Issues

1. **Import errors for langchain packages**
   ```bash
   pip install --upgrade langchain langchain-google-genai langchain-community langchain-qdrant
   ```

2. **Qdrant connection issues**
   - Make sure Qdrant is running: `docker ps` (if using Docker)
   - Check `QDRANT_URL` in `.env` file
   - For local development: `docker run -p 6333:6333 qdrant/qdrant`

3. **Google API key issues**
   - Get your API key from: https://makersuite.google.com/app/apikey
   - Make sure it's correctly set in `.env` file

4. **Mem0 connection issues**
   - Mem0 will use local mode if API key is not provided
   - For production, sign up at mem0.ai

5. **Policy documents not found**
   ```bash
   python create_policy.py  # Create sample policy
   python setup.py          # Ingest into RAG system
   ```

## 🚀 Deployment

### Production Considerations

1. **Environment Setup**
   - Use production Google API key
   - Configure Qdrant cloud instance or self-hosted server
   - Configure Mem0 cloud instance
   - Use environment-specific .env files

2. **Scaling**
   - Deploy Streamlit on cloud (Streamlit Cloud, AWS, Azure)
   - Use Qdrant Cloud for managed vector DB
   - Implement request queuing for high traffic

3. **Security**
   - Never commit .env file
   - Use secrets management (AWS Secrets Manager, Azure Key Vault)
   - Implement user authentication
   - Add rate limiting

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions:
- Check the logs in `logs/` directory
- Review the troubleshooting section
- Open an issue on GitHub

---

Built with ❤️ using LangGraph, Google Gemini, Mem0, Qdrant, and Streamlit
