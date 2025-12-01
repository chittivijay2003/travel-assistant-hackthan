# 🚀 Quick Start Guide

## Setup in 5 Minutes

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Configure Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your-key-here
# GEMINI_MODEL=gemini-1.5-flash
```

### 3. Start Qdrant (Local Development)
```bash
# Using Docker (easiest)
docker run -p 6333:6333 qdrant/qdrant

# Keep this running in a separate terminal
```

### 4. Create Sample Policy Document
```bash
python create_policy.py
```
This creates a company travel policy PDF in `data/policies/`

### 5. Ingest Policy into RAG System
```bash
python setup.py
```
This processes the PDF and creates the vector store

### 6. Run the Application

**Option A: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser

**Option B: CLI Mode**
```bash
python main.py
```

## Example Usage

### 1. Share Your Preferences
```
You: I love trekking in the monsoon season
Assistant: Thank you for sharing! I've noted that...
```

### 2. Get an Itinerary
```
You: Suggest a 3-day itinerary in Japan
Assistant: [Detailed 3-day itinerary with activities, timing, food recommendations]
```

### 3. Plan a Complete Trip
```
You: I need a travel plan to Tokyo with flights and cabs
Assistant: To create the best travel plan, I need:
- Travel dates
- Departure city
- Start time of day
[After providing info, you get a complete plan within company policy]
```

### 4. Get Trip Support
```
You: Suggest lounge facilities at Tokyo Narita airport
Assistant: [Lounge recommendations based on your travel history]
```

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade langchain langchain-google-genai langchain-community langgraph streamlit mem0ai qdrant-client langchain-qdrant
```

### Google API Key Not Set
Make sure `.env` file exists with:
```
GOOGLE_API_KEY=your-actual-key-here
```
Get your key from: https://makersuite.google.com/app/apikey

### Qdrant Not Running
```bash
# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Or check if it's running
docker ps
```

### Policy Documents Not Found
```bash
python create_policy.py
python setup.py
```

## Need Help?

Check the full README.md for:
- Detailed architecture
- Configuration options
- Deployment guide
- Advanced usage

---

Ready to travel smarter? 🎒✈️🌍
