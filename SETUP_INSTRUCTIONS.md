# 🚀 Setup Instructions

## Quick Setup (3 Steps)

### Step 1: Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### Step 2: Configure Environment File

Open the `.env` file in the project root and add your API key:

```bash
# Edit the .env file
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

**Important:** Replace `your_actual_api_key_here` with the actual API key you copied in Step 1.

### Step 3: Start Qdrant (for local development)

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant
```

Keep this running in a separate terminal window.

## Complete Setup & Run

```bash
# 1. Install dependencies
pip install -e .

# 2. Create sample policy document
python create_policy.py

# 3. Ingest policies into vector database
python setup.py

# 4. Run the application
streamlit run app.py
```

## Configuration Options

All configuration is in the `.env` file:

```bash
# Required
GOOGLE_API_KEY=your_api_key_here

# Optional (with defaults)
GEMINI_MODEL=gemini-1.5-flash        # or gemini-1.5-pro for better quality
QDRANT_URL=http://localhost:6333    # Qdrant server URL
TEMPERATURE=0.7                      # LLM temperature (0.0-1.0)
LOG_LEVEL=INFO                       # Logging level

# Optional (for cloud services)
QDRANT_API_KEY=                      # Leave empty for local Qdrant
MEM0_API_KEY=                        # Leave empty for local Mem0
```

## Verification

To verify your setup is correct:

```bash
# Test configuration
python -c "from src.config import Config; print('✅ Configuration is valid!')"
```

If you see any errors, check that:
1. `.env` file exists in the project root
2. `GOOGLE_API_KEY` is set with your actual API key
3. No extra spaces or quotes around the API key

## Troubleshooting

### Error: "GOOGLE_API_KEY is not set"
- Make sure `.env` file exists
- Check that you added your actual API key (not the placeholder text)
- Ensure there are no spaces: `GOOGLE_API_KEY=AIza...` (not `GOOGLE_API_KEY = AIza...`)

### Error: "Qdrant connection failed"
- Make sure Qdrant is running: `docker ps`
- Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- Check `QDRANT_URL` in `.env` file

### Need Help?
- Check the full README.md for detailed documentation
- See QUICKSTART.md for a 5-minute setup guide

---

Ready to travel! ✈️🌍
