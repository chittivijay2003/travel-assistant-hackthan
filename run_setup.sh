#!/bin/bash

# AI Travel Assistant - Setup and Run Script

echo "🚀 AI Travel Assistant - Setup Script"
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your OPENAI_API_KEY"
    echo "   Open .env in a text editor and replace 'your_openai_api_key_here'"
    echo ""
    read -p "Press Enter after you've added your API key..."
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -e . || {
    echo "❌ Failed to install dependencies"
    exit 1
}
echo "✓ Dependencies installed"

# Create sample policy
echo ""
echo "📄 Creating sample travel policy..."
python create_policy.py || {
    echo "⚠️  Warning: Failed to create sample policy"
}

# Setup RAG system
echo ""
echo "🔧 Setting up RAG system..."
python setup.py || {
    echo "⚠️  Warning: RAG setup encountered issues"
}

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "To run the application:"
echo "  Streamlit UI: streamlit run app.py"
echo "  CLI Mode:     python main.py"
echo ""
echo "For help, see README.md or QUICKSTART.md"
