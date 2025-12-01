"""Setup script to ingest policy documents into RAG"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.workflow import TravelAssistantGraph
from src.utils.logger import setup_logger

logger = setup_logger("setup")


def main():
    """Main setup function"""
    try:
        logger.info("Starting policy document ingestion...")

        # Initialize graph (which includes RAG manager)
        graph = TravelAssistantGraph()

        # Ingest policies
        graph.ingest_policies()

        logger.info("Setup complete! Policy documents have been ingested.")
        print(
            "\n✅ Setup complete! You can now run the application with: streamlit run app.py"
        )

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
