"""Main entry point for Travel Assistant - CLI version"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.workflow import TravelAssistantGraph
from src.utils.logger import setup_logger
import uuid

logger = setup_logger("main")


def main():
    """Main function for CLI interaction"""
    print("=" * 70)
    print("✈️  AI Travel Assistant - CLI Mode")
    print("=" * 70)
    print("\nInitializing...")

    try:
        # Initialize graph
        graph = TravelAssistantGraph()
        print("✅ Travel Assistant initialized successfully!\n")

        # Generate user ID
        user_id = str(uuid.uuid4())

        print("You can ask me about:")
        print("  📝 Travel preferences (I'll remember them)")
        print("  🗺️  Itinerary suggestions")
        print("  🛫 Complete travel plans (flights, cabs, hotels)")
        print("  🆘 In-trip support (lounges, food, accessories)")
        print("\nType 'quit' or 'exit' to end the conversation.\n")

        # Chat loop
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print(
                        "\n👋 Thanks for using AI Travel Assistant! Have a great trip!"
                    )
                    break

                # Process message
                print("\n🤖 Assistant: ", end="", flush=True)
                response = graph.process_message(user_id, user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n❌ Error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"\n❌ Failed to initialize: {e}")
        print("\nPlease ensure:")
        print("  1. You have created a .env file with OPENAI_API_KEY")
        print("  2. You have run: python create_policy.py")
        print("  3. You have run: python setup.py")
        print("\nFor the Streamlit UI, run: streamlit run app.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
