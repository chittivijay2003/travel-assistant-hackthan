"""Test script to verify workflow end-to-end"""

import sys
from src.workflow import TravelAssistantGraph


def test_travel_plan():
    """Test the complete travel plan flow"""
    print("=" * 80)
    print("Testing Travel Plan Workflow")
    print("=" * 80)

    # Initialize graph
    print("\n1. Initializing Travel Assistant Graph...")
    graph = TravelAssistantGraph()
    print("   ✅ Graph initialized successfully")

    # Test user ID
    user_id = "test_user_123"

    # Test case: Save preference
    print("\n2. Testing preference storage...")
    response1 = graph.process_message(
        user_id=user_id,
        user_input="I love trekking in the monsoon season",
        conversation_history=[],
    )
    print(f"   Response length: {len(response1)} characters")
    print(f"   Preview: {response1[:100]}...")

    # Test case: Request itinerary
    print("\n3. Testing itinerary generation...")
    response2 = graph.process_message(
        user_id=user_id,
        user_input="Suggest a 3-day itinerary in Japan",
        conversation_history=[
            {"role": "user", "content": "I love trekking in the monsoon season"},
            {"role": "assistant", "content": response1},
        ],
    )
    print(f"   Response length: {len(response2)} characters")
    print(f"   Preview: {response2[:100]}...")

    # Test case: Request travel plan
    print("\n4. Testing travel plan request (without details)...")
    response3 = graph.process_message(
        user_id=user_id,
        user_input="Suggest a travel plan with options for cabs and flights",
        conversation_history=[
            {"role": "user", "content": "I love trekking in the monsoon season"},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": "Suggest a 3-day itinerary in Japan"},
            {"role": "assistant", "content": response2},
        ],
    )
    print(f"   Response length: {len(response3)} characters")
    print(f"   Preview: {response3[:200]}...")

    # Test case: Provide complete travel info
    print("\n5. Testing complete travel info input...")
    print("   Input: 'tokyo to osaka march 10th to 12th morning 2 $1500'")
    response4 = graph.process_message(
        user_id=user_id,
        user_input="tokyo to osaka march 10th to 12th morning 2 $1500",
        conversation_history=[
            {"role": "user", "content": "I love trekking in the monsoon season"},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": "Suggest a 3-day itinerary in Japan"},
            {"role": "assistant", "content": response2},
            {
                "role": "user",
                "content": "Suggest a travel plan with options for cabs and flights",
            },
            {"role": "assistant", "content": response3},
        ],
    )
    print(f"   Response length: {len(response4)} characters")
    print(f"\n   ✅ FULL RESPONSE:")
    print("   " + "=" * 76)
    print(response4)
    print("   " + "=" * 76)

    # Check if response contains travel plan elements
    print("\n6. Validating response content...")
    keywords = ["flight", "cab", "hotel", "tokyo", "osaka", "budget", "$"]
    found_keywords = [kw for kw in keywords if kw.lower() in response4.lower()]
    print(f"   Found keywords: {found_keywords}")

    if len(found_keywords) >= 4:
        print("   ✅ Response contains travel plan elements")
    else:
        print("   ⚠️  Response may be incomplete")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_travel_plan()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
