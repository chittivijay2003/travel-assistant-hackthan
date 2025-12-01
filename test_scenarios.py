"""
Test scenarios for Travel Assistant

This file contains example inputs for testing each node
"""

TEST_SCENARIOS = {
    "information_node": [
        "I love trekking in the monsoon season",
        "I prefer vegetarian food with tea in the mountains",
        "I enjoy cultural experiences and historical sites",
        "I'm allergic to seafood",
        "I prefer window seats on flights",
        "I like staying in boutique hotels",
    ],
    "itinerary_node": [
        "Suggest a 3-day itinerary in Japan",
        "Plan a 5-day trip to Paris",
        "Create a week-long itinerary for exploring Thailand",
        "I need a 4-day itinerary for Iceland",
        "Suggest weekend trip ideas for Rome",
    ],
    "travel_plan_node": [
        "Suggest a travel plan with options for cabs and flights to Tokyo",
        "Plan my business trip to London with flights and ground transportation",
        "I need a complete travel plan for a conference in Berlin next month",
        "Book my trip to New York with flights, cabs, and hotel",
        "Create a travel plan for Singapore including all transportation",
    ],
    "support_trip_node": [
        # Lounge queries
        "Suggest lounge facilities at Tokyo Narita airport",
        "Which lounges are available at JFK airport?",
        "Airport lounge recommendations for my flight from London",
        # Food queries
        "Suggest food places for day 1 of my trip",
        "Vegetarian restaurants near my hotel in Paris",
        "Where can I get good coffee in Tokyo?",
        # Accessories queries
        "Suggest travel accessories for my upcoming 3-day trip",
        "What should I pack for a mountain trek?",
        "Essential items for international travel",
        # General support
        "Emergency contacts in Japan",
        "Best SIM card options in Thailand",
        "Currency exchange tips for Europe",
    ],
    "mixed_scenarios": [
        # First share preferences
        "I love adventure sports and spicy food",
        # Then ask for itinerary (should use preferences)
        "Suggest a 1-week itinerary for New Zealand",
        # Then ask for travel plan
        "Create a complete travel plan for this New Zealand trip with flights from San Francisco",
        # Then ask for support
        "What travel accessories do I need for adventure activities?",
    ],
}


def print_scenarios():
    """Print all test scenarios"""
    print("=" * 80)
    print("TRAVEL ASSISTANT - TEST SCENARIOS")
    print("=" * 80)
    print()

    for node_type, scenarios in TEST_SCENARIOS.items():
        print(f"\n{'=' * 80}")
        print(f"{node_type.upper().replace('_', ' ')}")
        print(f"{'=' * 80}")

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario}")

    print("\n" + "=" * 80)
    print("RECOMMENDED TESTING FLOW")
    print("=" * 80)
    print("""
1. Start by sharing 2-3 preferences (Information Node)
   Example: "I love trekking and vegetarian food"

2. Ask for an itinerary (Itinerary Node)
   Example: "Suggest a 3-day itinerary in Japan"
   → Should incorporate your preferences

3. Request a complete travel plan (Travel Plan Node)
   Example: "Plan this trip with flights and cabs"
   → System will ask for missing details (dates, origin, etc.)
   → After providing details, you'll get a policy-compliant plan

4. Ask for trip support (Support Trip Node)
   Example: "Suggest lounges at Tokyo airport"
   → Should reference your travel plan

5. Verify that:
   ✓ Preferences are remembered across conversations
   ✓ Itineraries reflect your preferences
   ✓ Travel plans comply with company policy
   ✓ Support queries use your travel context
""")


def get_scenario_by_type(node_type: str, index: int = 0):
    """Get a specific test scenario"""
    scenarios = TEST_SCENARIOS.get(node_type, [])
    if 0 <= index < len(scenarios):
        return scenarios[index]
    return None


if __name__ == "__main__":
    print_scenarios()
