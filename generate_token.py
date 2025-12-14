#!/usr/bin/env python3
"""LiveKit Token Generator and Agent Dispatcher"""

import os
import asyncio
from dotenv import load_dotenv
from livekit import api
from agent import generate_token, get_livekit_url

load_dotenv()


async def dispatch_agent_to_room(room_name: str = None):
    """Create room and dispatch agent to handle voice conversations."""
    if room_name is None:
        room_name = "travel-demo-room"

    url = get_livekit_url()
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    livekit_api = api.LiveKitAPI(url, api_key, api_secret)

    try:
        try:
            await livekit_api.room.create_room(api.CreateRoomRequest(name=room_name))
            print(f"✓ Room created: {room_name}")
        except Exception:
            print(f"✓ Using existing room: {room_name}")

        await livekit_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name="test-assistant-travel",
            )
        )
        print(f"✓ Agent dispatched to room: {room_name}")
        return room_name

    except Exception as e:
        print(f"✗ Failed to dispatch agent: {e}")
        raise
    finally:
        await livekit_api.aclose()


def print_playground_instructions(
    room_name: str = None, participant_name: str = "user-1"
):
    """Generate token and display connection instructions for LiveKit Playground."""
    room_name = asyncio.run(dispatch_agent_to_room(room_name))
    jwt_token = generate_token(room_name, participant_name)

    print("\n=== LiveKit Playground Configuration ===\n")
    print("1. Go to: https://agents-playground.livekit.io/\n")
    print("2. Enter these details:")
    print(f"   LiveKit URL: {get_livekit_url()}\n")
    print("3. Paste this token:\n")
    print(jwt_token)
    print("\n4. Click 'Connect' and start talking!\n")

    return jwt_token


if __name__ == "__main__":
    print_playground_instructions()
