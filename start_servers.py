#!/usr/bin/env python3
"""Start Streamlit UI and LiveKit Agent"""

import subprocess
import time
import signal
import sys

streamlit_process = None
agent_process = None
token_process = None


def stop_servers(sig=None, frame=None):
    """Stop all servers"""
    print("\nStopping servers...")
    if streamlit_process:
        streamlit_process.terminate()
    if agent_process:
        agent_process.terminate()
    if token_process:
        token_process.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, stop_servers)
signal.signal(signal.SIGTERM, stop_servers)

print("🚀 Starting Streamlit UI...")
streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])
print("✅ Streamlit started at http://localhost:8501\n")

print("⏳ Waiting 10 seconds...")
time.sleep(10)

print("🎤 Starting LiveKit Agent...")
agent_process = subprocess.Popen(["python3", "agent.py", "dev"])
print("✅ Agent started\n")

print("⏳ Waiting 10 seconds for token...")
time.sleep(10)

print("🎤 Starting Generate Token...")
token_process = subprocess.Popen(["python3", "generate_token.py", "dev"])
print("✅ Generate started\n")

print("=" * 50)
print("✅ Both services running")
print("=" * 50)
print("📱 UI: http://localhost:8501")
print("🎤 Agent: Background")
print("\nPress Ctrl+C to stop\n")

try:
    streamlit_process.wait()
except KeyboardInterrupt:
    stop_servers()
