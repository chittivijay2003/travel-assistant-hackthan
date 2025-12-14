"""LiveKit Voice Agent for Travel Assistant

Voice-enabled AI agent using LiveKit integrated with Travel Assistant backend.
Handles STT, multi-model LLM routing, and TTS for conversational travel planning.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from datetime import timedelta
from livekit import api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    AgentServer,
    cli,
    llm,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero
from src.workflow import TravelAssistantGraph

load_dotenv()


@dataclass
class LiveKitConfig:
    """LiveKit configuration from environment variables."""

    url: str
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "LiveKitConfig":
        """Load LiveKit configuration from environment variables."""
        url = os.environ.get("LIVEKIT_URL")
        api_key = os.environ.get("LIVEKIT_API_KEY")
        api_secret = os.environ.get("LIVEKIT_API_SECRET")

        if not url:
            raise ValueError("LIVEKIT_URL environment variable is required")
        if not api_key:
            raise ValueError("LIVEKIT_API_KEY environment variable is required")
        if not api_secret:
            raise ValueError("LIVEKIT_API_SECRET environment variable is required")

        return cls(url=url, api_key=api_key, api_secret=api_secret)

    def validate(self) -> bool:
        """Validate that all required configuration values are present."""
        return bool(self.url and self.api_key and self.api_secret)


@dataclass
class VoiceConfig:
    """Voice pipeline configuration."""

    stt_provider: str = "openai"
    stt_model: str = "whisper-1"
    stt_language: str = "en"

    tts_provider: str = "openai"
    tts_voice: str = "alloy"
    tts_speed: float = 1.0
    tts_model: str = "tts-1"

    vad_provider: str = "silero"

    agent_instructions: str = (
        "You are a travel assistant. Keep responses brief and conversational."
    )

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        """Load voice configuration from environment variables with defaults."""
        return cls(
            stt_provider=os.environ.get("STT_PROVIDER", "openai"),
            stt_model=os.environ.get("STT_MODEL", "whisper-1"),
            stt_language=os.environ.get("STT_LANGUAGE", "en"),
            tts_provider=os.environ.get("TTS_PROVIDER", "openai"),
            tts_voice=os.environ.get("TTS_VOICE", "alloy"),
            tts_speed=float(os.environ.get("TTS_SPEED", "1.0")),
            tts_model=os.environ.get("TTS_MODEL", "tts-1"),
            vad_provider=os.environ.get("VAD_PROVIDER", "silero"),
            agent_instructions=os.environ.get(
                "AGENT_INSTRUCTIONS",
                "You are a travel assistant. Keep responses brief and conversational.",
            ),
        )


# Global configuration instances
livekit_config: Optional[LiveKitConfig] = None
voice_config: Optional[VoiceConfig] = None


def init_config():
    """Initialize global configuration from environment variables."""
    global livekit_config, voice_config

    livekit_config = LiveKitConfig.from_env()
    voice_config = VoiceConfig.from_env()

    print("✅ LiveKit configuration loaded:")
    print(f"   URL: {livekit_config.url}")
    print(f"   STT: {voice_config.stt_provider} ({voice_config.stt_model})")
    print(f"   TTS: {voice_config.tts_provider} ({voice_config.tts_voice})")
    print(f"   VAD: {voice_config.vad_provider}")

    return livekit_config, voice_config


def get_livekit_url() -> str:
    """Get LiveKit URL from config or environment."""
    if livekit_config:
        return livekit_config.url
    return os.environ.get("LIVEKIT_URL", "")


def generate_token(room_name: str, participant_name: str) -> str:
    """Generate JWT access token for LiveKit room authentication (valid 1 hour)."""
    if livekit_config:
        api_key = livekit_config.api_key
        api_secret = livekit_config.api_secret
    else:
        api_key = os.environ.get("LIVEKIT_API_KEY")
        api_secret = os.environ.get("LIVEKIT_API_SECRET")

    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name)
    token.with_name(participant_name)
    token.with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        )
    )
    token.with_ttl(timedelta(hours=1))
    return token.to_jwt()


class SimpleLLMStream(llm.LLMStream):
    """Wraps backend text response into LiveKit streaming format for TTS."""

    def __init__(
        self,
        text: str,
        llm_instance: llm.LLM,
        chat_ctx: llm.ChatContext,
        tools: list | None = None,
        conn_options: dict | None = None,
    ):
        super().__init__(
            llm=llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._text = text

    async def _run(self) -> None:
        """Convert backend response to ChatChunk for TTS processing."""
        import uuid

        text_content = self._text
        if isinstance(text_content, list):
            text_content = " ".join(str(item) for item in text_content)
        elif not isinstance(text_content, str):
            text_content = str(text_content)

        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(content=text_content, role="assistant"),
            )
        )

    async def aclose(self) -> None:
        await super().aclose()


class LangGraphLLMAdapter(llm.LLM):
    """Bridges LiveKit voice pipeline with Travel Assistant multi-model backend."""

    def __init__(self, graph, user_id: str = "voice_user"):
        super().__init__()
        self.graph = graph
        self.user_id = user_id
        self.conversation_history = []

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list | None = None,
        tool_choice: str | None = None,
        conn_options: dict | None = None,
        **kwargs,
    ) -> llm.LLMStream:
        """Extract user message, route through backend, return stream for TTS."""
        user_message = ""
        for item in reversed(chat_ctx.items):
            if item.role == "user":
                if isinstance(item.content, str):
                    user_message = item.content
                elif isinstance(item.content, list):
                    for part in item.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_message = part.get("text", "")
                            break
                        elif isinstance(part, str):
                            user_message = part
                            break
                    if not user_message:
                        user_message = " ".join(
                            str(p) for p in item.content if isinstance(p, str)
                        )
                else:
                    user_message = str(item.content)
                break

        if not user_message:
            user_message = "Hello"

        user_message = str(user_message)

        print(f"\n🔵 CALLING TRAVEL ASSISTANT: {user_message[:100]}...")

        backend_response = self.graph.process_message(
            user_id=self.user_id,
            user_input=user_message,
            conversation_history=self.conversation_history,
        )

        print(
            f"🟢 BACKEND RESPONSE ({len(backend_response)} chars): {backend_response[:200]}...\n"
        )

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": backend_response}
        )

        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return SimpleLLMStream(backend_response, self, chat_ctx, tools, conn_options)


def create_graph():
    """Initialize Travel Assistant backend (multi-model, LangFuse, Mem0, RAG)."""
    print("🔧 Initializing Travel Assistant backend...")
    graph = TravelAssistantGraph()
    print("✅ Travel Assistant backend ready")
    return graph


def create_langgraph_adapter(user_id: str = "voice_user"):
    """Create Travel Assistant adapter for LiveKit voice pipeline."""
    graph = create_graph()
    return LangGraphLLMAdapter(graph, user_id)


def create_voice_agent(user_id: str = "voice_user") -> Agent:
    """Create voice agent with VAD, STT, LLM, and TTS pipeline using config."""
    global voice_config

    if not voice_config:
        voice_config = VoiceConfig.from_env()

    travel_assistant_llm = create_langgraph_adapter(user_id)

    # Initialize STT based on config
    if voice_config.stt_provider == "openai":
        stt = openai.STT(
            model=voice_config.stt_model, language=voice_config.stt_language
        )
    else:
        stt = openai.STT()  # Default fallback

    # Initialize TTS based on config
    if voice_config.tts_provider == "openai":
        tts = openai.TTS(
            voice=voice_config.tts_voice,
            speed=voice_config.tts_speed,
            model=voice_config.tts_model,
        )
    else:
        tts = openai.TTS()  # Default fallback

    # Initialize VAD based on config
    if voice_config.vad_provider == "silero":
        vad = silero.VAD.load()
    else:
        vad = silero.VAD.load()  # Default fallback

    return Agent(
        vad=vad,
        stt=stt,
        llm=travel_assistant_llm,
        tts=tts,
        instructions=voice_config.agent_instructions,
    )


server = AgentServer()


@server.rtc_session(agent_name="test-assistant-travel")
async def entrypoint(ctx: JobContext):
    """Setup voice pipeline when agent joins room."""
    global livekit_config, voice_config

    if not livekit_config or not voice_config:
        init_config()

    print(f"🔥 Agent received job for room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    print(f"✅ Connected to room: {ctx.room.name}")

    print("🔧 Creating voice agent...")
    user_id = f"voice_{ctx.room.name}"
    assistant = create_voice_agent(user_id)
    print("✓ Voice agent created")

    session = AgentSession()
    print("🚀 Starting session...")
    await session.start(assistant, room=ctx.room)
    print("✓ Session started")


if __name__ == "__main__":
    print("🎙️  Travel Assistant Voice Agent")
    print("=" * 50)

    try:
        init_config()
        print("\n🚀 Starting LiveKit agent server...")
        cli.run_app(server)
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nRequired environment variables:")
        print("  - LIVEKIT_URL")
        print("  - LIVEKIT_API_KEY")
        print("  - LIVEKIT_API_SECRET")
        print("  - OPENAI_API_KEY")
        print("\nOptional environment variables:")
        print("  - STT_PROVIDER (default: openai)")
        print("  - STT_MODEL (default: whisper-1)")
        print("  - STT_LANGUAGE (default: en)")
        print("  - TTS_PROVIDER (default: openai)")
        print("  - TTS_VOICE (default: alloy)")
        print("  - TTS_SPEED (default: 1.0)")
        print("  - TTS_MODEL (default: tts-1)")
        print("  - VAD_PROVIDER (default: silero)")
        print("  - AGENT_INSTRUCTIONS (default: travel assistant prompt)")
        exit(1)
