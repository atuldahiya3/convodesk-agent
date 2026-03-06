import os
import certifi
import logging
import json
import asyncio
import httpx
from typing import Annotated, Optional
from dotenv import load_dotenv

# Fix for macOS SSL Certificate errors
os.environ['SSL_CERT_FILE'] = certifi.where()

from livekit import agents, api, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions, JobContext
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    sarvam,
)
from livekit.agents import llm

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telephony-agent")

import config

# =============================================================================
#  SPRING BOOT CLIENT
# =============================================================================

async def fetch_tenant_config(called_number: str, customer_phone: str) -> dict:
    """
    Fetches per-tenant config from Spring Boot.
    called_number  = sip.trunkPhoneNumber = your Vobiz DID = identifies the business
    customer_phone = sip.phoneNumber      = the caller

    Returns {} on any failure — agent then falls back to static config.py values.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{config.BACKEND_URL}/telephony/context",
                params={"calledNumber": called_number, "customerPhone": customer_phone},
                headers={"X-Agent-Secret": config.AGENT_SECRET},
                timeout=8.0,
            )
            r.raise_for_status()
            data = r.json()
            logger.info(f"Tenant config loaded — business: {data.get('businessName')}, callLogId: {data.get('callLogId')}")
            return data
    except Exception as e:
        logger.error(f"Could not fetch tenant config: {e}. Falling back to config.py defaults.")
        return {}


async def post_segment(call_log_id: int, speaker: str, text: str):
    if not call_log_id:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{config.BACKEND_URL}/calls/segment",
                json={"callId": call_log_id, "sender": speaker, "text": text},
                headers={"X-Agent-Secret": config.AGENT_SECRET},
                timeout=5.0,
            )
    except Exception as e:
        logger.warning(f"Segment post failed: {e}")


async def end_call(call_log_id: int):
    if not call_log_id:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{config.BACKEND_URL}/business/calls/{call_log_id}/end",
                headers={"X-Agent-Secret": config.AGENT_SECRET},
                timeout=5.0,
            )
        logger.info(f"Call {call_log_id} marked as ended.")
    except Exception as e:
        logger.warning(f"end_call failed: {e}")


# =============================================================================
#  PROVIDER BUILDERS — same as your original working version
# =============================================================================

def _build_tts(config_provider: str = None, config_voice: str = None):
    provider = (config_provider or os.getenv("TTS_PROVIDER", config.DEFAULT_TTS_PROVIDER)).lower()

    # Force Sarvam for specific Indian voices
    if config_voice in ["anushka", "aravind", "amartya", "dhruv"]:
        provider = "sarvam"

    if provider == "cartesia":
        return cartesia.TTS(model=config.CARTESIA_MODEL, voice=config.CARTESIA_VOICE)

    if provider == "sarvam":
        voice = config_voice or os.getenv("SARVAM_VOICE", "anushka")
        return sarvam.TTS(model=config.SARVAM_MODEL, speaker=voice, target_language_code=config.SARVAM_LANGUAGE)

    # Default to OpenAI
    voice = config_voice or os.getenv("OPENAI_TTS_VOICE", config.DEFAULT_TTS_VOICE)
    return openai.TTS(model="tts-1", voice=voice)


def _build_llm(config_provider: str = None, config_model: str = None):
    provider = (config_provider or os.getenv("LLM_PROVIDER", config.DEFAULT_LLM_PROVIDER)).lower()
    if provider == "groq":
        return openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model=config_model or os.getenv("GROQ_MODEL", config.GROQ_MODEL),
        )
    return openai.LLM(
        model=config_model or config.DEFAULT_LLM_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
# =============================================================================
#  TOOLS — same as your original working version
# =============================================================================

class AssistantTools(llm.ToolContext):
    def __init__(self, ctx: JobContext, phone: str = None, transfer_number: str = None):
        super().__init__(tools=[])
        self.ctx = ctx
        self.phone = phone
        self.transfer_number = transfer_number or config.DEFAULT_TRANSFER_NUMBER

    @llm.function_tool(description="Transfer the call to a human support agent.")
    async def transfer_call(
        self,
        reason: Annotated[
            str,
            llm.TypeInfo(description="The reason why the caller needs to be transferred")
        ] = "User requested transfer"
    ):
        logger.info(f"Transfer requested. Reason: {reason}")

        participant_identity = None
        for p in self.ctx.room.remote_participants.values():
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                participant_identity = p.identity
                break

        if not participant_identity:
            return "I'm sorry, I couldn't identify your call line to perform the transfer."

        try:
            await self.ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self.ctx.room.name,
                    participant_identity=participant_identity,
                    transfer_to=self.transfer_number,
                    play_dialtone=True
                )
            )
            return "One moment please, I am transferring your call to a human representative."
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return "I encountered an error while trying to transfer your call. Please stay on the line."


# =============================================================================
#  SEGMENT STREAMER — buffers transcript, flushes to Spring Boot every 5s
# =============================================================================

class SegmentStreamer:
    def __init__(self, call_log_id: int):
        self.call_log_id = call_log_id
        self._buffer: list = []
        self._lock = asyncio.Lock()
        self._task = asyncio.create_task(self._flush_loop())

    async def push(self, speaker: str, text: str):
        async with self._lock:
            self._buffer.append({"speaker": speaker, "text": text})

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(5)
            await self._do_flush()

    async def _do_flush(self):
        async with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()
        for seg in batch:
            await post_segment(self.call_log_id, seg["speaker"], seg["text"])

    async def close(self):
        self._task.cancel()
        await self._do_flush()  # final flush on call end


# =============================================================================
#  MAIN ENTRYPOINT
# =============================================================================

async def entrypoint(ctx: JobContext):
    logger.info(f"--- New Job Received in room: {ctx.room.name} ---")

    # 1. Connect immediately to acknowledge the job
    await ctx.connect()

    # 2. Wait for the SIP participant (the Vobiz caller)
    participant = await ctx.wait_for_participant()

    caller_number = None
    called_number = None

    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        # LiveKit sets these automatically from the SIP INVITE headers
        caller_number = participant.attributes.get("sip.phoneNumber")       # +919996978591
        called_number = participant.attributes.get("sip.trunkPhoneNumber")  # +918049280412
        logger.info(f"INBOUND SIP — caller: {caller_number}, called: {called_number}")

    # 3. Fetch per-tenant config from Spring Boot
    #    On any failure this returns {} and everything below falls back to config.py
    tenant = {}
    if called_number and caller_number:
        tenant = await fetch_tenant_config(called_number, caller_number)

    # 4. Extract values — tenant config wins, config.py is the fallback
    call_log_id  = tenant.get("callLogId")
    system_prompt = tenant.get("systemPrompt") or config.SYSTEM_PROMPT
    greeting     = tenant.get("greeting")      or config.INITIAL_GREETING
    tts_provider = tenant.get("ttsProvider")   or None
    tts_voice    = tenant.get("ttsVoice")      or None
    llm_provider = tenant.get("llmProvider") or None
    llm_model    = tenant.get("llmModel")    or None
    transfer_num = tenant.get("transferNumber") or config.DEFAULT_TRANSFER_NUMBER
    faqs         = tenant.get("faqs", [])

    # 5. Inject FAQs into the system prompt
    if faqs:
        faq_lines = "\n".join([
            f"Q: {f.get('question', '')}\nA: {f.get('answer', '')}"
            for f in faqs
        ])
        system_prompt += f"\n\n## Knowledge Base\nUse the following to answer caller questions:\n\n{faq_lines}"

    logger.info(f"Starting session — business: '{tenant.get('businessName', 'default')}', "
                f"call_log_id: {call_log_id}, llm: {llm_provider or 'default'}, tts_voice: {tts_voice or 'default'}")

    # 6. Initialize tools and session — same pattern as your original
    fnc_ctx  = AssistantTools(ctx, caller_number, transfer_num)
    streamer = SegmentStreamer(call_log_id) if call_log_id else None

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model=config.STT_MODEL, language=config.STT_LANGUAGE),
        llm=_build_llm(llm_provider, llm_model),
        tts=_build_tts(tts_provider, tts_voice),
    )

    # 7. Wire transcript segments to Spring Boot (only if call_log_id was returned)
    if streamer:
        @session.on("user_speech_committed")
        def on_user_speech(event):
            asyncio.create_task(streamer.push("caller", event.transcript))

        @session.on("agent_speech_committed")
        def on_agent_speech(event):
            asyncio.create_task(streamer.push("agent", event.transcript))

    # 8. Start the agent — fix deprecated RoomInputOptions → RoomOptions
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=system_prompt,
            tools=list(fnc_ctx.function_tools.values())
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )

    # 9. Greet the caller
    await session.generate_reply(instructions=greeting)

    # 10. Wait for room to close, THEN clean up
    #     ctx.room.on("disconnected") fires when the SIP caller hangs up
    disconnected = asyncio.Event()
    ctx.room.on("disconnected", lambda *_: disconnected.set())
    await disconnected.wait()

    # 11. Cleanup after call ends
    if streamer:
        await streamer.close()
    if call_log_id:
        await end_call(call_log_id)
    logger.info(f"Call ended — room: {ctx.room.name}")

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="convodesk-agent",
        )
    )