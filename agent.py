import os
import certifi
import logging
import json
import asyncio
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

# --- PROVIDER BUILDERS ---

def _build_tts(config_provider: str = None, config_voice: str = None):
    provider = (config_provider or os.getenv("TTS_PROVIDER", config.DEFAULT_TTS_PROVIDER)).lower()
    
    # Force Sarvam for specific voices
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

def _build_llm(config_provider: str = None):
    provider = (config_provider or os.getenv("LLM_PROVIDER", config.DEFAULT_LLM_PROVIDER)).lower()
    if provider == "groq":
        return openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model=config.GROQ_MODEL,
        )
    return openai.LLM(model=config.DEFAULT_LLM_MODEL)

# --- TOOLS (FIXED) ---

class AssistantTools(llm.ToolContext):
    def __init__(self, ctx: JobContext, phone: str = None):
        # Pass tools=[] to satisfy the base class
        super().__init__(tools=[])
        self.ctx = ctx
        self.phone = phone

    @llm.function_tool(description="Transfer the call to a human support agent.")
    async def transfer_call(
        self,
        reason: Annotated[
            str, 
            llm.TypeInfo(description="The reason why the caller needs to be transferred")
        ] = "User requested transfer"
    ):
        """
        Transfers the current SIP call to a human receptionist.
        """
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
                    transfer_to=config.DEFAULT_TRANSFER_NUMBER,
                    play_dialtone=True
                )
            )
            return "One moment please, I am transferring your call to a human representative."
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return "I encountered an error while trying to transfer your call. Please stay on the line."
# --- MAIN ENTRYPOINT ---

async def entrypoint(ctx: JobContext):
    logger.info(f"--- New Job Received in room: {ctx.room.name} ---")
    
    # 1. Connect immediately to acknowledge the job and room
    await ctx.connect()
    
    phone_number = None
    config_dict = {}
    is_inbound = False

    # 2. Check for Outbound Metadata
    try:
        if ctx.job.metadata:
            data = json.loads(ctx.job.metadata)
            phone_number = data.get("phone_number")
            config_dict = data
    except Exception:
        pass

    # 3. Identify the Caller
    # For inbound, the SIP participant is already joining.
    participant = await ctx.wait_for_participant()
    
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        inbound_phone = participant.attributes.get("sip.phoneNumber")
        # If we don't have a phone number from metadata, it's definitely inbound
        if not phone_number:
            phone_number = inbound_phone
            is_inbound = True
            logger.info(f"INBOUND detected from: {phone_number}")

    # 4. Initialize Tools and Session
    fnc_ctx = AssistantTools(ctx, phone_number)
    
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model=config.STT_MODEL, language=config.STT_LANGUAGE),
        llm=_build_llm(config_dict.get("model_provider")),
        tts=_build_tts(config_dict.get("model_provider"), config_dict.get("voice_id")),
    )

    # 5. Start the AI Assistant
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=config.SYSTEM_PROMPT,
            tools=list(fnc_ctx.function_tools.values())
        ),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )

    # 6. Routing Logic: Greet or Dial
    if is_inbound:
        logger.info("Starting inbound greeting flow...")
        # Immediate greeting for callers
        await session.generate_reply(instructions="Greet the caller: 'Hello! Thank you for calling the clinic. How can I help you today?'")
    
    elif phone_number:
        # Check if the user is already here (dashboard dispatch)
        user_in_room = any(f"sip_{phone_number}" in p.identity for p in ctx.room.remote_participants.values())
        
        if not user_in_room:
            logger.info(f"Initiating dial-out to {phone_number}...")
            try:
                await ctx.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=ctx.room.name,
                        sip_trunk_id=config.SIP_TRUNK_ID,
                        sip_call_to=phone_number,
                        sip_number=config.YOUR_VOBIZ_NUMBER,
                        participant_identity=f"sip_{phone_number}",
                        wait_until_answered=True,
                    )
                )
                await session.generate_reply(instructions=config.INITIAL_GREETING)
            except Exception as e:
                logger.error(f"Outbound dial failed: {e}")
                ctx.shutdown()
        else:
            # User already in room from dashboard
            await session.generate_reply(instructions=config.INITIAL_GREETING)

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller", 
        )
    )