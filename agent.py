import os
import certifi
import logging
import asyncio
import httpx
from typing import Annotated
from dotenv import load_dotenv
from datetime import datetime, timedelta

os.environ['SSL_CERT_FILE'] = certifi.where()

from livekit import agents, api, rtc
from livekit.agents import AgentSession, Agent, JobContext
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero, sarvam
from livekit.agents import llm

load_dotenv(".env")

# =============================================================================
#  LOGGING — structured with prefix for easy filtering
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("convodesk-agent")

import config

# =============================================================================
#  HTTP CLIENT HELPER — HIGH TIMEOUTS + NETWORK RESILIENCE
# =============================================================================

async def _call_backend(
    method: str,
    path: str,
    params: dict = None,
    json: dict = None,
    timeout: float = 45.0,
) -> dict:
    """
    Makes authenticated request to Spring Boot backend.
    - High default timeout to survive slow telephony/DB/calendar ops
    - Catches network issues gracefully
    - Logs full request/response for debugging
    """
    url = f"{config.BACKEND_URL}{path}"
    headers = {"X-Agent-Secret": config.AGENT_SECRET}

    logger.info(f"▶ {method} {url} (timeout={timeout}s)")
    if params: logger.info(f"  params: {params}")
    if json:   logger.info(f"  body:   {json}")

    try:
        async with httpx.AsyncClient() as client:
            r = await client.request(
                method, url,
                params=params,
                json=json,
                headers=headers,
                timeout=timeout,
            )
            logger.info(f"◀ {method} {url} → HTTP {r.status_code}")

            if r.status_code >= 400:
                logger.error(f"  ERROR response body: {r.text}")
                return {"error": r.text, "status": r.status_code}

            data = r.json()
            logger.info(f"  response: {data}")
            return data

    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
            httpx.WriteError, httpx.RemoteProtocolError, httpx.HTTPError) as e:
        logger.warning(f"  NETWORK ISSUE on {method} {url}: {type(e).__name__} — {e}")
        return {"error": f"network_error: {str(e)}"}

    except Exception as e:
        logger.error(f"  FAILED: {method} {url} — {e}")
        return {"error": str(e)}


# =============================================================================
#  SPRING BOOT API CALLS
# =============================================================================

async def fetch_tenant_config(called_number: str, customer_phone: str) -> dict:
    logger.info(f"[CONFIG] Fetching tenant config — called: {called_number}, caller: {customer_phone}")
    data = await _call_backend(
        "GET", "/telephony/context",
        params={"calledNumber": called_number, "customerPhone": customer_phone},
        timeout=15.0,
    )
    if data.get("error"):
        logger.error(f"[CONFIG] Failed to load tenant config: {data}")
        return {}

    logger.info(
        f"[CONFIG] Loaded — business: '{data.get('businessName')}' (id={data.get('businessId')}), "
        f"callLogId: {data.get('callLogId')}, "
        f"bookingEnabled: {data.get('bookingEnabled')}, "
        f"faqs: {len(data.get('faqs', []))} entries"
    )
    return data


async def post_segment(call_log_id: int, speaker: str, text: str):
    """POST /telephony/calls/{id}/segment — correct path matching TelephonyContextController"""
    if not call_log_id or not text.strip():
        return
    logger.debug(f"[SEGMENT] call={call_log_id} [{speaker}]: {text[:60]}...")
    await _call_backend(
        "POST", f"/telephony/calls/{call_log_id}/segment",
        json={"sender": speaker, "text": text},
        timeout=8.0,
    )


async def save_summary(call_log_id: int, summary: str, full_transcript: str):
    if not call_log_id:
        return
    logger.info(f"[SUMMARY] Saving summary for call {call_log_id} — {len(full_transcript)} chars transcript")
    result = await _call_backend(
        "PATCH", f"/telephony/calls/{call_log_id}/summary",
        json={"summary": summary, "fullTranscript": full_transcript},
        timeout=30.0,
    )
    if result.get("error"):
        logger.error(f"[SUMMARY] Failed to save summary: {result}")
    else:
        logger.info(f"[SUMMARY] Saved successfully for call {call_log_id}")


async def end_call(call_log_id: int):
    """POST /telephony/calls/{id}/end — agent-accessible endpoint on TelephonyContextController"""
    if not call_log_id:
        return
    logger.info(f"[END_CALL] Ending call {call_log_id}")
    result = await _call_backend(
        "POST", f"/telephony/calls/{call_log_id}/end",
        timeout=20.0,
    )
    if result.get("error"):
        logger.error(f"[END_CALL] Failed: {result}")
    else:
        logger.info(
            f"[END_CALL] Call {call_log_id} marked COMPLETED — "
            f"duration: {result.get('durationSeconds', '?')}s"
        )


async def generate_summary_via_groq(transcript_text: str) -> str:
    if not transcript_text.strip():
        return ""

    logger.info(f"[SUMMARY_GEN] Generating summary for {len(transcript_text)} char transcript")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Summarize this call transcript in 2-3 sentences. "
                                "Include: the caller's main query, what was resolved, "
                                "and any action taken (appointment booked, transferred, etc.). "
                                "Be concise and factual."
                            )
                        },
                        {"role": "user", "content": transcript_text}
                    ],
                    "max_tokens": 200,
                },
                timeout=30.0,
            )
            summary = r.json()["choices"][0]["message"]["content"]
            logger.info(f"[SUMMARY_GEN] Generated: {summary}")
            return summary
    except Exception as e:
        logger.error(f"[SUMMARY_GEN] Failed: {e}")
        return ""


# =============================================================================
#  PROVIDER BUILDERS
# =============================================================================

SARVAM_VOICES = {"anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"}

def _build_tts(config_provider: str = None, config_voice: str = None, stt_language: str = None):
    provider = (config_provider or os.getenv("TTS_PROVIDER", config.DEFAULT_TTS_PROVIDER)).lower()

    if config_voice in SARVAM_VOICES:
        provider = "sarvam"

    if provider == "sarvam":
        voice = config_voice if config_voice in SARVAM_VOICES else "anushka"
        lang_code = "hi-IN" if stt_language in ["hi", "multi"] else os.getenv("SARVAM_LANGUAGE", "en-IN")
        logger.info(f"[TTS] Sarvam — voice: {voice}, lang: {lang_code}")
        return sarvam.TTS(
            model=config.SARVAM_MODEL,
            speaker=voice,
            target_language_code=lang_code,
        )

    if provider == "cartesia":
        logger.info(f"[TTS] Cartesia — model: {config.CARTESIA_MODEL}")
        return cartesia.TTS(model=config.CARTESIA_MODEL, voice=config.CARTESIA_VOICE)

    if provider == "deepgram":
        voice = config_voice or "aura-asteria-en"
        logger.info(f"[TTS] Deepgram Aura — voice: {voice}")
        return deepgram.TTS(model=voice)

    voice = config_voice or os.getenv("OPENAI_TTS_VOICE", config.DEFAULT_TTS_VOICE)
    logger.info(f"[TTS] OpenAI — voice: {voice}")
    return openai.TTS(model="tts-1", voice=voice)


def _build_llm(config_provider: str = None, config_model: str = None):
    provider = (config_provider or os.getenv("LLM_PROVIDER", config.DEFAULT_LLM_PROVIDER)).lower()
    if provider == "groq":
        model = config_model or os.getenv("GROQ_MODEL", config.GROQ_MODEL)
        logger.info(f"[LLM] Groq — model: {model}")
        return openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model=model,
        )
    model = config_model or config.DEFAULT_LLM_MODEL
    logger.info(f"[LLM] OpenAI — model: {model}")
    return openai.LLM(model=model, api_key=os.getenv("OPENAI_API_KEY"))


def _build_stt(stt_language: str = None):
    """
    FIX: Use config.STT_MODEL (nova-2) instead of hardcoded nova-3.
    nova-3 is not a valid Deepgram model name and silently falls back,
    causing poor transcription quality.
    """
    language = stt_language or os.getenv("DEEPGRAM_LANGUAGE", config.STT_LANGUAGE)
    model = os.getenv("DEEPGRAM_MODEL", config.STT_MODEL)  # nova-2 from config
    logger.info(f"[STT] Deepgram {model} — language: {language}")
    return deepgram.STT(
        model=model,
        language=language,
        smart_format=True,
    )


# =============================================================================
#  SEGMENT STREAMER
# =============================================================================

class SegmentStreamer:
    def __init__(self, call_log_id: int):
        self.call_log_id = call_log_id
        self._buffer: list = []
        self._all: list = []
        self._lock = asyncio.Lock()
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(f"[STREAMER] Started for call {call_log_id}")

    async def push(self, speaker: str, text: str):
        if not text.strip():
            return
        async with self._lock:
            entry = {"speaker": speaker, "text": text}
            self._buffer.append(entry)
            self._all.append(entry)
        logger.info(f"[TRANSCRIPT] [{speaker.upper()}]: {text}")

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

        logger.info(f"[STREAMER] Flushing {len(batch)} segments to backend")
        for seg in batch:
            await post_segment(self.call_log_id, seg["speaker"], seg["text"])

    def get_full_transcript(self) -> str:
        return "\n".join([
            f"{s['speaker'].upper()}: {s['text']}"
            for s in self._all
        ])

    async def close(self):
        self._task.cancel()
        await self._do_flush()
        logger.info(f"[STREAMER] Closed — total segments: {len(self._all)}")


# =============================================================================
#  TOOLS — ENFORCED BOOKING FLOW
# =============================================================================

class AssistantTools(llm.ToolContext):
    def __init__(self, ctx: JobContext, caller_phone: str,
                 transfer_number: str, call_log_id: int,
                 business_id: int, booking_enabled: bool):
        super().__init__(tools=[])
        self.ctx             = ctx
        self.caller_phone    = caller_phone
        self.transfer_number = transfer_number or config.DEFAULT_TRANSFER_NUMBER
        self.call_log_id     = call_log_id
        self.business_id     = business_id
        self.booking_enabled = booking_enabled

        logger.info(
            f"[TOOLS] Initialized — businessId: {business_id}, "
            f"callLogId: {call_log_id}, bookingEnabled: {booking_enabled}, "
            f"callerPhone: {caller_phone}, transferNumber: {transfer_number}"
        )

    # ------------------------------------------------------------------
    #  TRANSFER CALL
    # ------------------------------------------------------------------
    @llm.function_tool(description="Transfer the call to a human staff member or receptionist.")
    async def transfer_call(
        self,
        reason: Annotated[str, llm.TypeInfo(
            description="Reason the caller needs to be transferred"
        )] = "User requested transfer",
    ):
        participant_identity = None
        for p in self.ctx.room.remote_participants.values():
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                participant_identity = p.identity
                break

        if not participant_identity:
            return "I'm sorry, I couldn't identify your call line."
        if not self.transfer_number:
            return "I'm sorry, no transfer number is configured for this business."

        try:
            await self.ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self.ctx.room.name,
                    participant_identity=participant_identity,
                    transfer_to=self.transfer_number,
                    play_dialtone=True,
                )
            )
            logger.info(f"[TOOL:transfer_call] Transfer initiated to {self.transfer_number}")
            return "One moment please, transferring you to our team now."
        except Exception as e:
            logger.error(f"[TOOL:transfer_call] Transfer failed: {e}")
            return "I encountered an error during transfer. Please stay on the line."

    # ------------------------------------------------------------------
    #  GET STAFF
    # ------------------------------------------------------------------
    @llm.function_tool(description=(
        "Get the list of available staff or service providers for this business. "
        "Call this at the START of any booking conversation before asking for a date."
    ))
    async def get_staff(
        self,
        _dummy: Annotated[str, llm.TypeInfo(
            description="Pass empty string."
        )] = "",
    ):
        if not self.booking_enabled:
            return "Booking is not enabled for this business."

        data = await _call_backend(
            "GET", "/calendar/staff",
            params={"businessId": self.business_id},
        )

        if data.get("error"):
            return "Any available staff member."

        staff = data.get("staff", [])
        if not staff:
            return "Any available staff member."
        if len(staff) == 1:
            return f"Available: {staff[0]}."
        return "Available staff: " + ", ".join(staff[:-1]) + " and " + staff[-1] + "."

    # ------------------------------------------------------------------
    #  CHECK AVAILABILITY
    # ------------------------------------------------------------------
    @llm.function_tool(description=(
        "Check available appointment slots for a given date. "
        "Always call this before booking to confirm the slot is actually free."
    ))
    async def check_availability(
        self,
        date: Annotated[str, llm.TypeInfo(
            description="Date in YYYY-MM-DD format. Convert 'tomorrow' or 'next Monday' to this format."
        )],
        staff_name: Annotated[str, llm.TypeInfo(
            description="Staff member name, or 'Any' if caller has no preference."
        )] = "Any",
    ):
        if not self.booking_enabled:
            return "Online booking is not available. Please call during working hours."

        data = await _call_backend(
            "GET", "/calendar/slots",
            params={
                "businessId": self.business_id,
                "staffName":  staff_name,
                "date":       date,
            },
        )

        if data.get("error"):
            return "I couldn't check availability right now. Please try again."

        slots = data.get("slots", [])

        if not slots:
            tomorrow = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            return (f"No slots available on {date}. "
                    f"Would you like me to check {tomorrow} or another date?")

        shown = slots[:4]
        more  = f" and {len(slots) - 4} more available" if len(slots) > 4 else ""
        return f"Available on {date}: {', '.join(shown)}{more}. Which time works for you?"

    # ------------------------------------------------------------------
    #  BOOK APPOINTMENT
    # ------------------------------------------------------------------
    @llm.function_tool(description=(
        "Book an appointment AFTER the caller has confirmed their name, date, time, and staff. "
        "Do NOT call this without first calling check_availability. "
        "Do NOT guess the customer name — ALWAYS ask the user."
    ))
    async def book_appointment(
        self,
        customer_name: Annotated[str, llm.TypeInfo(
            description="Full name of the customer. Always ask if not provided."
        )],
        preferred_date: Annotated[str, llm.TypeInfo(
            description="Date in YYYY-MM-DD format."
        )],
        preferred_time: Annotated[str, llm.TypeInfo(
            description="Time in h:mm AM/PM format e.g. '10:30 AM'. Must be from available slots."
        )],
        staff_name: Annotated[str, llm.TypeInfo(
            description="Staff member name or 'Any' if no preference."
        )] = "Any",
        customer_email: Annotated[str, llm.TypeInfo(
            description="Customer email if provided. Leave blank if not given."
        )] = "",
    ):
        logger.info(
            f"[TOOL:book_appointment] businessId={self.business_id}, "
            f"customer='{customer_name}', phone={self.caller_phone}, "
            f"date={preferred_date}, time={preferred_time}, staff={staff_name}"
        )

        # Enforce: must have a real customer name
        if not customer_name or len(str(customer_name).strip()) < 2 or \
                str(customer_name).lower().strip() in {"", "unknown", "the customer", "caller", "guest"}:
            logger.warning(f"[TOOL:book_appointment] Missing/invalid customer name — asking user")
            return "To book the appointment I need your full name. Could you please tell me your name?"

        if not self.booking_enabled:
            return "Online booking is not available. Please call during working hours."

        payload = {
            "businessId":    self.business_id,
            "callLogId":     self.call_log_id,
            "customerName":  customer_name,
            "customerPhone": self.caller_phone,
            "customerEmail": customer_email,
            "staffName":     staff_name,
            "preferredDate": preferred_date,
            "preferredTime": preferred_time,
        }

        result = await _call_backend(
            "POST", "/calendar/book",
            json=payload,
            timeout=20.0,
        )

        logger.info(f"[TOOL:book_appointment] Result: {result}")

        if result.get("error"):
            logger.error(f"[TOOL:book_appointment] Backend error: {result}")
            return "I wasn't able to complete the booking right now. Let me transfer you to our team."

        if result.get("success"):
            logger.info(
                f"[TOOL:book_appointment] ✅ Booked — "
                f"appointmentId={result.get('appointmentId')}, "
                f"date={result.get('date')}, time={result.get('time')}, "
                f"staff={result.get('staffName')}, provider={result.get('provider')}"
            )
            return (
                f"Done! Appointment confirmed — "
                f"{result['date']} at {result['time']} "
                f"with {result['staffName']} at {result['businessName']}. "
                f"You'll receive a WhatsApp confirmation shortly."
            )
        else:
            msg = result.get("message", "I couldn't complete the booking. Please try another time.")
            logger.warning(f"[TOOL:book_appointment] Booking rejected: {msg}")
            return msg


# =============================================================================
#  MAIN ENTRYPOINT
# =============================================================================

async def entrypoint(ctx: JobContext):
    logger.info(f"[ENTRYPOINT] New job — room: {ctx.room.name}")
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    logger.info(f"[ENTRYPOINT] Participant joined — kind: {participant.kind}, identity: {participant.identity}")

    caller_number = None
    called_number = None

    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        caller_number = participant.attributes.get("sip.phoneNumber")
        called_number = participant.attributes.get("sip.trunkPhoneNumber")
        logger.info(f"[ENTRYPOINT] INBOUND SIP — caller: {caller_number}, called: {called_number}")

    tenant = {}
    if called_number and caller_number:
        tenant = await fetch_tenant_config(called_number, caller_number)

    call_log_id     = tenant.get("callLogId")
    business_id     = tenant.get("businessId")
    business_name   = tenant.get("businessName", "our business")
    system_prompt   = tenant.get("systemPrompt") or config.SYSTEM_PROMPT
    greeting        = tenant.get("greeting")     or config.INITIAL_GREETING
    tts_provider    = tenant.get("ttsProvider")
    tts_voice       = tenant.get("ttsVoice")
    llm_provider    = tenant.get("llmProvider")
    llm_model       = tenant.get("llmModel")
    stt_language    = tenant.get("sttLanguage")
    transfer_num    = tenant.get("transferNumber") or config.DEFAULT_TRANSFER_NUMBER
    booking_enabled = tenant.get("bookingEnabled", False)
    faqs            = tenant.get("faqs", [])

    # Append strict booking rules to system prompt
    BOOKING_RULES = """
BOOKING RULES — FOLLOW STRICTLY:
1. Never call book_appointment unless the caller has explicitly given their FULL name.
2. Always call check_availability first for any date.
3. Ask the caller for name, confirm date/time, and get explicit "yes" before booking.
4. If any detail is missing when you reach book_appointment, the tool will ask the user for it.
"""
    system_prompt += BOOKING_RULES
    logger.info("[ENTRYPOINT] Added strict booking rules to system prompt")

    # Inject FAQs into system prompt
    if faqs:
        faq_lines = "\n".join([
            f"Q: {f.get('question', '')}\nA: {f.get('answer', '')}"
            for f in faqs
        ])
        system_prompt += f"\n\n## Knowledge Base\n{faq_lines}"

    tools_ctx = AssistantTools(
        ctx, caller_number, transfer_num,
        call_log_id, business_id, booking_enabled
    )
    streamer = SegmentStreamer(call_log_id) if call_log_id else None

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=_build_stt(stt_language),
        llm=_build_llm(llm_provider, llm_model),
        tts=_build_tts(tts_provider, tts_voice, stt_language),
    )

    if streamer:
        @session.on("user_speech_committed")
        def on_user_speech(event):
            asyncio.create_task(streamer.push("caller", event.transcript))

        @session.on("agent_speech_committed")
        def on_agent_speech(event):
            asyncio.create_task(streamer.push("agent", event.transcript))

    # FIX: RoomInputOptions is deprecated — use room_options with RoomOptions
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=system_prompt,
            tools=list(tools_ctx.function_tools.values()),
        ),
        room_input_options=agents.RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    await session.generate_reply(instructions=greeting)

    disconnected = asyncio.Event()
    ctx.room.on("disconnected", lambda *_: disconnected.set())
    await disconnected.wait()
    logger.info("[ENTRYPOINT] Room disconnected")

    if streamer:
        await streamer.close()

    # CLEANUP — generate summary then end call
    if call_log_id and streamer:
        transcript_text = streamer.get_full_transcript()
        if transcript_text.strip():
            try:
                summary = await generate_summary_via_groq(transcript_text)
                await save_summary(call_log_id, summary, transcript_text)
            except Exception as e:
                logger.error(f"[CLEANUP] Summary generation/save failed: {e}")

    if call_log_id:
        try:
            await end_call(call_log_id)
        except Exception as e:
            logger.error(f"[CLEANUP] End call failed: {e}")

    logger.info(f"[ENTRYPOINT] ✅ Call completed — room: {ctx.room.name}, callLogId: {call_log_id}")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="convodesk-agent",
        )
    )