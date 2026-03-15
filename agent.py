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
from livekit.agents import AgentSession, Agent, JobContext, RoomInputOptions
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero, sarvam
from livekit.agents import llm

load_dotenv(".env")

# =============================================================================
#  LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("convodesk-agent")

import config

# =============================================================================
#  HTTP CLIENT HELPER
# =============================================================================

async def _call_backend(method, path, params=None, json=None, timeout=45.0):
    url = f"{config.BACKEND_URL}{path}"
    headers = {"X-Agent-Secret": config.AGENT_SECRET}
    logger.info(f"▶ {method} {url} (timeout={timeout}s)")
    if params: logger.info(f"  params: {params}")
    if json:   logger.info(f"  body:   {json}")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.request(method, url, params=params, json=json,
                                     headers=headers, timeout=timeout)
            logger.info(f"◀ {method} {url} → HTTP {r.status_code}")
            if r.status_code >= 400:
                logger.error(f"  ERROR: {r.text}")
                return {"error": r.text, "status": r.status_code}
            data = r.json()
            logger.info(f"  response: {data}")
            return data
    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
            httpx.WriteError, httpx.RemoteProtocolError, httpx.HTTPError) as e:
        logger.warning(f"  NETWORK ISSUE: {type(e).__name__} — {e}")
        return {"error": f"network_error: {str(e)}"}
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {"error": str(e)}


# =============================================================================
#  SPRING BOOT API CALLS
# =============================================================================

async def fetch_tenant_config(called_number, customer_phone):
    data = await _call_backend(
        "GET", "/telephony/context",
        params={"calledNumber": called_number, "customerPhone": customer_phone},
        timeout=15.0,
    )
    if data.get("error"):
        logger.error(f"[CONFIG] Failed: {data}")
        return {}
    logger.info(
        f"[CONFIG] Loaded — business: '{data.get('businessName')}' (id={data.get('businessId')}), "
        f"callLogId: {data.get('callLogId')}, bookingEnabled: {data.get('bookingEnabled')}, "
        f"faqs: {len(data.get('faqs', []))} entries"
    )
    return data


async def post_segment(call_log_id, speaker, text):
    if not call_log_id or not text.strip():
        return
    logger.info(f"[SEGMENT] Posting — call={call_log_id} [{speaker}]: {text[:80]}...")
    await _call_backend(
        "POST", f"/telephony/calls/{call_log_id}/segment",
        json={"sender": speaker, "text": text},
        timeout=8.0,
    )


async def save_summary(call_log_id, summary, full_transcript):
    if not call_log_id:
        return
    result = await _call_backend(
        "PATCH", f"/telephony/calls/{call_log_id}/summary",
        json={"summary": summary, "fullTranscript": full_transcript},
        timeout=30.0,
    )
    if result.get("error"):
        logger.error(f"[SUMMARY] Failed: {result}")
    else:
        logger.info(f"[SUMMARY] Saved for call {call_log_id}")


async def end_call(call_log_id):
    if not call_log_id:
        return
    result = await _call_backend("POST", f"/telephony/calls/{call_log_id}/end", timeout=20.0)
    if result.get("error"):
        logger.error(f"[END_CALL] Failed: {result}")
    else:
        logger.info(f"[END_CALL] Call {call_log_id} COMPLETED — duration: {result.get('durationSeconds','?')}s")


async def generate_summary(transcript_text):
    if not transcript_text.strip():
        return ""
    truncated = transcript_text[:8000]
    system = (
        "You are a call summarizer. Plain text only — no headings, bullets, markdown. "
        "2-3 sentences: caller's query, what was resolved, action taken. Max 500 chars."
    )
    payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": truncated}],
        "max_tokens": 150,
    }
    logger.info(f"[SUMMARY_GEN] Generating for {len(transcript_text)} chars")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                json={**payload, "model": "llama-3.3-70b-versatile"},
                timeout=30.0,
            )
            s = r.json()["choices"][0]["message"]["content"].strip()[:500]
            logger.info(f"[SUMMARY_GEN] Done ({len(s)} chars): {s}")
            return s
    except Exception as e:
        logger.error(f"[SUMMARY_GEN] Groq failed: {e}")
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={**payload, "model": "gpt-4o-mini"},
                    timeout=30.0,
                )
                s = r.json()["choices"][0]["message"]["content"].strip()[:500]
                logger.info(f"[SUMMARY_GEN] OpenAI fallback ({len(s)} chars): {s}")
                return s
        except Exception as e2:
            logger.error(f"[SUMMARY_GEN] OpenAI fallback failed: {e2}")
    return ""


# =============================================================================
#  PROVIDER BUILDERS
# =============================================================================

SARVAM_VOICES = {"anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"}


def _build_tts(config_provider=None, config_voice=None, stt_language=None):
    provider = (config_provider or os.getenv("TTS_PROVIDER", config.DEFAULT_TTS_PROVIDER)).lower()
    if config_voice in SARVAM_VOICES:
        provider = "sarvam"
    if provider == "sarvam":
        voice = config_voice if config_voice in SARVAM_VOICES else "anushka"
        lang_code = "hi-IN" if stt_language in ["hi", "multi"] else os.getenv("SARVAM_LANGUAGE", "en-IN")
        logger.info(f"[TTS] Sarvam — voice: {voice}, lang: {lang_code}")
        return sarvam.TTS(model=config.SARVAM_MODEL, speaker=voice, target_language_code=lang_code)
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


def _build_llm(config_provider=None, config_model=None):
    provider = (config_provider or os.getenv("LLM_PROVIDER", config.DEFAULT_LLM_PROVIDER)).lower()
    if provider == "groq":
        model = config_model or os.getenv("GROQ_MODEL", config.GROQ_MODEL)
        groq_llm = openai.LLM(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model=model,
        )
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            logger.info(f"[LLM] Groq — model: {model} (no fallback)")
            return groq_llm
        try:
            fb = llm.FallbackAdapter([groq_llm, openai.LLM(model="gpt-4o-mini", api_key=key)])
            logger.info(f"[LLM] Groq — model: {model} (FallbackAdapter → gpt-4o-mini)")
            return fb
        except AttributeError:
            logger.warning(f"[LLM] FallbackAdapter unavailable — Groq only")
            return groq_llm
    model = config_model or config.DEFAULT_LLM_MODEL
    logger.info(f"[LLM] OpenAI — model: {model}")
    return openai.LLM(model=model, api_key=os.getenv("OPENAI_API_KEY"))


def _build_stt(stt_language=None):
    language = stt_language or os.getenv("DEEPGRAM_LANGUAGE", config.STT_LANGUAGE)
    model = os.getenv("DEEPGRAM_MODEL", config.STT_MODEL)
    logger.info(f"[STT] Deepgram {model} — language: {language}")
    return deepgram.STT(
        model=model,
        language=language,
        smart_format=True,
        endpointing_ms=800,
    )


# =============================================================================
#  SEGMENT STREAMER
# =============================================================================

class SegmentStreamer:
    def __init__(self, call_log_id):
        self.call_log_id = call_log_id
        self._buffer = []
        self._all = []
        self._lock = asyncio.Lock()
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(f"[STREAMER] Started for call {call_log_id}")

    async def push(self, speaker, text):
        text = text.strip()
        if not text:
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
        logger.info(f"[STREAMER] Flushing {len(batch)} segments")
        for seg in batch:
            await post_segment(self.call_log_id, seg["speaker"], seg["text"])

    def get_full_transcript(self):
        return "\n".join(f"{s['speaker'].upper()}: {s['text']}" for s in self._all)

    async def close(self):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        await self._do_flush()
        logger.info(f"[STREAMER] Closed — total segments: {len(self._all)}")


# =============================================================================
#  CONVODESK AGENT
#
#  FIX (booking tools): The get_staff tool had a parameter named "_dummy".
#  Groq's API rejects any tool schema containing parameter names that start
#  with an underscore — this causes the ENTIRE tools array to be dropped
#  silently from the API request, so the LLM never sees any tools at all.
#  Renamed to "placeholder" which is valid in JSON Schema.
#
#  Session/agent wiring for livekit-agents 1.4.x (confirmed from crash log):
#    AgentSession.__init__() does NOT accept agent= keyword argument.
#    agent= must be passed to session.start() — that was always the correct API.
# =============================================================================

class ConvoDeskAgent(Agent):
    """Agent subclass with booking and transfer tools embedded as methods."""

    def __init__(self, ctx: JobContext, caller_phone: str, transfer_number: str,
                 call_log_id: int, business_id: int, booking_enabled: bool,
                 instructions: str):
        super().__init__(instructions=instructions)
        self._ctx            = ctx
        self.caller_phone    = caller_phone
        self.transfer_number = transfer_number or config.DEFAULT_TRANSFER_NUMBER
        self.call_log_id     = call_log_id
        self.business_id     = business_id
        self.booking_enabled = booking_enabled
        logger.info(
            f"[AGENT] ConvoDeskAgent — businessId={business_id}, callLogId={call_log_id}, "
            f"bookingEnabled={booking_enabled}, callerPhone={caller_phone}, "
            f"transferNumber={self.transfer_number}"
        )

    @llm.function_tool(description="Transfer the call to a human staff member.")
    async def transfer_call(
        self,
        reason: Annotated[str, llm.TypeInfo(description="Reason for transfer")] = "User requested",
    ) -> str:
        logger.info(f"[TOOL:transfer_call] Called — reason: {reason}")
        pid = None
        for p in self._ctx.room.remote_participants.values():
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                pid = p.identity
                break
        if not pid:
            return "Couldn't identify your call line."
        if not self.transfer_number:
            return "No transfer number configured."
        try:
            await self._ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self._ctx.room.name,
                    participant_identity=pid,
                    transfer_to=self.transfer_number,
                    play_dialtone=True,
                )
            )
            logger.info(f"[TOOL:transfer_call] Transfer → {self.transfer_number}")
            return "Transferring you now."
        except Exception as e:
            logger.error(f"[TOOL:transfer_call] Failed: {e}")
            return "Transfer failed. Please stay on the line."

    @llm.function_tool(description=(
        "Get available staff for this business. "
        "Call this FIRST when a caller wants to book an appointment, before asking anything else."
    ))
    async def get_staff(
        self,
        # FIX: was "_dummy" — Groq rejects tool schemas with underscore-prefixed
        # parameter names, silently dropping ALL tools from the API request.
        placeholder: Annotated[str, llm.TypeInfo(description="Pass empty string.")] = "",
    ) -> str:
        logger.info(f"[TOOL:get_staff] Called — businessId={self.business_id}")
        if not self.booking_enabled:
            return "Booking is not enabled for this business."
        data = await _call_backend("GET", "/calendar/staff", params={"businessId": self.business_id})
        if data.get("error"):
            return "Any available staff member."
        staff = data.get("staff", [])
        if not staff:
            return "Any available staff member."
        if len(staff) == 1:
            return f"Only one staff available: {staff[0]}."
        return "Staff: " + ", ".join(staff[:-1]) + " and " + staff[-1] + "."

    @llm.function_tool(description=(
        "Check available appointment slots for a given date. "
        "Always call this after the caller tells you their preferred date."
    ))
    async def check_availability(
        self,
        date: Annotated[str, llm.TypeInfo(description="Date in YYYY-MM-DD format.")],
        staff_name: Annotated[str, llm.TypeInfo(description="Staff name or 'Any'.")] = "Any",
    ) -> str:
        logger.info(f"[TOOL:check_availability] Called — date={date}, staff={staff_name}")
        if not self.booking_enabled:
            return "Online booking is not available."
        data = await _call_backend(
            "GET", "/calendar/slots",
            params={"businessId": self.business_id, "staffName": staff_name, "date": date},
        )
        if data.get("error"):
            return "Couldn't check availability right now."
        slots = data.get("slots", [])
        if not slots:
            tomorrow = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            return f"No slots on {date}. Want me to check {tomorrow}?"
        shown = slots[:4]
        more = f" and {len(slots) - 4} more" if len(slots) > 4 else ""
        return f"Available on {date}: {', '.join(shown)}{more}."

    @llm.function_tool(description=(
        "Book the appointment. "
        "Call this IMMEDIATELY once the caller confirms (says yes/okay/sure/right/yep/go ahead). "
        "Required: customer_name (MUST be provided by caller, NEVER guess), "
        "preferred_date (YYYY-MM-DD), preferred_time (from check_availability slots). "
        "Do NOT ask for reconfirmation again — one yes = call this now."
    ))
    async def book_appointment(
        self,
        customer_name: Annotated[str, llm.TypeInfo(description="Full name the caller provided.")],
        preferred_date: Annotated[str, llm.TypeInfo(description="Date in YYYY-MM-DD format.")],
        preferred_time: Annotated[str, llm.TypeInfo(description="Time like '3:00 PM' from available slots.")],
        staff_name: Annotated[str, llm.TypeInfo(description="Staff name or 'Any'.")] = "Any",
        customer_email: Annotated[str, llm.TypeInfo(description="Email if given, else empty.")] = "",
    ) -> str:
        logger.info(
            f"[TOOL:book_appointment] Called — businessId={self.business_id}, "
            f"customer='{customer_name}', date={preferred_date}, time={preferred_time}, staff={staff_name}"
        )
        name = str(customer_name).strip()
        if len(name) < 2 or name.lower() in {"", "unknown", "the customer", "caller", "guest", "user"}:
            return "I need your full name to complete the booking. What's your name?"
        if not self.booking_enabled:
            return "Online booking isn't available right now."
        result = await _call_backend(
            "POST", "/calendar/book",
            json={
                "businessId": self.business_id, "callLogId": self.call_log_id,
                "customerName": name, "customerPhone": self.caller_phone,
                "customerEmail": customer_email, "staffName": staff_name,
                "preferredDate": preferred_date, "preferredTime": preferred_time,
            },
            timeout=20.0,
        )
        if result.get("error"):
            logger.error(f"[TOOL:book_appointment] Backend error: {result}")
            return "Couldn't complete the booking. Want me to transfer you to the team?"
        if result.get("success"):
            logger.info(
                f"[TOOL:book_appointment] ✅ Booked — id={result.get('appointmentId')}, "
                f"date={result.get('date')}, time={result.get('time')}, staff={result.get('staffName')}"
            )
            return (
                f"Done! Booked — {result['date']} at {result['time']} "
                f"with {result['staffName']} at {result['businessName']}. "
                f"You'll get a WhatsApp confirmation shortly."
            )
        msg = result.get("message", "Couldn't book that slot. Want to try another time?")
        logger.warning(f"[TOOL:book_appointment] Rejected: {msg}")
        return msg


# =============================================================================
#  SYSTEM PROMPT BUILDER
# =============================================================================

def _build_system_prompt(base_prompt, business_name, booking_enabled, faqs):
    if "## YOUR ROLE" in base_prompt:
        base_prompt = base_prompt[:base_prompt.rfind("## YOUR ROLE")].rstrip()

    booking_section = ""
    if booking_enabled:
        booking_section = """

## BOOKING — 3 PHASES ONLY

### PHASE 1: COLLECT (ask one thing at a time)
- Say "I'll check what's available — do you have a preferred staff member or is anyone fine?"
- Get their date → call check_availability silently → tell them the slots.
- They pick a time → ask "And your name please?"

### PHASE 2: CONFIRM (one sentence)
- Say: "So [name], [date] at [time] — shall I confirm that?"

### PHASE 3: BOOK (immediate, no delay)
- ANY confirmation word (yes/okay/sure/right/go ahead/yep/done/book it) → call book_appointment IMMEDIATELY.
- Read out the result. Done.

### CRITICAL RULES FOR BOOKING:
- "Yes", "Okay", "Right", "Sure" after your confirmation question = CALL book_appointment NOW.
- Do NOT say "Let me book that" and wait. Call the tool and speak the result.
- Do NOT ask for confirmation twice. One confirm → one booking.
- If caller says a time that is not from check_availability slots, tell them it's not available and offer the nearest available slot instead.
- Never book without a name. Never guess a name.
"""

    faq_section = ""
    if faqs:
        lines = "\n".join(f"Q: {f.get('question','')}\nA: {f.get('answer','')}" for f in faqs)
        faq_section = f"\n\n## KNOWLEDGE BASE — answer these directly and confidently\n{lines}"

    rules = f"""

## CORE RULES
- Short natural sentences. Phone call, not an essay.
- Never say: "Great question!", "Absolutely!", "Of course!", "Certainly!"
- Never repeat what the caller just said.
- Never narrate your actions — don't say what you're about to do, just do it.
- Never mention tool or function names.
- Detect language in first sentence — match immediately and keep it throughout.
  Hindi → full Hindi with respectful "aap". English → natural English. Mixed → match their mix.
- Transfer only if caller explicitly asks for a human, OR stuck after 2–3 turns.
{booking_section}{faq_section}"""

    final = base_prompt + rules
    logger.info(f"[PROMPT] Built — {len(final)} chars (~{len(final)//4} tokens est.)")
    return final


# =============================================================================
#  TRANSCRIPT CAPTURE
#
#  FIX (agent transcripts): Replaced fragile private-attribute history polling
#  (_history, _agent.chat_ctx, etc.) with the stable public event
#  `conversation_item_added`. This fires every time the session appends any
#  message to conversation history. Filtering to role == "assistant" captures
#  agent turns reliably across all 1.4.x patch versions.
#
#  agent_speech_committed is kept as a fallback for greetings (generate_reply)
#  and any edge case where conversation_item_added doesn't fire. It only pushes
#  if seen count is still 0, preventing double-posting.
# =============================================================================

def _extract_text_from_any(obj) -> str:
    """Exhaustively extract plain text from any livekit-agents object."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    content = getattr(obj, "content", None)
    if isinstance(content, list):
        parts = [p for p in content if isinstance(p, str) and p.strip()]
        if parts:
            return " ".join(parts).strip()
    elif isinstance(content, str) and content.strip():
        return content.strip()
    tc = getattr(obj, "text_content", None)
    if tc and isinstance(tc, str) and tc.strip():
        return tc.strip()
    item = getattr(obj, "item", None)
    if item is not None and item is not obj:
        result = _extract_text_from_any(item)
        if result:
            return result
    for attr in ("transcript", "text", "message", "output", "speech", "utterance"):
        val = getattr(obj, attr, None)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


class TranscriptCapture:
    """
    Captures caller and agent speech using stable public session events.

    - user_input_transcribed   → caller turns (unchanged, always reliable)
    - conversation_item_added  → agent turns (replaces broken private polling)
    - agent_speech_committed   → fallback only for greetings / edge cases
    """

    def __init__(self, session: AgentSession, streamer: SegmentStreamer):
        self._session = session
        self._streamer = streamer
        self._seen_caller: dict = {}
        self._seen_agent: dict = {}

        @session.on("user_input_transcribed")
        def on_user(event):
            if not getattr(event, "is_final", True):
                return
            text = _extract_text_from_any(event)
            if text:
                logger.info(f"[EVENT] user_input_transcribed: '{text}'")
                self._push("caller", text, self._seen_caller)

        @session.on("conversation_item_added")
        def on_item_added(event):
            """
            Fires whenever a new item is appended to the conversation history.
            Filter to assistant role only to capture agent turns.
            """
            item = getattr(event, "item", None) or event
            role = getattr(item, "role", None)
            role_str = (role.value if hasattr(role, "value") else str(role or "")).lower()
            if "assistant" not in role_str:
                return
            text = _extract_text_from_any(item)
            if text:
                logger.info(f"[EVENT] conversation_item_added (agent): '{text[:80]}'")
                self._push("agent", text, self._seen_agent)

        @session.on("agent_speech_committed")
        def on_agent_committed(event):
            """
            Fallback only. Pushes only if conversation_item_added hasn't already
            captured this text (seen count == 0), avoiding double-posting.
            """
            text = _extract_text_from_any(event)
            if not text:
                return
            if self._seen_agent.get(text, 0) == 0:
                logger.info(f"[EVENT] agent_speech_committed (fallback): '{text[:80]}'")
                self._push("agent", text, self._seen_agent)

    def _push(self, speaker, text, seen):
        text = text.strip()
        if not text:
            return
        count = seen.get(text, 0)
        if count >= 3:
            return
        seen[text] = count + 1
        asyncio.create_task(self._streamer.push(speaker, text))


# =============================================================================
#  MAIN ENTRYPOINT
# =============================================================================

async def entrypoint(ctx: JobContext):
    logger.info(f"[ENTRYPOINT] New job — room: {ctx.room.name}")
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    logger.info(f"[ENTRYPOINT] Participant — kind: {participant.kind}, identity: {participant.identity}")

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

    system_prompt = _build_system_prompt(system_prompt, business_name, booking_enabled, faqs)

    agent = ConvoDeskAgent(
        ctx=ctx,
        caller_phone=caller_number,
        transfer_number=transfer_num,
        call_log_id=call_log_id,
        business_id=business_id,
        booking_enabled=booking_enabled,
        instructions=system_prompt,
    )
    logger.info(f"[TOOLS] ConvoDeskAgent created with 4 tools: "
                f"transfer_call, get_staff, check_availability, book_appointment")

    streamer = SegmentStreamer(call_log_id) if call_log_id else None

    # AgentSession.__init__() does NOT accept agent= in livekit-agents 1.4.x.
    # agent= is passed to start() — that is the correct and only API.
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=_build_stt(stt_language),
        llm=_build_llm(llm_provider, llm_model),
        tts=_build_tts(tts_provider, tts_voice, stt_language),
    )

    # Attach transcript capture BEFORE session.start()
    if streamer:
        TranscriptCapture(session, streamer)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    await session.generate_reply(instructions=greeting)

    try:
        await session.wait_for_close()
    except AttributeError:
        disconnected = asyncio.Event()
        ctx.room.on("disconnected", lambda *_: disconnected.set())

        @ctx.room.on("participant_disconnected")
        def on_participant_left(p):
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                logger.info(f"[ENTRYPOINT] SIP participant disconnected: {p.identity}")
                disconnected.set()

        await disconnected.wait()

    logger.info("[ENTRYPOINT] Session ended — starting cleanup")

    if streamer:
        await streamer.close()

    if call_log_id and streamer:
        transcript_text = streamer.get_full_transcript()
        logger.info(f"[CLEANUP] {len(transcript_text)} chars, {len(streamer._all)} segments")
        if transcript_text.strip():
            try:
                summary = await generate_summary(transcript_text)
                await save_summary(call_log_id, summary, transcript_text)
            except Exception as e:
                logger.error(f"[CLEANUP] Summary failed: {e}")
        else:
            logger.warning("[CLEANUP] Empty transcript — skipping summary")

    if call_log_id:
        try:
            await end_call(call_log_id)
        except Exception as e:
            logger.error(f"[CLEANUP] End call failed: {e}")

    logger.info(f"[ENTRYPOINT] ✅ Done — room: {ctx.room.name}, callLogId: {call_log_id}")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="convodesk-agent",
        )
    )