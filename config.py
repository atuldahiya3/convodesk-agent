import os
from dotenv import load_dotenv

load_dotenv()

# =========================================================================================
#  CONVODESK AI - AGENT CONFIGURATION
#  Static fallback defaults — used when Spring Boot is unreachable or during local dev.
#  Per-tenant values (prompt, voice, FAQs) override these at runtime via Spring Boot.
# =========================================================================================

# --- 1. FALLBACK PERSONA ---
SYSTEM_PROMPT = """
You are a helpful and polite AI receptionist.
Answer questions concisely and professionally.
If you don't know something, offer to transfer the caller to a human.
"""

INITIAL_GREETING = "The user has picked up the call. Greet them warmly and ask how you can help."

# --- 2. STT SETTINGS ---
STT_PROVIDER = "deepgram"
STT_MODEL    = "nova-2"
STT_LANGUAGE = "en"

# # --- 3. TTS SETTINGS ---
# DEFAULT_TTS_PROVIDER = "openai"
# DEFAULT_TTS_VOICE    = "alloy"

SARVAM_MODEL    = "bulbul:v2"
SARVAM_LANGUAGE = "en-IN"

CARTESIA_MODEL = "sonic-2"
CARTESIA_VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"

# --- 4. LLM SETTINGS ---
DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_LLM_MODEL    = "llama-3.3-70b-versatile"

DEFAULT_TTS_PROVIDER = "sarvam"
DEFAULT_TTS_VOICE    = "anushka"

GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.7

# --- 5. TELEPHONY ---
DEFAULT_TRANSFER_NUMBER = os.getenv("DEFAULT_TRANSFER_NUMBER", "")
SIP_TRUNK_ID            = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")
YOUR_VOBIZ_NUMBER       = os.getenv("YOUR_VOBIZ_NUMBER", "")
SIP_DOMAIN              = os.getenv("VOBIZ_SIP_DOMAIN", "")

# --- 6. SPRING BOOT BACKEND ---
BACKEND_URL  = os.getenv("BACKEND_URL", "http://localhost:8080")
AGENT_SECRET = os.getenv("AGENT_API_SECRET", "")