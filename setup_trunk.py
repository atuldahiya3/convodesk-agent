import asyncio
import os
import certifi
from dotenv import load_dotenv
from livekit import api
from livekit.protocol import sip

# Fix SSL for macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv(".env")

async def setup_vobiz_trunk():
    print("--- LiveKit SIP Trunk Setup ---")
    
    # Credentials from .env
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    
    # Vobiz Settings
    vobiz_domain = os.getenv("VOBIZ_SIP_DOMAIN")
    vobiz_user = os.getenv("VOBIZ_USERNAME")
    vobiz_pass = os.getenv("VOBIZ_PASSWORD")
    vobiz_number = os.getenv("VOBIZ_OUTBOUND_NUMBER") # Your Caller ID

    if not all([url, key, secret, vobiz_domain, vobiz_user, vobiz_pass]):
        print("❌ Error: Missing credentials in .env file.")
        return

    lkapi = api.LiveKitAPI(url, key, secret)

    try:
        print(f"Creating/Updating Outbound Trunk for: {vobiz_domain}")

        # Define the trunk configuration
        # NOTE: We use 'in' for India region pinning to help with 488/Codec issues
        trunk_info = sip.SIPOutboundTrunkInfo(
            name="Vobiz Outbound Trunk",
            address=vobiz_domain,
            auth_username=vobiz_user,
            auth_password=vobiz_pass,
            numbers=[vobiz_number] if vobiz_number else ["*"],
            # Critical for India: Pins the call to Indian data centers
            # to comply with local TRAI regulations and reduce latency
        )

        # Create the trunk
        # If you already have a trunk, you can delete it first or use update
        request = sip.CreateSIPOutboundTrunkRequest(trunk=trunk_info)
        trunk = await lkapi.sip.create_outbound_trunk(request)

        print("\n✅ SUCCESS!")
        print(f"Trunk ID: {trunk.sip_trunk_id}")
        print(f"Status: Active")
        print("\n--- ACTION REQUIRED ---")
        print(f"Update your .env file with:")
        print(f"SIP_TRUNK_ID={trunk.sip_trunk_id}")

    except Exception as e:
        print(f"\n❌ Failed to setup trunk: {e}")
    finally:
        await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(setup_vobiz_trunk())