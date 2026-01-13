"""
Gemini_proxy_claudecode - Google Gemini to Anthropic API Proxy (OAuth)

Translates Anthropic Messages API requests to Google Gemini API format,
using Google OAuth for authentication (no API key required).
"""

import os
import sys
import json
import logging
import webbrowser
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini-proxy-claudecode")

app = FastAPI(
    title="Gemini_proxy_claudecode",
    description="Google Gemini to Anthropic API Proxy (OAuth)",
    version="1.0.0"
)

# Token storage directory
TOKEN_DIR = Path.home() / ".gemini-proxy"
TOKEN_FILE = TOKEN_DIR / "google-oauth.json"

# API Key mode (simpler, recommended for personal use)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Google OAuth Configuration (for OAuth mode)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"

# Gemini API endpoint (using OAuth)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Scopes needed for Gemini API
SCOPES = [
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/generative-language.tuning",
    "https://www.googleapis.com/auth/cloud-platform",
]

# Supported models - maps display names to actual Gemini API model names
MODEL_MAPPING = {
    # ===== Gemini 3 (Latest - Dec 2025) =====
    "Gemini-3-Flash": "gemini-3-flash-preview",
    "Gemini-3-Pro": "gemini-3-pro-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",

    # ===== Gemini 2.5 =====
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "Gemini-2.5-Pro": "gemini-2.5-pro",
    "Gemini-2.5-Flash-Lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",

    # ===== Gemini 2.0 =====
    "Gemini-2-Flash": "gemini-2.0-flash",
    "Gemini-2-Flash-Lite": "gemini-2.0-flash-lite",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",

    # ===== Gemini 1.5 (Legacy) =====
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-pro-latest": "gemini-1.5-pro-latest",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
    "gemini-pro": "gemini-1.5-pro",
}


class GoogleOAuthManager:
    """Manages Google OAuth tokens for Gemini API access."""

    def __init__(self):
        self.token_data: Optional[dict] = None
        self.load_tokens()

    def load_tokens(self) -> bool:
        """Load tokens from disk."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, "r") as f:
                    self.token_data = json.load(f)
                logger.info(f"Loaded tokens for {self.token_data.get('email', 'unknown')}")
                return True
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")
        return False

    def save_tokens(self):
        """Save tokens to disk."""
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            json.dump(self.token_data, f, indent=2)
        # Secure the file
        os.chmod(TOKEN_FILE, 0o600)
        logger.info("Tokens saved")

    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        if not self.token_data:
            return False
        if not self.token_data.get("access_token"):
            return False
        return True

    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.token_data:
            return True
        expires_at = self.token_data.get("expires_at")
        if not expires_at:
            return True
        return datetime.fromisoformat(expires_at) < datetime.now()

    async def refresh_token(self) -> bool:
        """Refresh the access token using refresh token."""
        if not self.token_data or not self.token_data.get("refresh_token"):
            logger.error("No refresh token available")
            return False

        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "refresh_token": self.token_data["refresh_token"],
                    "grant_type": "refresh_token",
                }
            )

            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.text}")
                return False

            data = response.json()
            self.token_data["access_token"] = data["access_token"]
            self.token_data["expires_at"] = (
                datetime.now() + timedelta(seconds=data.get("expires_in", 3600))
            ).isoformat()
            self.token_data["last_refresh"] = datetime.now().isoformat()
            self.save_tokens()
            logger.info("Token refreshed successfully")
            return True

    async def get_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        if not self.is_authenticated():
            return None

        if self.is_expired():
            success = await self.refresh_token()
            if not success:
                return None

        return self.token_data.get("access_token")

    async def device_flow_login(self):
        """Perform OAuth login using device flow."""
        print("\n" + "=" * 50)
        print("  Google OAuth Login for Gemini API")
        print("=" * 50)

        async with httpx.AsyncClient() as client:
            # Step 1: Request device code
            response = await client.post(
                GOOGLE_DEVICE_CODE_URL,
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "scope": " ".join(SCOPES),
                }
            )

            if response.status_code != 200:
                print(f"❌ Failed to get device code: {response.text}")
                return False

            device_data = response.json()
            device_code = device_data["device_code"]
            user_code = device_data["user_code"]
            verification_url = device_data["verification_url"]
            expires_in = device_data.get("expires_in", 1800)
            interval = device_data.get("interval", 5)

            print(f"\n📱 Please visit: {verification_url}")
            print(f"🔑 Enter code: {user_code}\n")

            # Try to open browser automatically
            try:
                webbrowser.open(verification_url)
                print("(Browser opened automatically)")
            except:
                pass

            print("Waiting for authorization...")

            # Step 2: Poll for token
            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < expires_in:
                await asyncio.sleep(interval)

                token_response = await client.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "client_id": GOOGLE_CLIENT_ID,
                        "client_secret": GOOGLE_CLIENT_SECRET,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    }
                )

                if token_response.status_code == 200:
                    token_data = token_response.json()

                    # Get user info
                    userinfo_response = await client.get(
                        "https://www.googleapis.com/oauth2/v2/userinfo",
                        headers={"Authorization": f"Bearer {token_data['access_token']}"}
                    )
                    userinfo = userinfo_response.json() if userinfo_response.status_code == 200 else {}

                    self.token_data = {
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data.get("refresh_token"),
                        "expires_at": (
                            datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600))
                        ).isoformat(),
                        "email": userinfo.get("email", "unknown"),
                        "name": userinfo.get("name", ""),
                        "created_at": datetime.now().isoformat(),
                        "last_refresh": datetime.now().isoformat(),
                    }
                    self.save_tokens()

                    print(f"\n✅ Logged in as: {self.token_data['email']}")
                    print("=" * 50 + "\n")
                    return True

                error = token_response.json().get("error")
                if error == "authorization_pending":
                    print(".", end="", flush=True)
                elif error == "slow_down":
                    interval += 2
                elif error in ["access_denied", "expired_token"]:
                    print(f"\n❌ Authorization failed: {error}")
                    return False

            print("\n❌ Authorization timed out")
            return False


# Global OAuth manager
oauth_manager = GoogleOAuthManager()


def get_gemini_model_name(model: str) -> str:
    """Map model name to Gemini model name."""
    return MODEL_MAPPING.get(model, model)


def convert_anthropic_to_gemini(anthropic_request: dict) -> dict:
    """Convert Anthropic Messages API format to Gemini REST API format."""

    contents = []
    system_instruction = None

    # Extract system instruction
    if "system" in anthropic_request:
        system_content = anthropic_request["system"]
        if isinstance(system_content, list):
            system_instruction = {"parts": [{"text": " ".join(
                block.get("text", "") for block in system_content
                if block.get("type") == "text"
            )}]}
        else:
            system_instruction = {"parts": [{"text": system_content}]}

    # Convert messages
    for msg in anthropic_request.get("messages", []):
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        parts.append({
                            "inlineData": {
                                "mimeType": source.get("media_type", "image/png"),
                                "data": source.get("data", "")
                            }
                        })

        if parts:
            contents.append({"role": role, "parts": parts})

    # Build request
    gemini_request = {"contents": contents}

    if system_instruction:
        gemini_request["systemInstruction"] = system_instruction

    # Generation config
    gen_config = {}
    if "max_tokens" in anthropic_request:
        gen_config["maxOutputTokens"] = anthropic_request["max_tokens"]
    if "temperature" in anthropic_request:
        gen_config["temperature"] = anthropic_request["temperature"]
    if "top_p" in anthropic_request:
        gen_config["topP"] = anthropic_request["top_p"]
    if "top_k" in anthropic_request:
        gen_config["topK"] = anthropic_request["top_k"]

    if gen_config:
        gemini_request["generationConfig"] = gen_config

    return gemini_request


def convert_gemini_to_anthropic(gemini_response: dict, model: str) -> dict:
    """Convert Gemini REST API response to Anthropic Messages format."""

    content = []
    stop_reason = "end_turn"

    try:
        candidates = gemini_response.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])

            for part in parts:
                if "text" in part:
                    content.append({"type": "text", "text": part["text"]})

            finish_reason = candidate.get("finishReason", "STOP")
            if finish_reason == "MAX_TOKENS":
                stop_reason = "max_tokens"
            elif finish_reason == "SAFETY":
                stop_reason = "end_turn"

    except Exception as e:
        logger.error(f"Error converting response: {e}")
        content.append({"type": "text", "text": str(gemini_response)})

    # Token usage
    usage_metadata = gemini_response.get("usageMetadata", {})

    return {
        "id": f"msg_gemini_{hash(str(gemini_response)) % 10000000}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_metadata.get("promptTokenCount", 0),
            "output_tokens": usage_metadata.get("candidatesTokenCount", 0)
        }
    }


async def stream_gemini_response(response: httpx.Response, model: str) -> AsyncGenerator[str, None]:
    """Stream Gemini responses converted to Anthropic SSE format."""

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': 'msg_gemini_stream', 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    full_content = ""

    async for line in response.aiter_lines():
        if not line.strip():
            continue

        # Handle SSE data lines
        if line.startswith("data: "):
            data = line[6:]
            try:
                chunk = json.loads(data)
                candidates = chunk.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if "text" in part:
                            text = part["text"]
                            full_content += text
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
            except json.JSONDecodeError:
                continue

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(full_content.split())}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@app.on_event("startup")
async def startup_event():
    """Check authentication on startup."""
    if GOOGLE_API_KEY:
        print(f"\n✅ Using API Key mode (GOOGLE_API_KEY set)")
        print(f"   Key: {GOOGLE_API_KEY[:10]}...{GOOGLE_API_KEY[-4:]}")
    elif GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        if not oauth_manager.is_authenticated():
            print("\n⚠️  Not logged in to Google. Starting OAuth flow...")
            await oauth_manager.device_flow_login()
        elif oauth_manager.is_expired():
            print("\n🔄 Token expired, refreshing...")
            success = await oauth_manager.refresh_token()
            if not success:
                print("⚠️  Refresh failed. Starting OAuth flow...")
                await oauth_manager.device_flow_login()
        else:
            print(f"\n✅ Logged in as: {oauth_manager.token_data.get('email', 'unknown')}")
    else:
        print("\n⚠️  No authentication configured!")
        print("   Set GOOGLE_API_KEY for API key mode")
        print("   Or set GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET for OAuth")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if GOOGLE_API_KEY:
        return {
            "status": "healthy",
            "service": "gemini-proxy-claudecode",
            "auth_mode": "api_key",
            "authenticated": True
        }
    return {
        "status": "healthy",
        "service": "gemini-proxy-claudecode",
        "auth_mode": "oauth",
        "authenticated": oauth_manager.is_authenticated(),
        "email": oauth_manager.token_data.get("email") if oauth_manager.token_data else None
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{"id": model, "object": "model"} for model in MODEL_MAPPING.keys()]
    }


@app.post("/login")
async def login():
    """Trigger OAuth login flow."""
    success = await oauth_manager.device_flow_login()
    if success:
        return {"status": "success", "email": oauth_manager.token_data.get("email")}
    raise HTTPException(status_code=401, detail="Login failed")


@app.post("/logout")
async def logout():
    """Clear stored tokens."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
    oauth_manager.token_data = None
    return {"status": "logged_out"}


@app.post("/v1/messages")
async def messages(request: Request):
    """Main messages endpoint - translates Anthropic Messages API to Gemini."""

    # Check authentication - API key or OAuth
    if GOOGLE_API_KEY:
        auth_mode = "api_key"
        auth_value = GOOGLE_API_KEY
    else:
        access_token = await oauth_manager.get_access_token()
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated. Set GOOGLE_API_KEY or configure OAuth."
            )
        auth_mode = "oauth"
        auth_value = access_token

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    model = body.get("model", "gemini-1.5-flash")
    gemini_model = get_gemini_model_name(model)
    logger.info(f"Request for model: {model} -> {gemini_model}")

    # Convert request
    gemini_request = convert_anthropic_to_gemini(body)
    logger.debug(f"Gemini request: {json.dumps(gemini_request, indent=2)}")

    # Build URL
    stream = body.get("stream", False)
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{GEMINI_API_BASE}/models/{gemini_model}:{action}"

    # Add auth to URL or headers based on mode
    if auth_mode == "api_key":
        url += f"?key={auth_value}"
        if stream:
            url += "&alt=sse"
        headers = {"Content-Type": "application/json"}
    else:
        if stream:
            url += "?alt=sse"
        headers = {
            "Authorization": f"Bearer {auth_value}",
            "Content-Type": "application/json",
        }

    async with httpx.AsyncClient(timeout=300.0) as client:
        if stream:
            async with client.stream("POST", url, json=gemini_request, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    logger.error(f"Gemini API error: {error_text}")
                    raise HTTPException(status_code=response.status_code, detail=str(error_text))

                return StreamingResponse(
                    stream_gemini_response(response, model),
                    media_type="text/event-stream"
                )
        else:
            response = await client.post(url, json=gemini_request, headers=headers)

            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)

            gemini_response = response.json()
            anthropic_response = convert_gemini_to_anthropic(gemini_response, model)
            return JSONResponse(content=anthropic_response)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "server_error", "message": str(exc)}}
    )


def run_login():
    """CLI command to trigger login."""
    async def do_login():
        await oauth_manager.device_flow_login()
    asyncio.run(do_login())


if __name__ == "__main__":
    import uvicorn

    # Check for login command
    if len(sys.argv) > 1 and sys.argv[1] == "--login":
        run_login()
    else:
        port = int(os.getenv("PORT", 8081))
        uvicorn.run(app, host="0.0.0.0", port=port)
