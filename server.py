"""
Gemini_proxy_claudecode - Google Gemini to Anthropic API Proxy

Translates Anthropic Messages API requests to Google Gemini API format.
Uses Google Application Default Credentials (ADC) - just run:
  gcloud auth application-default login

Or set GOOGLE_API_KEY environment variable for API key auth.
"""

import os
import sys
import json
import logging
import subprocess
import secrets
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
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini-proxy")

app = FastAPI(
    title="Gemini_proxy_claudecode",
    description="Google Gemini to Anthropic API Proxy",
    version="1.0.0"
)

# Gemini API
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ADC credentials file location
ADC_FILE = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

# Model mapping
MODEL_MAPPING = {
    "Gemini-3-Flash": "gemini-3-flash-preview",
    "Gemini-3-Pro": "gemini-3-pro-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "Gemini-2.5-Pro": "gemini-2.5-pro",
    "Gemini-2.5-Flash-Lite": "gemini-2.5-flash-lite",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "Gemini-2-Flash": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
}


class GeminiAuth:
    """Manages Google authentication for Gemini API using ADC or API key."""

    def __init__(self):
        self.access_token = None
        self.token_expiry = None
        self.email = None
        self.auth_method = None  # 'adc', 'api_key', or None
        self._check_auth()

    def _check_auth(self):
        """Check available authentication methods."""
        # Priority 1: API key from environment
        if os.getenv("GOOGLE_API_KEY"):
            self.auth_method = "api_key"
            logger.info("Using GOOGLE_API_KEY for authentication")
            return

        # Priority 2: Application Default Credentials
        if ADC_FILE.exists():
            try:
                creds = json.loads(ADC_FILE.read_text())
                if creds.get("client_id") and creds.get("refresh_token"):
                    self.auth_method = "adc"
                    logger.info("Using Application Default Credentials")
                    return
            except Exception as e:
                logger.warning(f"Failed to read ADC file: {e}")

        self.auth_method = None

    @property
    def is_configured(self) -> bool:
        return self.auth_method is not None

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv("GOOGLE_API_KEY")

    def login(self) -> bool:
        """Trigger gcloud auth application-default login."""
        print("\n" + "=" * 60)
        print("  GEMINI PROXY - GOOGLE LOGIN")
        print("=" * 60)
        print("\nRunning: gcloud auth application-default login")
        print("This will open your browser for Google login.\n")

        try:
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "login",
                 "--scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever"],
                check=True
            )
            if result.returncode == 0:
                self._check_auth()
                if self.auth_method == "adc":
                    print("\n" + "=" * 60)
                    print("  LOGIN SUCCESSFUL!")
                    print("=" * 60 + "\n")
                    return True
        except FileNotFoundError:
            print("\n" + "-" * 60)
            print("  gcloud CLI not found!")
            print("-" * 60)
            print("\nInstall options:")
            print("  brew install google-cloud-sdk")
            print("  # or visit: https://cloud.google.com/sdk/docs/install")
            print("\nAlternatively, set GOOGLE_API_KEY environment variable")
            print("  Get key from: https://aistudio.google.com/apikey")
        except subprocess.CalledProcessError as e:
            print(f"\n  Login failed: {e}")

        return False

    async def get_token(self) -> Optional[str]:
        """Get valid access token from ADC."""
        if self.auth_method == "api_key":
            return None  # API key auth doesn't use bearer tokens

        if self.auth_method != "adc":
            return None

        # Check if we have a valid cached token
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry:
                return self.access_token

        # Refresh token using ADC
        try:
            creds = json.loads(ADC_FILE.read_text())
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "client_id": creds["client_id"],
                        "client_secret": creds["client_secret"],
                        "refresh_token": creds["refresh_token"],
                        "grant_type": "refresh_token",
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.access_token = data["access_token"]
                    self.token_expiry = datetime.now() + timedelta(seconds=data.get("expires_in", 3600) - 60)
                    return self.access_token
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")

        return None

    def get_email(self) -> str:
        """Get logged in email from ADC."""
        if self.auth_method == "api_key":
            return "(API Key)"
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "account"],
                capture_output=True, text=True
            )
            return result.stdout.strip() or "unknown"
        except:
            return "unknown"


# Global auth manager
auth = GeminiAuth()


def get_model(name: str) -> str:
    return MODEL_MAPPING.get(name, name)


def to_gemini(req: dict) -> dict:
    """Convert Anthropic request to Gemini format."""
    contents = []
    system = None

    if "system" in req:
        s = req["system"]
        if isinstance(s, list):
            s = " ".join(b.get("text", "") for b in s if b.get("type") == "text")
        system = {"parts": [{"text": s}]}

    for msg in req.get("messages", []):
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        parts = []

        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for b in content:
                if b.get("type") == "text":
                    parts.append({"text": b.get("text", "")})
                elif b.get("type") == "image":
                    src = b.get("source", {})
                    if src.get("type") == "base64":
                        parts.append({"inlineData": {"mimeType": src.get("media_type", "image/png"), "data": src.get("data", "")}})
                elif b.get("type") == "tool_use":
                    parts.append({"functionCall": {"name": b.get("name", ""), "args": b.get("input", {})}})
                elif b.get("type") == "tool_result":
                    tc = b.get("content", "")
                    if isinstance(tc, list):
                        tc = " ".join(x.get("text", "") for x in tc if x.get("type") == "text")
                    parts.append({"functionResponse": {"name": b.get("tool_use_id", "tool"), "response": {"result": tc}}})

        if parts:
            contents.append({"role": role, "parts": parts})

    result = {"contents": contents}
    if system:
        result["systemInstruction"] = system

    cfg = {}
    if "max_tokens" in req: cfg["maxOutputTokens"] = req["max_tokens"]
    if "temperature" in req: cfg["temperature"] = req["temperature"]
    if "top_p" in req: cfg["topP"] = req["top_p"]
    if cfg:
        result["generationConfig"] = cfg

    if "tools" in req:
        funcs = [{"name": t.get("name", ""), "description": t.get("description", ""), "parameters": t.get("input_schema", {})} for t in req["tools"]]
        if funcs:
            result["tools"] = [{"functionDeclarations": funcs}]

    return result


def from_gemini(resp: dict, model: str) -> dict:
    """Convert Gemini response to Anthropic format."""
    content = []
    stop = "end_turn"

    try:
        cands = resp.get("candidates", [])
        if cands:
            for part in cands[0].get("content", {}).get("parts", []):
                if "text" in part:
                    content.append({"type": "text", "text": part["text"]})
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    content.append({"type": "tool_use", "id": f"toolu_{secrets.token_hex(4)}", "name": fc.get("name", ""), "input": fc.get("args", {})})
                    stop = "tool_use"

            fr = cands[0].get("finishReason", "STOP")
            if fr == "MAX_TOKENS": stop = "max_tokens"
    except: pass

    usage = resp.get("usageMetadata", {})
    return {
        "id": f"msg_{secrets.token_hex(8)}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("promptTokenCount", 0), "output_tokens": usage.get("candidatesTokenCount", 0)}
    }


async def stream_response(resp: httpx.Response, model: str) -> AsyncGenerator[str, None]:
    """Stream Gemini response as Anthropic SSE."""
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{secrets.token_hex(8)}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    full = ""
    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            try:
                chunk = json.loads(line[6:])
                for part in chunk.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    if "text" in part:
                        t = part["text"]
                        full += t
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': t}})}\n\n"
            except: pass

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(full.split())}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@app.on_event("startup")
async def startup():
    """Check auth on startup and prompt for API key if needed."""
    print("\n" + "=" * 50)
    print("  GEMINI PROXY FOR CLAUDE CODE")
    print("=" * 50)

    if not auth.is_configured:
        print("\n⚠️  No API key found!")
        print("\n" + "-" * 50)
        print("  GET YOUR FREE API KEY:")
        print("-" * 50)
        print("\n  1. Go to: https://aistudio.google.com/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Copy the key")
        print("\n  Then start proxy with:")
        print("  GOOGLE_API_KEY=your-key start-gemini-proxy")
        print("\n  Or add to ~/.zshrc:")
        print("  export GOOGLE_API_KEY='your-key'")
        print("-" * 50 + "\n")

        # Try to open AI Studio
        import webbrowser
        print("Opening Google AI Studio...")
        webbrowser.open("https://aistudio.google.com/apikey")
        sys.exit(1)

    print(f"\n✅ Ready!")
    print(f"   Auth: {auth.auth_method.upper()}")
    if auth.auth_method == "api_key":
        print(f"   Key: {auth.api_key[:8]}...{auth.api_key[-4:]}")
    else:
        print(f"   Account: {auth.get_email()}")
    print("=" * 50 + "\n")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "gemini-proxy",
        "auth_method": auth.auth_method,
        "configured": auth.is_configured,
        "email": auth.get_email() if auth.is_configured else None
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in MODEL_MAPPING]}


@app.post("/v1/messages")
async def messages(request: Request):
    """Main endpoint - translates Anthropic to Gemini."""
    if not auth.is_configured:
        raise HTTPException(401, "Not authenticated. Restart proxy to login.")

    try:
        body = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON")

    model = body.get("model", "gemini-2.0-flash")
    gemini_model = get_model(model)
    logger.info(f"Request: {model} → {gemini_model}")

    gemini_req = to_gemini(body)
    stream = body.get("stream", False)
    action = "streamGenerateContent" if stream else "generateContent"

    # Build URL based on auth method
    if auth.auth_method == "api_key":
        url = f"{GEMINI_API_BASE}/models/{gemini_model}:{action}?key={auth.api_key}"
        if stream:
            url += "&alt=sse"
        headers = {"Content-Type": "application/json"}
    else:
        token = await auth.get_token()
        if not token:
            raise HTTPException(401, "Token refresh failed. Run: gemini-login")
        url = f"{GEMINI_API_BASE}/models/{gemini_model}:{action}"
        if stream:
            url += "?alt=sse"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=300) as client:
        if stream:
            async with client.stream("POST", url, json=gemini_req, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = (await resp.aread()).decode()
                    logger.error(f"Gemini error: {error_text}")
                    raise HTTPException(resp.status_code, error_text)
                return StreamingResponse(stream_response(resp, model), media_type="text/event-stream")
        else:
            resp = await client.post(url, json=gemini_req, headers=headers)
            if resp.status_code != 200:
                logger.error(f"Gemini error: {resp.text}")
                raise HTTPException(resp.status_code, resp.text)
            return JSONResponse(from_gemini(resp.json(), model))


@app.post("/login")
async def trigger_login():
    """Trigger login flow."""
    if auth.login():
        return {"status": "ok", "email": auth.get_email()}
    raise HTTPException(401, "Login failed")


@app.post("/logout")
async def logout():
    """Clear cached tokens (ADC file remains)."""
    auth.access_token = None
    auth.token_expiry = None
    return {"status": "logged out"}


if __name__ == "__main__":
    import uvicorn

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--login":
            auth.login()
        elif cmd == "--status":
            print("\n" + "=" * 40)
            print("  GEMINI PROXY STATUS")
            print("=" * 40)
            print(f"  Configured: {'Yes' if auth.is_configured else 'No'}")
            print(f"  Auth method: {auth.auth_method or 'None'}")
            print(f"  Account: {auth.get_email()}")
            print("=" * 40 + "\n")
        else:
            print("Commands: --login, --status")
    else:
        # Start server - will auto-prompt for login if needed
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8081)))
