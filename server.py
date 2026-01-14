"""
Gemini_proxy_claudecode - Google Gemini to Anthropic API Proxy (OAuth)

Translates Anthropic Messages API requests to Google Gemini API format,
using Google OAuth for authentication - just like CLIProxyAPI for ChatGPT.
"""

import os
import sys
import json
import logging
import webbrowser
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, parse_qs, urlparse
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
    description="Google Gemini to Anthropic API Proxy (OAuth)",
    version="1.0.0"
)

# Storage
TOKEN_DIR = Path.home() / ".gemini-proxy"
TOKEN_FILE = TOKEN_DIR / "google-oauth.json"
CONFIG_FILE = TOKEN_DIR / "config.json"

# OAuth Configuration
OAUTH_CALLBACK_PORT = 8089
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Gemini API
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Scopes for Gemini
SCOPES = [
    "https://www.googleapis.com/auth/generative-language.retriever",
    "https://www.googleapis.com/auth/cloud-platform",
    "openid", "email", "profile",
]

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


class OAuthCallback(BaseHTTPRequestHandler):
    """Handles OAuth callback."""
    auth_code = None
    error = None

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if "code" in params:
            OAuthCallback.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html><body style="font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;background:linear-gradient(135deg,#4285f4,#34a853);">
                <div style="background:white;padding:40px;border-radius:16px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,0.2);">
                <h1 style="color:#22c55e;">Login Successful!</h1>
                <p>You can close this window.</p>
                </div></body></html>
            """)
        elif "error" in params:
            OAuthCallback.error = params.get("error_description", ["Login failed"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h1>Error: {OAuthCallback.error}</h1></body></html>".encode())

    def log_message(self, *args): pass


class GeminiAuth:
    """Manages Google OAuth for Gemini API."""

    def __init__(self):
        self.config = {}
        self.tokens = None
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if CONFIG_FILE.exists():
            self.config = json.loads(CONFIG_FILE.read_text())
        if TOKEN_FILE.exists():
            self.tokens = json.loads(TOKEN_FILE.read_text())
        # Env vars override
        if os.getenv("GOOGLE_CLIENT_ID"):
            self.config["client_id"] = os.getenv("GOOGLE_CLIENT_ID")
        if os.getenv("GOOGLE_CLIENT_SECRET"):
            self.config["client_secret"] = os.getenv("GOOGLE_CLIENT_SECRET")

    def _save_config(self):
        CONFIG_FILE.write_text(json.dumps(self.config, indent=2))
        os.chmod(CONFIG_FILE, 0o600)

    def _save_tokens(self):
        TOKEN_FILE.write_text(json.dumps(self.tokens, indent=2))
        os.chmod(TOKEN_FILE, 0o600)

    @property
    def is_configured(self) -> bool:
        return bool(self.config.get("client_id") and self.config.get("client_secret"))

    @property
    def is_logged_in(self) -> bool:
        return bool(self.tokens and self.tokens.get("access_token"))

    @property
    def is_expired(self) -> bool:
        if not self.tokens: return True
        exp = self.tokens.get("expires_at")
        if not exp: return True
        return datetime.fromisoformat(exp) < datetime.now()

    def setup(self) -> bool:
        """One-time OAuth client setup."""
        print("\n" + "=" * 60)
        print("  GEMINI PROXY - ONE-TIME SETUP")
        print("=" * 60)
        print("\nYou need to create Google OAuth credentials (one time only):")
        print("\n1. Go to: https://console.cloud.google.com/apis/credentials")
        print("2. Create a project (or select existing)")
        print("3. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'")
        print("4. Configure consent screen if prompted (External, just your email)")
        print("5. Select 'Desktop app' as application type")
        print("6. Copy the Client ID and Client Secret")
        print("\n" + "-" * 60)

        # Try to open the URL
        print("\nOpening Google Cloud Console...")
        webbrowser.open("https://console.cloud.google.com/apis/credentials")

        client_id = input("\nPaste Client ID: ").strip()
        client_secret = input("Paste Client Secret: ").strip()

        if client_id and client_secret:
            self.config["client_id"] = client_id
            self.config["client_secret"] = client_secret
            self._save_config()
            print("\n✅ Setup complete! Now logging in...")
            return True
        print("\n❌ Setup cancelled")
        return False

    def login(self) -> bool:
        """OAuth login via browser."""
        if not self.is_configured:
            if not self.setup():
                return False

        print("\n" + "=" * 50)
        print("  GOOGLE LOGIN")
        print("=" * 50)

        # Build auth URL
        state = secrets.token_urlsafe(16)
        auth_url = f"{GOOGLE_AUTH_URL}?" + urlencode({
            "client_id": self.config["client_id"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        })

        # Reset callback state
        OAuthCallback.auth_code = None
        OAuthCallback.error = None

        # Start callback server
        server = HTTPServer(("localhost", OAUTH_CALLBACK_PORT), OAuthCallback)
        server.timeout = 300

        print("\n📱 Opening browser for Google login...")
        webbrowser.open(auth_url)
        print("⏳ Waiting for login...\n")

        # Wait for callback
        while OAuthCallback.auth_code is None and OAuthCallback.error is None:
            server.handle_request()
        server.server_close()

        if OAuthCallback.error:
            print(f"❌ Login failed: {OAuthCallback.error}")
            return False

        # Exchange code for tokens
        print("🔄 Getting tokens...")
        try:
            import urllib.request
            import urllib.parse

            data = urllib.parse.urlencode({
                "client_id": self.config["client_id"],
                "client_secret": self.config["client_secret"],
                "code": OAuthCallback.auth_code,
                "grant_type": "authorization_code",
                "redirect_uri": OAUTH_REDIRECT_URI,
            }).encode()

            req = urllib.request.Request(GOOGLE_TOKEN_URL, data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())

            # Get user info
            info_req = urllib.request.Request(GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {resp['access_token']}"})
            info = json.loads(urllib.request.urlopen(info_req, timeout=10).read())

            self.tokens = {
                "access_token": resp["access_token"],
                "refresh_token": resp.get("refresh_token"),
                "expires_at": (datetime.now() + timedelta(seconds=resp.get("expires_in", 3600))).isoformat(),
                "email": info.get("email", "unknown"),
                "name": info.get("name", ""),
            }
            self._save_tokens()

            print(f"\n✅ Logged in as: {self.tokens['email']}")
            print("=" * 50 + "\n")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    async def refresh(self) -> bool:
        """Refresh access token."""
        if not self.tokens or not self.tokens.get("refresh_token"):
            return False

        async with httpx.AsyncClient() as client:
            resp = await client.post(GOOGLE_TOKEN_URL, data={
                "client_id": self.config["client_id"],
                "client_secret": self.config["client_secret"],
                "refresh_token": self.tokens["refresh_token"],
                "grant_type": "refresh_token",
            })
            if resp.status_code != 200:
                return False

            data = resp.json()
            self.tokens["access_token"] = data["access_token"]
            self.tokens["expires_at"] = (datetime.now() + timedelta(seconds=data.get("expires_in", 3600))).isoformat()
            self._save_tokens()
            return True

    async def get_token(self) -> Optional[str]:
        """Get valid access token."""
        if not self.is_logged_in:
            return None
        if self.is_expired:
            if not await self.refresh():
                return None
        return self.tokens.get("access_token")


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
    """Check auth on startup and prompt login if needed."""
    print("\n" + "=" * 50)
    print("  GEMINI PROXY FOR CLAUDE CODE")
    print("=" * 50)

    if not auth.is_configured:
        print("\n⚠️  Not configured. Starting setup...")
        if not auth.setup():
            print("\n❌ Setup required. Exiting.")
            sys.exit(1)

    if not auth.is_logged_in:
        print("\n⚠️  Not logged in. Starting login...")
        if not auth.login():
            print("\n❌ Login required. Exiting.")
            sys.exit(1)
    elif auth.is_expired:
        print("\n🔄 Token expired, refreshing...")
        if not await auth.refresh():
            print("⚠️  Refresh failed. Re-logging in...")
            if not auth.login():
                print("\n❌ Login required. Exiting.")
                sys.exit(1)

    print(f"\n✅ Ready! Logged in as: {auth.tokens.get('email', 'unknown')}")
    print("=" * 50 + "\n")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "gemini-proxy",
        "configured": auth.is_configured,
        "authenticated": auth.is_logged_in,
        "email": auth.tokens.get("email") if auth.tokens else None
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in MODEL_MAPPING]}


@app.post("/v1/messages")
async def messages(request: Request):
    """Main endpoint - translates Anthropic to Gemini."""
    token = await auth.get_token()
    if not token:
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
    url = f"{GEMINI_API_BASE}/models/{gemini_model}:{action}"
    if stream: url += "?alt=sse"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=300) as client:
        if stream:
            async with client.stream("POST", url, json=gemini_req, headers=headers) as resp:
                if resp.status_code != 200:
                    raise HTTPException(resp.status_code, (await resp.aread()).decode())
                return StreamingResponse(stream_response(resp, model), media_type="text/event-stream")
        else:
            resp = await client.post(url, json=gemini_req, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, resp.text)
            return JSONResponse(from_gemini(resp.json(), model))


if __name__ == "__main__":
    import uvicorn

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--login":
            auth.login()
        elif cmd == "--setup":
            auth.setup()
        elif cmd == "--status":
            print(f"\nConfigured: {'✅' if auth.is_configured else '❌'}")
            print(f"Logged in:  {'✅' if auth.is_logged_in else '❌'}")
            if auth.tokens:
                print(f"Email: {auth.tokens.get('email')}")
                print(f"Expires: {auth.tokens.get('expires_at')}")
        else:
            print("Commands: --login, --setup, --status")
    else:
        # Start server - will auto-prompt for login if needed
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8081)))
