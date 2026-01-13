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
import secrets
import threading
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
CONFIG_FILE = TOKEN_DIR / "config.json"

# OAuth Configuration
OAUTH_CALLBACK_PORT = 8089
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

# Gemini API endpoint
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Scopes needed for Gemini API
SCOPES = [
    "https://www.googleapis.com/auth/generative-language.retriever",
    "https://www.googleapis.com/auth/cloud-platform",
    "openid",
    "email",
    "profile",
]

# Supported models
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


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    auth_code = None
    error = None

    def do_GET(self):
        """Handle GET request from OAuth redirect."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <head><title>Login Successful</title></head>
                <body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div style="background: white; padding: 40px; border-radius: 16px; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.2);">
                        <h1 style="color: #22c55e; margin: 0 0 16px 0;">Login Successful!</h1>
                        <p style="color: #666; margin: 0;">You can close this window and return to the terminal.</p>
                    </div>
                </body>
                </html>
            """)
        elif "error" in params:
            OAuthCallbackHandler.error = params.get("error_description", params["error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
                <html>
                <head><title>Login Failed</title></head>
                <body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #fee;">
                    <div style="background: white; padding: 40px; border-radius: 16px; text-align: center;">
                        <h1 style="color: #ef4444;">Login Failed</h1>
                        <p style="color: #666;">{OAuthCallbackHandler.error}</p>
                    </div>
                </body>
                </html>
            """.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass


class GoogleOAuthManager:
    """Manages Google OAuth tokens for Gemini API access."""

    def __init__(self):
        self.token_data: Optional[dict] = None
        self.config: dict = {}
        self.load_config()
        self.load_tokens()

    def load_config(self):
        """Load OAuth client config."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        # Also check environment variables
        if os.getenv("GOOGLE_CLIENT_ID"):
            self.config["client_id"] = os.getenv("GOOGLE_CLIENT_ID")
        if os.getenv("GOOGLE_CLIENT_SECRET"):
            self.config["client_secret"] = os.getenv("GOOGLE_CLIENT_SECRET")

    def save_config(self):
        """Save OAuth client config."""
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)
        os.chmod(CONFIG_FILE, 0o600)

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
        os.chmod(TOKEN_FILE, 0o600)
        logger.info("Tokens saved")

    def is_configured(self) -> bool:
        """Check if OAuth client is configured."""
        return bool(self.config.get("client_id") and self.config.get("client_secret"))

    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        return bool(self.token_data and self.token_data.get("access_token"))

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

        if not self.is_configured():
            logger.error("OAuth client not configured")
            return False

        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": self.config["client_id"],
                    "client_secret": self.config["client_secret"],
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

    def setup_oauth_client(self):
        """Interactive setup for OAuth client credentials."""
        print("\n" + "=" * 60)
        print("  Google OAuth Setup for Gemini API")
        print("=" * 60)
        print("\nTo use this proxy, you need to create OAuth credentials:")
        print("\n1. Go to: https://console.cloud.google.com/apis/credentials")
        print("2. Create a new project (or select existing)")
        print("3. Click '+ CREATE CREDENTIALS' > 'OAuth client ID'")
        print("4. Select 'Desktop app' as application type")
        print("5. Name it (e.g., 'Gemini Proxy')")
        print("6. Copy the Client ID and Client Secret")
        print("\n" + "-" * 60)

        client_id = input("\nEnter Client ID: ").strip()
        client_secret = input("Enter Client Secret: ").strip()

        if client_id and client_secret:
            self.config["client_id"] = client_id
            self.config["client_secret"] = client_secret
            self.save_config()
            print("\n✅ OAuth client configured!")
            return True
        else:
            print("\n❌ Invalid credentials")
            return False

    def oauth_login(self) -> bool:
        """Perform OAuth login using browser redirect flow."""
        if not self.is_configured():
            print("\n❌ OAuth client not configured. Run setup first.")
            return False

        print("\n" + "=" * 50)
        print("  Google OAuth Login")
        print("=" * 50)

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        auth_params = {
            "client_id": self.config["client_id"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{GOOGLE_AUTH_URL}?{urlencode(auth_params)}"

        # Reset handler state
        OAuthCallbackHandler.auth_code = None
        OAuthCallbackHandler.error = None

        # Start local server for callback
        server = HTTPServer(("localhost", OAUTH_CALLBACK_PORT), OAuthCallbackHandler)
        server.timeout = 300  # 5 minute timeout

        print(f"\n📱 Opening browser for Google login...")
        print(f"   (If browser doesn't open, visit this URL:)")
        print(f"   {auth_url[:80]}...")

        # Open browser
        webbrowser.open(auth_url)

        print("\n⏳ Waiting for login...")

        # Wait for callback
        while OAuthCallbackHandler.auth_code is None and OAuthCallbackHandler.error is None:
            server.handle_request()

        server.server_close()

        if OAuthCallbackHandler.error:
            print(f"\n❌ Login failed: {OAuthCallbackHandler.error}")
            return False

        if not OAuthCallbackHandler.auth_code:
            print("\n❌ No authorization code received")
            return False

        # Exchange code for tokens
        print("\n🔄 Exchanging code for tokens...")

        try:
            import urllib.request
            import urllib.parse

            token_data = {
                "client_id": self.config["client_id"],
                "client_secret": self.config["client_secret"],
                "code": OAuthCallbackHandler.auth_code,
                "grant_type": "authorization_code",
                "redirect_uri": OAUTH_REDIRECT_URI,
            }

            req = urllib.request.Request(
                GOOGLE_TOKEN_URL,
                data=urllib.parse.urlencode(token_data).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                tokens = json.loads(response.read().decode())

            # Get user info
            userinfo_req = urllib.request.Request(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {tokens['access_token']}"}
            )

            with urllib.request.urlopen(userinfo_req, timeout=10) as response:
                userinfo = json.loads(response.read().decode())

            # Save tokens
            self.token_data = {
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token"),
                "expires_at": (
                    datetime.now() + timedelta(seconds=tokens.get("expires_in", 3600))
                ).isoformat(),
                "email": userinfo.get("email", "unknown"),
                "name": userinfo.get("name", ""),
                "picture": userinfo.get("picture", ""),
                "created_at": datetime.now().isoformat(),
                "last_refresh": datetime.now().isoformat(),
            }
            self.save_tokens()

            print(f"\n✅ Logged in as: {self.token_data['email']}")
            print("=" * 50 + "\n")
            return True

        except Exception as e:
            print(f"\n❌ Token exchange failed: {e}")
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
                elif block.get("type") == "tool_use":
                    # Handle tool calls
                    parts.append({
                        "functionCall": {
                            "name": block.get("name", ""),
                            "args": block.get("input", {})
                        }
                    })
                elif block.get("type") == "tool_result":
                    # Handle tool results
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = " ".join(
                            b.get("text", "") for b in tool_content
                            if b.get("type") == "text"
                        )
                    parts.append({
                        "functionResponse": {
                            "name": block.get("tool_use_id", "tool"),
                            "response": {"result": tool_content}
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

    # Handle tools for function calling
    if "tools" in anthropic_request:
        function_declarations = []
        for tool in anthropic_request["tools"]:
            func_decl = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            if "input_schema" in tool:
                func_decl["parameters"] = tool["input_schema"]
            function_declarations.append(func_decl)

        if function_declarations:
            gemini_request["tools"] = [{"functionDeclarations": function_declarations}]

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
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    content.append({
                        "type": "tool_use",
                        "id": f"toolu_{fc.get('name', 'unknown')}_{secrets.token_hex(4)}",
                        "name": fc.get("name", ""),
                        "input": fc.get("args", {})
                    })
                    stop_reason = "tool_use"

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
        "id": f"msg_gemini_{secrets.token_hex(8)}",
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

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_gemini_{secrets.token_hex(8)}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
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
    print("\n" + "=" * 50)
    print("  Gemini Proxy for Claude Code (OAuth)")
    print("=" * 50)

    if not oauth_manager.is_configured():
        print("\n⚠️  OAuth client not configured!")
        print("   Run: gemini-login (to set up and login)")
    elif not oauth_manager.is_authenticated():
        print("\n⚠️  Not logged in to Google")
        print("   Run: gemini-login")
    elif oauth_manager.is_expired():
        print("\n🔄 Token expired, refreshing...")
        success = await oauth_manager.refresh_token()
        if not success:
            print("⚠️  Refresh failed. Run: gemini-login")
        else:
            print(f"✅ Logged in as: {oauth_manager.token_data.get('email', 'unknown')}")
    else:
        print(f"\n✅ Logged in as: {oauth_manager.token_data.get('email', 'unknown')}")

    print("=" * 50 + "\n")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gemini-proxy-claudecode",
        "configured": oauth_manager.is_configured(),
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
    if not oauth_manager.is_configured():
        raise HTTPException(status_code=400, detail="OAuth not configured. Run gemini-login from terminal first.")

    # This endpoint is for API calls, actual login should be done via CLI
    return {"message": "Please run 'gemini-login' from terminal for interactive login"}


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

    # Check authentication
    access_token = await oauth_manager.get_access_token()
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Run 'gemini-login' to login."
        )

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    model = body.get("model", "gemini-2.0-flash")
    gemini_model = get_gemini_model_name(model)
    logger.info(f"Request for model: {model} -> {gemini_model}")

    # Convert request
    gemini_request = convert_anthropic_to_gemini(body)
    logger.debug(f"Gemini request: {json.dumps(gemini_request, indent=2)}")

    # Build URL
    stream = body.get("stream", False)
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{GEMINI_API_BASE}/models/{gemini_model}:{action}"
    if stream:
        url += "?alt=sse"

    headers = {
        "Authorization": f"Bearer {access_token}",
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


def cli_setup():
    """CLI command to set up OAuth client."""
    oauth_manager.setup_oauth_client()


def cli_login():
    """CLI command to perform OAuth login."""
    if not oauth_manager.is_configured():
        print("\n⚠️  OAuth client not configured. Setting up...")
        if not oauth_manager.setup_oauth_client():
            return

    oauth_manager.oauth_login()


def cli_status():
    """CLI command to check status."""
    print("\n" + "=" * 50)
    print("  Gemini Proxy Status")
    print("=" * 50)
    print(f"\nConfig file: {CONFIG_FILE}")
    print(f"Token file:  {TOKEN_FILE}")
    print(f"\nOAuth configured: {'✅ Yes' if oauth_manager.is_configured() else '❌ No'}")
    print(f"Authenticated:    {'✅ Yes' if oauth_manager.is_authenticated() else '❌ No'}")

    if oauth_manager.token_data:
        print(f"\nLogged in as: {oauth_manager.token_data.get('email', 'unknown')}")
        print(f"Token expires: {oauth_manager.token_data.get('expires_at', 'unknown')}")
        print(f"Last refresh:  {oauth_manager.token_data.get('last_refresh', 'unknown')}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    import uvicorn

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--setup":
            cli_setup()
        elif cmd == "--login":
            cli_login()
        elif cmd == "--status":
            cli_status()
        elif cmd == "--help":
            print("""
Gemini Proxy for Claude Code

Commands:
  --setup   Configure OAuth client credentials
  --login   Login with Google OAuth
  --status  Check authentication status
  --help    Show this help message

To start the proxy server:
  python server.py
  # or
  uvicorn server:app --host 0.0.0.0 --port 8081
""")
        else:
            print(f"Unknown command: {cmd}")
            print("Use --help for available commands")
    else:
        port = int(os.getenv("PORT", 8081))
        uvicorn.run(app, host="0.0.0.0", port=port)
