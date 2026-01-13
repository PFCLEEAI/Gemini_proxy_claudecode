# Gemini_proxy_claudecode

> Use Google Gemini models with Claude Code via **Google OAuth** - No API key required!

```
   ____                _       _
  / ___| ___ _ __ ___ (_)_ __ (_)  _ __  _ __ _____  ___   _
 | |  _ / _ \ '_ ` _ \| | '_ \| | | '_ \| '__/ _ \ \/ / | | |
 | |_| |  __/ | | | | | | | | | | | |_) | | | (_) >  <| |_| |
  \____|\___|_| |_| |_|_|_| |_|_| | .__/|_|  \___/_/\_\\__, |
                                  |_|   claude code    |___/
```

## What is this?

This proxy lets you use **Google Gemini models** with **Claude Code CLI** using your **Google account login** (OAuth). No API key needed - just login with your Google account!

Similar to how CLIProxyAPI works for ChatGPT/Codex, this proxy:
- Opens a browser for Google OAuth login
- Stores tokens securely in `~/.gemini-proxy/`
- Auto-refreshes tokens when they expire
- Translates Anthropic API format to Gemini

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/PFCLEEAI/Gemini_proxy_claudecode.git
cd Gemini_proxy_claudecode

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Start the Proxy (Will prompt for login)

```bash
# Using uv
uv run uvicorn server:app --host 0.0.0.0 --port 8081 --reload
```

On first run, it will:
1. Open your browser to Google's login page
2. Ask you to authorize the app
3. Store your tokens locally

### 3. Run Claude Code with Gemini

```bash
ANTHROPIC_BASE_URL="http://localhost:8081" \
ANTHROPIC_AUTH_TOKEN="test" \
ANTHROPIC_MODEL="gemini-2.0-flash" \
claude --dangerously-skip-permissions
```

## Shell Aliases (Recommended)

Add these to your `~/.zshrc` or `~/.bashrc`:

```bash
# Start Gemini proxy (will prompt for OAuth login if needed)
alias start-gemini-proxy='cd ~/Documents/Coding/github/Gemini_proxy_claudecode && uv run uvicorn server:app --host 0.0.0.0 --port 8081 --reload'
alias stop-gemini-proxy='lsof -ti:8081 | xargs kill -9 2>/dev/null && echo "Gemini proxy stopped" || echo "No proxy running"'

# Re-login to Gemini (if token expires or you want to switch accounts)
alias gemini-login='cd ~/Documents/Coding/github/Gemini_proxy_claudecode && uv run python server.py --login'

# Check Gemini proxy status
alias gemini-status='curl -s http://localhost:8081/health 2>/dev/null | python3 -m json.tool || echo "Proxy not running"'

# Claude Code with Gemini
alias claude-gemini='ANTHROPIC_BASE_URL="http://localhost:8081" \
  ANTHROPIC_AUTH_TOKEN="test" \
  ANTHROPIC_MODEL="gemini-2.0-flash" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="gemini-1.5-pro" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="gemini-2.0-flash" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="gemini-1.5-flash" \
  CLAUDE_CODE_SUBAGENT_MODEL="gemini-2.0-flash" \
  claude --dangerously-skip-permissions'
```

Then reload:
```bash
source ~/.zshrc
```

## Supported Models

| Model ID | Description | Context |
|----------|-------------|---------|
| `gemini-2.0-flash` | Latest Gemini 2.0 Flash | 1M tokens |
| `gemini-2.0-flash-exp` | Gemini 2.0 Flash Experimental | 1M tokens |
| `gemini-2.0-flash-thinking-exp` | Reasoning/thinking model | 1M tokens |
| `gemini-1.5-pro` | Gemini 1.5 Pro | 2M tokens |
| `gemini-1.5-pro-latest` | Latest Gemini 1.5 Pro | 2M tokens |
| `gemini-1.5-flash` | Fast, efficient model | 1M tokens |
| `gemini-1.5-flash-latest` | Latest Gemini 1.5 Flash | 1M tokens |
| `gemini-exp-1206` | Experimental model | 1M tokens |

## How OAuth Works

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  First Run   │───▶│ Google OAuth    │───▶│  Token Storage   │
│              │    │ (Browser Login) │    │ ~/.gemini-proxy/ │
└──────────────┘    └─────────────────┘    └──────────────────┘
                                                    │
                                                    ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Claude Code  │───▶│ Gemini Proxy    │───▶│   Gemini API     │
│              │◀───│ (Auto-refresh)  │◀───│ (OAuth Bearer)   │
└──────────────┘    └─────────────────┘    └──────────────────┘
```

1. **First run**: Opens browser for Google login, stores refresh token
2. **Subsequent runs**: Uses stored token, auto-refreshes when expired
3. **Token storage**: `~/.gemini-proxy/google-oauth.json` (permission 600)

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Main messages endpoint (Anthropic format) |
| `POST /login` | Trigger OAuth login flow |
| `POST /logout` | Clear stored tokens |
| `GET /v1/models` | List available models |
| `GET /health` | Health check (shows login status) |

## Commands

```bash
# Start proxy (auto-login on first run)
start-gemini-proxy

# Check status and logged-in account
gemini-status

# Re-login or switch accounts
gemini-login

# Stop proxy
stop-gemini-proxy

# Use with Claude Code
claude-yolo1  # If using the alias setup
```

## Multi-Proxy Setup

Use Claude, Gemini, AND OpenAI:

```bash
# Native Claude (paid subscription)
alias claude-yolo='unset ANTHROPIC_BASE_URL ... && claude --dangerously-skip-permissions'

# Gemini via OAuth proxy (port 8081)
alias claude-yolo1='ANTHROPIC_BASE_URL="http://localhost:8081" ... claude --dangerously-skip-permissions'

# OpenAI/Codex via CLIProxyAPI (port 8317)
alias claude-yolo2='ANTHROPIC_BASE_URL="http://localhost:8317" ... claude --dangerously-skip-permissions'
```

## Troubleshooting

### "Not authenticated"
Run `gemini-login` or restart the proxy to trigger OAuth flow.

### "Token refresh failed"
Your refresh token may have expired. Run `gemini-login` to re-authenticate.

### "Connection refused"
Make sure the proxy is running:
```bash
gemini-status
# or
curl http://localhost:8081/health
```

### Switch Google accounts
```bash
# Logout and re-login
curl -X POST http://localhost:8081/logout
gemini-login
```

## Security

- Tokens stored in `~/.gemini-proxy/` with 600 permissions
- Refresh tokens allow long-term access without re-login
- Uses Google's standard OAuth 2.0 device flow
- No API keys stored or transmitted

## Contributing

PRs welcome! This is a fun open-source project.

## License

MIT - Do whatever you want with it!

## Related Projects

- [GPT_proxy_claudecode](https://github.com/PFCLEEAI/GPT_proxy_claudecode) - OpenAI proxy for Claude Code

---

Made with and Google OAuth magic
