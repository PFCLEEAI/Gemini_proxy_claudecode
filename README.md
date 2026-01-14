# Gemini_proxy_claudecode

> Use Google Gemini models with Claude Code - **Free API key from Google AI Studio!**

```
   ____                _       _
  / ___| ___ _ __ ___ (_)_ __ (_)  _ __  _ __ _____  ___   _
 | |  _ / _ \ '_ ` _ \| | '_ \| | | '_ \| '__/ _ \ \/ / | | |
 | |_| |  __/ | | | | | | | | | | | |_) | | | (_) >  <| |_| |
  \____|\\___|_| |_| |_|_|_| |_|_| | .__/|_|  \___/_/\_\\__, |
                                  |_|   claude code    |___/
```

## What is this?

This proxy lets you use **Google Gemini models** with **Claude Code CLI** using a **free API key** from Google AI Studio. No paid subscription needed!

## Quick Start (2 minutes)

### 1. Get Free API Key

1. Go to: **https://aistudio.google.com/apikey**
2. Click "Create API Key"
3. Copy the key

### 2. Add to your shell

```bash
# Add to ~/.zshrc or ~/.bashrc
export GOOGLE_API_KEY='your-api-key-here'
```

Then reload: `source ~/.zshrc`

### 3. Start the Proxy

```bash
# Clone (first time only)
git clone https://github.com/PFCLEEAI/Gemini_proxy_claudecode.git
cd Gemini_proxy_claudecode
uv sync  # or: pip install -r requirements.txt

# Start proxy
uv run uvicorn server:app --host 0.0.0.0 --port 8081 --reload
```

### 4. Run Claude Code with Gemini

```bash
ANTHROPIC_BASE_URL="http://localhost:8081" \
ANTHROPIC_AUTH_TOKEN="test" \
ANTHROPIC_MODEL="gemini-2.0-flash" \
claude --dangerously-skip-permissions
```

## Shell Aliases (Recommended)

Add these to your `~/.zshrc`:

```bash
# Your Gemini API key
export GOOGLE_API_KEY='your-api-key-here'

# Start/Stop Gemini proxy
alias start-gemini-proxy='cd ~/path/to/Gemini_proxy_claudecode && uv run uvicorn server:app --host 0.0.0.0 --port 8081 --reload'
alias stop-gemini-proxy='lsof -ti:8081 | xargs kill -9 2>/dev/null && echo "Stopped" || echo "Not running"'
alias gemini-status='curl -s http://localhost:8081/health | python3 -m json.tool'

# Claude Code with Gemini (uses latest models)
alias claude-gemini='ANTHROPIC_BASE_URL="http://localhost:8081" \
  ANTHROPIC_AUTH_TOKEN="test" \
  ANTHROPIC_MODEL="gemini-2.0-flash" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="gemini-2.5-pro" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="gemini-2.0-flash" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="gemini-1.5-flash" \
  CLAUDE_CODE_SUBAGENT_MODEL="gemini-2.0-flash" \
  claude --dangerously-skip-permissions'
```

## Supported Models

| Model ID | Description | Best For |
|----------|-------------|----------|
| `gemini-2.5-pro` | Latest Pro model | Complex reasoning |
| `gemini-2.5-flash` | Fast, capable | General use |
| `gemini-2.0-flash` | Very fast | Quick tasks |
| `gemini-1.5-pro` | Stable Pro | Long context (2M) |
| `gemini-1.5-flash` | Stable Flash | Cost-effective |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Main messages endpoint (Anthropic format) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check & auth status |

## How It Works

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Claude Code  │───▶│ Gemini Proxy    │───▶│   Gemini API     │
│              │◀───│ (Translates)    │◀───│ (API Key Auth)   │
└──────────────┘    └─────────────────┘    └──────────────────┘
     Port 8081           localhost         generativelanguage.googleapis.com
```

1. Claude Code sends Anthropic-format requests to proxy
2. Proxy translates to Gemini format
3. Gemini API responds
4. Proxy translates back to Anthropic format

## Multi-Proxy Setup

Use Claude, Gemini, AND OpenAI/Codex:

```bash
# Native Claude (paid subscription)
alias claude-yolo='unset ANTHROPIC_BASE_URL ... && claude --dangerously-skip-permissions'

# Gemini via this proxy (port 8081) - FREE
alias claude-yolo1='ANTHROPIC_BASE_URL="http://localhost:8081" ... claude --dangerously-skip-permissions'

# OpenAI/Codex via CLIProxyAPI (port 8317)
alias claude-yolo2='ANTHROPIC_BASE_URL="http://localhost:8317" ... claude --dangerously-skip-permissions'
```

## Troubleshooting

### "No API key found"
Get your free key from https://aistudio.google.com/apikey and set `GOOGLE_API_KEY`.

### "Model not found"
Check available models with `gemini-models` or `curl http://localhost:8081/v1/models`.

### "Connection refused"
Make sure proxy is running: `start-gemini-proxy`

## Free Tier Limits

Google AI Studio free tier is generous:
- 15 requests/minute for Flash models
- 2 requests/minute for Pro models
- No credit card required

For higher limits, consider Gemini API paid tier.

## Contributing

PRs welcome! This is a fun open-source project.

## License

MIT - Do whatever you want with it!

## Related Projects

- [GPT_proxy_claudecode](https://github.com/PFCLEEAI/GPT_proxy_claudecode) - OpenAI proxy for Claude Code

---

Made with Gemini magic
