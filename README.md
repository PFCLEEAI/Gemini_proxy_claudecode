# Gemini_proxy_claudecode

> Use Google Gemini models with Claude Code - A proxy that translates Anthropic API calls to Google Gemini API format.

```
   ____                _       _
  / ___| ___ _ __ ___ (_)_ __ (_)  _ __  _ __ _____  ___   _
 | |  _ / _ \ '_ ` _ \| | '_ \| | | '_ \| '__/ _ \ \/ / | | |
 | |_| |  __/ | | | | | | | | | | | |_) | | | (_) >  <| |_| |
  \____|\___|_| |_| |_|_|_| |_|_| | .__/|_|  \___/_/\_\\__, |
                                  |_|   claude code    |___/
```

## What is this?

This proxy lets you use **Google Gemini models** (Gemini 2.0, 1.5 Pro, 1.5 Flash, etc.) with **Claude Code CLI**. It translates Anthropic's Messages API format to Google's Generative AI API.

Perfect for:
- Comparing Claude vs Gemini responses
- Using Gemini's large context window (1M+ tokens)
- Experimenting with Gemini's multimodal capabilities
- Having fun with AI tooling

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Gemini_proxy_claudecode.git
cd Gemini_proxy_claudecode

# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Set your Google API Key

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

```bash
export GOOGLE_API_KEY="your-api-key-here"
# or
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

### 3. Start the Proxy

```bash
# Using uv
uv run uvicorn server:app --host 0.0.0.0 --port 8081 --reload

# Or directly
uvicorn server:app --host 0.0.0.0 --port 8081 --reload
```

### 4. Run Claude Code with the Proxy

```bash
ANTHROPIC_BASE_URL="http://localhost:8081" \
ANTHROPIC_AUTH_TOKEN="test" \
ANTHROPIC_MODEL="gemini-1.5-pro" \
claude --dangerously-skip-permissions
```

## Shell Aliases (Recommended)

Add these to your `~/.zshrc` or `~/.bashrc`:

```bash
# Start the proxy
alias start-gemini-proxy='cd ~/path/to/Gemini_proxy_claudecode && uv run uvicorn server:app --host 0.0.0.0 --port 8081 --reload'

# Claude Code with Gemini 2.0 Flash
alias claude-gemini='ANTHROPIC_BASE_URL="http://localhost:8081" \
  ANTHROPIC_AUTH_TOKEN="test" \
  ANTHROPIC_MODEL="gemini-2.0-flash" \
  ANTHROPIC_DEFAULT_OPUS_MODEL="gemini-1.5-pro" \
  ANTHROPIC_DEFAULT_SONNET_MODEL="gemini-2.0-flash" \
  ANTHROPIC_DEFAULT_HAIKU_MODEL="gemini-1.5-flash" \
  CLAUDE_CODE_SUBAGENT_MODEL="gemini-2.0-flash" \
  claude --dangerously-skip-permissions'

# Claude Code with Gemini 1.5 Pro (large context)
alias claude-gemini-pro='ANTHROPIC_BASE_URL="http://localhost:8081" \
  ANTHROPIC_AUTH_TOKEN="test" \
  ANTHROPIC_MODEL="gemini-1.5-pro" \
  claude --dangerously-skip-permissions'

# Claude Code with Gemini 2.0 Flash Thinking (experimental reasoning)
alias claude-gemini-thinking='ANTHROPIC_BASE_URL="http://localhost:8081" \
  ANTHROPIC_AUTH_TOKEN="test" \
  ANTHROPIC_MODEL="gemini-2.0-flash-thinking-exp" \
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

## Multi-Proxy Setup

Want to switch between Claude, Gemini, AND OpenAI? Here's the full setup:

```bash
# Native Claude (paid subscription)
alias claude-native='unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL && claude --dangerously-skip-permissions'

# Gemini via Gemini_proxy_claudecode (port 8081)
alias claude-gemini='ANTHROPIC_BASE_URL="http://localhost:8081" ANTHROPIC_AUTH_TOKEN="test" ANTHROPIC_MODEL="gemini-2.0-flash" claude --dangerously-skip-permissions'

# OpenAI via GPT_proxy_claudecode (port 8082)
alias claude-gpt='ANTHROPIC_BASE_URL="http://localhost:8082" ANTHROPIC_AUTH_TOKEN="test" ANTHROPIC_MODEL="gpt-4.1" claude --dangerously-skip-permissions'

# Start proxies
alias start-gemini-proxy='cd ~/path/to/Gemini_proxy_claudecode && uv run uvicorn server:app --port 8081 --reload'
alias start-gpt-proxy='cd ~/path/to/GPT_proxy_claudecode && uv run uvicorn server:app --port 8082 --reload'
```

## How It Works

```
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
│   Claude Code   │────▶│  Gemini_proxy_claudecode │────▶│   Gemini API    │
│      CLI        │◀────│        (Proxy)           │◀────│                 │
└─────────────────┘     └──────────────────────────┘     └─────────────────┘
                              │
                              ▼
                    Translates Anthropic
                    Messages API format
                    to Gemini API
```

The proxy:
1. Receives Anthropic-formatted requests from Claude Code
2. Transforms them to Google Generative AI format
3. Sends to Gemini API
4. Converts the response back to Anthropic format
5. Returns to Claude Code

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Main messages endpoint (Anthropic format) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google AI API key | Required |
| `GEMINI_API_KEY` | Alternative env name | - |
| `PORT` | Proxy server port | 8081 |
| `LOG_LEVEL` | Logging verbosity | INFO |

## Troubleshooting

### "Connection refused"
Make sure the proxy is running:
```bash
curl http://localhost:8081/health
```

### "Invalid API Key"
Check your Google API key is set:
```bash
echo $GOOGLE_API_KEY
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey)

### Model not found
Verify you're using a supported model ID (see table above).

### Rate limits
Google AI API has generous free tier limits. Check [pricing](https://ai.google.dev/pricing).

## Important Notes

- This uses the **Google AI API** (generativelanguage.googleapis.com)
- Free tier includes 15 RPM for Gemini 1.5 Pro, 60 RPM for Flash models
- Paid tier available for higher limits
- Get your API key at [Google AI Studio](https://aistudio.google.com/apikey)

## Features

- Full streaming support
- Tool/function calling support
- Multimodal (images) support
- System instructions support
- Temperature, top_p, top_k parameters

## Contributing

PRs welcome! This is a fun open-source project.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/cool-thing`)
3. Commit changes (`git commit -am 'Add cool thing'`)
4. Push (`git push origin feature/cool-thing`)
5. Open a PR

## License

MIT - Do whatever you want with it!

## Related Projects

- [GPT_proxy_claudecode](https://github.com/PFCLEEAI/GPT_proxy_claudecode) - OpenAI proxy for Claude Code

---

Made with and lots of API calls
