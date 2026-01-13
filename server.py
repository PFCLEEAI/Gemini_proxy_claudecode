"""
Gemini_proxy_claudecode - Google Gemini to Anthropic API Proxy

Translates Anthropic Messages API requests to Google Gemini API format,
allowing Claude Code CLI to use Gemini models.
"""

import os
import json
import logging
import base64
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import google.generativeai as genai

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
    description="Google Gemini to Anthropic API Proxy",
    version="1.0.0"
)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Supported models mapping (Anthropic model name -> Gemini model name)
MODEL_MAPPING = {
    # Direct Gemini model names
    "gemini-2.0-flash-exp": "gemini-2.0-flash-exp",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-pro-latest": "gemini-1.5-pro-latest",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
    "gemini-1.0-pro": "gemini-1.0-pro",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-pro-vision": "gemini-1.5-pro",
    # Experimental models
    "gemini-exp-1206": "gemini-exp-1206",
    "gemini-2.0-flash-thinking-exp": "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-1219": "gemini-2.0-flash-thinking-exp-1219",
    # Map Claude model names to Gemini equivalents
    "claude-3-opus": "gemini-1.5-pro",
    "claude-3-sonnet": "gemini-1.5-flash",
    "claude-3-haiku": "gemini-1.5-flash",
}


def get_gemini_model_name(model: str) -> str:
    """Map Anthropic/custom model name to Gemini model name."""
    return MODEL_MAPPING.get(model, model)


def convert_anthropic_to_gemini(anthropic_request: dict) -> tuple[str, list, dict]:
    """
    Convert Anthropic Messages API format to Gemini format.
    Returns: (system_instruction, contents, generation_config)
    """

    # Extract system instruction
    system_instruction = None
    if "system" in anthropic_request:
        system_content = anthropic_request["system"]
        if isinstance(system_content, list):
            system_instruction = " ".join(
                block.get("text", "") for block in system_content
                if block.get("type") == "text"
            )
        else:
            system_instruction = system_content

    # Convert messages to Gemini format
    contents = []

    for msg in anthropic_request.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        # Map roles: Anthropic uses "user"/"assistant", Gemini uses "user"/"model"
        gemini_role = "model" if role == "assistant" else "user"

        parts = []

        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif block.get("type") == "image":
                    # Handle base64 images
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        parts.append({
                            "inline_data": {
                                "mime_type": media_type,
                                "data": data
                            }
                        })
                elif block.get("type") == "tool_use":
                    # Convert tool use to function call
                    parts.append({
                        "function_call": {
                            "name": block.get("name", ""),
                            "args": block.get("input", {})
                        }
                    })
                elif block.get("type") == "tool_result":
                    # Convert tool result to function response
                    tool_result = block.get("content", "")
                    if isinstance(tool_result, list):
                        tool_result = " ".join(
                            b.get("text", "") for b in tool_result if b.get("type") == "text"
                        )
                    parts.append({
                        "function_response": {
                            "name": block.get("tool_use_id", "unknown"),
                            "response": {"result": tool_result}
                        }
                    })

        if parts:
            contents.append({
                "role": gemini_role,
                "parts": parts
            })

    # Build generation config
    generation_config = {}

    if "max_tokens" in anthropic_request:
        generation_config["max_output_tokens"] = anthropic_request["max_tokens"]

    if "temperature" in anthropic_request:
        generation_config["temperature"] = anthropic_request["temperature"]

    if "top_p" in anthropic_request:
        generation_config["top_p"] = anthropic_request["top_p"]

    if "top_k" in anthropic_request:
        generation_config["top_k"] = anthropic_request["top_k"]

    return system_instruction, contents, generation_config


def convert_tools_to_gemini(anthropic_tools: list) -> list:
    """Convert Anthropic tools format to Gemini function declarations."""

    function_declarations = []

    for tool in anthropic_tools:
        func_decl = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
        }

        # Convert input_schema to parameters
        if "input_schema" in tool:
            func_decl["parameters"] = tool["input_schema"]

        function_declarations.append(func_decl)

    return function_declarations


def convert_gemini_to_anthropic(gemini_response, model: str) -> dict:
    """Convert Gemini response to Anthropic Messages format."""

    content = []
    stop_reason = "end_turn"

    try:
        # Get the response text/parts
        candidate = gemini_response.candidates[0]

        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                content.append({
                    "type": "text",
                    "text": part.text
                })
            elif hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                content.append({
                    "type": "tool_use",
                    "id": f"toolu_{func_call.name}",
                    "name": func_call.name,
                    "input": dict(func_call.args) if func_call.args else {}
                })
                stop_reason = "tool_use"

        # Map finish reason
        if hasattr(candidate, 'finish_reason'):
            finish_reason = str(candidate.finish_reason)
            if "STOP" in finish_reason:
                stop_reason = "end_turn"
            elif "MAX_TOKENS" in finish_reason:
                stop_reason = "max_tokens"
            elif "SAFETY" in finish_reason:
                stop_reason = "end_turn"

    except Exception as e:
        logger.error(f"Error converting Gemini response: {e}")
        content.append({
            "type": "text",
            "text": str(gemini_response.text) if hasattr(gemini_response, 'text') else ""
        })

    # Estimate token usage (Gemini doesn't always provide this)
    input_tokens = 0
    output_tokens = 0

    if hasattr(gemini_response, 'usage_metadata'):
        usage = gemini_response.usage_metadata
        input_tokens = getattr(usage, 'prompt_token_count', 0)
        output_tokens = getattr(usage, 'candidates_token_count', 0)

    return {
        "id": f"msg_gemini_{hash(str(gemini_response)) % 10000000}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }


async def stream_gemini_to_anthropic(response_stream, model: str) -> AsyncGenerator[str, None]:
    """Stream Gemini responses converted to Anthropic SSE format."""

    # Send initial message_start event
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_gemini_stream', 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    # Send content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    full_content = ""

    try:
        for chunk in response_stream:
            if hasattr(chunk, 'text') and chunk.text:
                text = chunk.text
                full_content += text

                # Send content_block_delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

    except Exception as e:
        logger.error(f"Error in stream: {e}")
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': f'Error: {str(e)}'}})}\n\n"

    # Send content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Send message_delta with stop_reason
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': len(full_content.split())}})}\n\n"

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "gemini-proxy-claudecode"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    models = [
        {"id": model, "object": "model"}
        for model in MODEL_MAPPING.keys()
    ]
    return {"object": "list", "data": models}


@app.post("/v1/messages")
async def messages(request: Request):
    """
    Main messages endpoint - translates Anthropic Messages API to Gemini.
    """

    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY or GEMINI_API_KEY not configured"
        )

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    model = body.get("model", "gemini-1.5-flash")
    gemini_model_name = get_gemini_model_name(model)
    logger.info(f"Received request for model: {model} -> {gemini_model_name}")

    # Convert request format
    system_instruction, contents, generation_config = convert_anthropic_to_gemini(body)
    logger.debug(f"System: {system_instruction}")
    logger.debug(f"Contents: {json.dumps(contents, indent=2, default=str)}")
    logger.debug(f"Config: {generation_config}")

    try:
        # Initialize Gemini model
        model_kwargs = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        # Handle tools
        if "tools" in body:
            function_declarations = convert_tools_to_gemini(body["tools"])
            model_kwargs["tools"] = [{"function_declarations": function_declarations}]

        gemini_model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config=generation_config if generation_config else None,
            **model_kwargs
        )

        if body.get("stream", False):
            # Streaming response
            response_stream = gemini_model.generate_content(
                contents,
                stream=True
            )

            return StreamingResponse(
                stream_gemini_to_anthropic(response_stream, model),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            response = gemini_model.generate_content(contents)
            anthropic_response = convert_gemini_to_anthropic(response, model)

            logger.debug(f"Returning response: {json.dumps(anthropic_response, indent=2)}")
            return JSONResponse(content=anthropic_response)

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "server_error", "message": str(exc)}}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
