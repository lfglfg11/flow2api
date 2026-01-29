"""Gemini API routes"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import re
import time
import uuid
from typing import List, Optional, Dict, Any
from ..core.auth import verify_api_key_header
from ..services.generation_handler import GenerationHandler, MODEL_CONFIG
from ..core.logger import debug_logger
from .routes import retrieve_image_data

from ..core.auth import AuthManager

async def verify_gemini_auth(
    request: Request,
    x_goog_api_key: Optional[str] = None,
    key: Optional[str] = None
):
    """
    Flexible auth for Gemini:
    1. x-goog-api-key header
    2. key query param
    3. Authorization header (Bearer or raw)
    """
    # 1. Check x-goog-api-key header
    if request.headers.get("x-goog-api-key"):
        token = request.headers.get("x-goog-api-key")
        if AuthManager.verify_api_key(token):
             return token
    
    # 2. Check query param 'key'
    if request.query_params.get("key"):
        token = request.query_params.get("key")
        if AuthManager.verify_api_key(token):
             return token
             
    # 3. Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header:
        # Support "Bearer <token>" and just "<token>"
        token = auth_header.replace("Bearer ", "").strip()
        if AuthManager.verify_api_key(token):
            return token
            
    raise HTTPException(status_code=401, detail="Invalid API key")

router = APIRouter()
generation_handler: GenerationHandler = None

def set_generation_handler(handler: GenerationHandler):
    global generation_handler
    generation_handler = handler

async def parse_gemini_content(body: Dict[str, Any]) -> tuple[str, List[bytes], Dict[str, Any]]:
    """Parse Gemini request body to extract prompt and images"""
    contents = body.get("contents", [])
    generation_config = body.get("generation_config", {})
    
    prompt_parts = []
    images: List[bytes] = []
    
    if not isinstance(contents, list):
        return "", [], generation_config

    # Iterate through contents to extract text and images
    for content in contents:
        parts = content.get("parts", [])
        if not parts:
            continue
            
        for part in parts:
            # Text part
            if "text" in part:
                text = part["text"]
                # Extract inline URLs from text if present (simple regex)
                # Similar to v03ai logic: regex extract http URLs
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, text)
                
                clean_text = text
                for url in urls:
                    # Remove URL from text
                    clean_text = clean_text.replace(url, "").strip()
                    # Download image
                    img_data = await retrieve_image_data(url)
                    if img_data:
                        images.append(img_data)
                
                if clean_text:
                    prompt_parts.append(clean_text)
            
            # Inline data (base64 or URL)
            elif "inline_data" in part:
                inline_data = part["inline_data"]
                mime_type = inline_data.get("mime_type", "")
                data = inline_data.get("data", "")
                
                if data:
                    # Case 1: URL in data
                    if data.startswith("http://") or data.startswith("https://"):
                        try:
                            # Reuse retrieve_image_data to download
                            img_data = await retrieve_image_data(data)
                            if img_data:
                                images.append(img_data)
                        except Exception as e:
                            debug_logger.log_warning(f"Failed to download image from inline_data URL: {data}, error: {str(e)}")

                    # Case 2: Base64
                    elif "image" in mime_type:
                        try:
                            # 尝试清理非base64字符(如换行)
                            clean_data = data.replace("\n", "").replace("\r", "").strip()
                            
                            # Check if it has base64 prefix
                            if "base64," in clean_data:
                                clean_data = clean_data.split("base64,")[1]
                            
                            import base64
                            try:
                                images.append(base64.b64decode(clean_data))
                            except Exception:
                                # Try adding padding
                                padded = clean_data + '=' * (-len(clean_data) % 4)
                                images.append(base64.b64decode(padded))
                                
                        except Exception as e:
                            debug_logger.log_warning(f"Failed to decode base64 inline_data: {str(e)}")
            
            # File data (file_uri) - Not supported directly without file API, but handle if passed
            elif "file_data" in part:
                # Need to download from file_uri if accessible, or ignore
                pass

    final_prompt = "\n".join(prompt_parts).strip()
    return final_prompt, images, generation_config

def format_gemini_response(text_content: str, model: str) -> Dict[str, Any]:
    """Format final response in Gemini JSON format"""
    return {
        "candidates": [{
            "content": {
                "parts": [{"text": text_content}],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0,
            "safetyRatings": []
        }],
        "modelVersion": model
    }

def format_gemini_chunk(text_content: str, is_last: bool = False) -> str:
    """Format streaming chunk in Gemini SSE format"""
    chunk_data = {
        "candidates": [{
            "content": {
                "parts": [{"text": text_content}],
                "role": "model"
            },
            "finishReason": "STOP" if is_last else None,
            "index": 0
        }]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"

@router.post("/v1beta/models/{model}:generateContent")
async def generate_content(
    model: str,
    request: Request,
    api_key: str = Depends(verify_gemini_auth)
):
    if not generation_handler:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    prompt, images, config = await parse_gemini_content(body)
    
    if not prompt and not images:
        raise HTTPException(status_code=400, detail="No content provided")

    # Handle Generation Config (aspect ratio)
    image_config = config.get("imageConfig", {}) or config.get("image_config", {})
    aspect_ratio_param = image_config.get("aspectRatio") or image_config.get("aspect_ratio")
    
    # Do NOT modify prompt. Pass explicitly.
    
    # Call generation handler (non-streaming)
    # We use stream=False to get the full result, then wrap it
    
    final_text = ""
    try:
        # We need to accumulate the response (which is OpenAI JSON format)
        openai_json_resp = None
        async for chunk in generation_handler.handle_generation(
            model=model,
            prompt=prompt,
            images=images,
            stream=False,
            aspect_ratio=aspect_ratio_param,
            skip_availability_check=True
        ):
            openai_json_resp = chunk
        
        if openai_json_resp:
            # Parse OpenAI response to get the content (image url markdown)
            try:
                data = json.loads(openai_json_resp)
                
                # Check for error first
                if "error" in data:
                    error_msg = data["error"]["message"]
                    debug_logger.log_error(f"Gemini generation error: {error_msg}")
                    # Return error as content or raise? 
                    # Gemini usually returns 500 or 400 if it fails, or finishes with error?
                    # Let's return the error message as text for now so user sees it in 'candidates' 
                    # or better: raise proper HTTP exception
                    raise Exception(error_msg)

                if "choices" in data and len(data["choices"]) > 0:
                    final_text = data["choices"][0]["message"]["content"]
            except Exception as e:
                # If it was our raised exception, re-raise
                if str(e) == data.get("error", {}).get("message"):
                     raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
                
                # Otherwise fall back
                final_text = str(openai_json_resp)
        else:
             final_text = "Generation failed"

    except Exception as e:
        debug_logger.log_error(f"Gemini generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return format_gemini_response(final_text, model)

@router.post("/v1beta/models/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    request: Request,
    api_key: str = Depends(verify_gemini_auth)
):
    if not generation_handler:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    prompt, images, config = await parse_gemini_content(body)

    image_config = config.get("imageConfig", {}) or config.get("image_config", {})
    aspect_ratio_param = image_config.get("aspectRatio") or image_config.get("aspect_ratio")
    
    async def generate_stream():
        # Call generation handler (streaming)
        # It yields OpenAI SSE lines: "data: { ... }" or "data: [DONE]"
        try:
            async for chunk in generation_handler.handle_generation(
                model=model,
                prompt=prompt,
                images=images,
                stream=True,
                aspect_ratio=aspect_ratio_param
            ):
                # chunk is "data: ..." string
                if chunk.startswith("data: "):
                    content_str = chunk[6:].strip()
                    if content_str == "[DONE]":
                        break
                    try:
                        data = json.loads(content_str)
                        # Extract delta content
                        # OpenAI Delta: choices[0].delta.content
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            text_part = delta.get("content", "")
                            if text_part:
                                yield format_gemini_chunk(text_part)
                    except:
                        pass
                else:
                    # Maybe raw text or comment? ignore or verify
                    pass
        except Exception as e:
            yield format_gemini_chunk(f"Error: {str(e)}", is_last=True)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

@router.get("/v1beta/models")
async def list_models(api_key: str = Depends(verify_gemini_auth)):
    """List available models in Gemini format"""
    models = []
    
    # Iterate over MODEL_CONFIG to build Gemini model list
    for model_key, config in MODEL_CONFIG.items():
        model_name = f"models/{model_key}"
        display_name = config.get("model_name", model_key)
        description = f"{config.get('type', 'unknown').capitalize()} generation model"
        
        models.append({
            "name": model_name,
            "version": "001", 
            "displayName": display_name,
            "description": description,
            "inputTokenLimit": 30720, # Dummy
            "outputTokenLimit": 2048, # Dummy
            "supportedGenerationMethods": ["generateContent", "countTokens"],
            "temperature": 0.9,
            "topP": 0.95,
            "topK": 40
        })

    return {"models": models}

@router.get("/v1beta/models/{model}")
async def get_model(
    model: str,
    api_key: str = Depends(verify_gemini_auth)
):
    """Get specific model info in Gemini format"""
    # model parameter might range from 'gemini-pro' to 'models/gemini-pro'
    clean_model_key = model.replace("models/", "")
    
    config = MODEL_CONFIG.get(clean_model_key)
    if not config:
        # Try finding if it matches a known alias, but for now simple check
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    model_name = f"models/{clean_model_key}"
    display_name = config.get("model_name", clean_model_key)
    description = f"{config.get('type', 'unknown').capitalize()} generation model"

    return {
        "name": model_name,
        "version": "001",
        "displayName": display_name,
        "description": description,
        "inputTokenLimit": 30720,
        "outputTokenLimit": 2048,
        "supportedGenerationMethods": ["generateContent", "countTokens"],
        "temperature": 0.9,
        "topP": 0.95,
        "topK": 40
    }
