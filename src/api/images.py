"""API routes for Images"""
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Body
from typing import Optional, List, Union
import json
import base64
from ..core.auth import verify_api_key_header
from ..services.generation_handler import GenerationHandler
from ..core.logger import debug_logger

router = APIRouter()
generation_handler: GenerationHandler = None

def set_generation_handler(handler: GenerationHandler):
    global generation_handler
    generation_handler = handler

@router.post("/v1/images/generations")
async def create_image_generation(
    request: Request,
    api_key: str = Depends(verify_api_key_header)
):
    """
    Generate images from a text prompt.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    prompt = body.get("prompt")
    model = body.get("model")
    size = body.get("size")
    n = body.get("n", 1)
    response_format = body.get("response_format", "url")
    user = body.get("user")
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    # Extract optional image input for I2I
    image_input = body.get("image")
    images_bytes = []

    if image_input:
        import re
        from ..api.routes import retrieve_image_data # Reuse existing function
        
        # Normalize to list
        inputs = image_input if isinstance(image_input, list) else [image_input]
        
        for item in inputs:
            if not isinstance(item, str):
                continue
                
            if item.startswith("data:image"):
                # Base64
                match = re.search(r"base64,(.+)", item)
                if match:
                    try:
                        image_base64 = match.group(1)
                        img_bytes = base64.b64decode(image_base64)
                        images_bytes.append(img_bytes)
                    except Exception as e:
                        debug_logger.log_warning(f"Failed to decode base64 image: {str(e)}")
            elif item.startswith("http://") or item.startswith("https://"):
                # URL
                try:
                    downloaded = await retrieve_image_data(item)
                    if downloaded:
                        images_bytes.append(downloaded)
                except Exception as e:
                    debug_logger.log_warning(f"Failed to download image from {item}: {str(e)}")

    if not generation_handler:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    try:
        result = await generation_handler.handle_image_api_generation(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            images=images_bytes if images_bytes else None,
            response_format=response_format,
            user=user
        )
        return result
    except Exception as e:
        debug_logger.log_error(f"Image generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/images/edits")
async def create_image_edit(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    size: Optional[str] = Form(None),
    response_format: Optional[str] = Form("url"),
    user: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key_header)
):
    """
    Edit or variation of an image.
    Flow2API treats this as Image-to-Image generation logic.
    """
    if not generation_handler:
        raise HTTPException(status_code=500, detail="Generation handler not initialized")

    # Read image
    try:
        image_bytes = await image.read()
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Image is empty")
    
    # We currently accept mask but don't strictly use it in flow logic yet (as I2I is main flow)
    # If upstream supports mask, we could pass it. But Flow usually just takes init image.
    
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    try:
        result = await generation_handler.handle_image_api_generation(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            images=[image_bytes],
            response_format=response_format,
            user=user
        )
        return result
    except Exception as e:
        debug_logger.log_error(f"Image edit failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
