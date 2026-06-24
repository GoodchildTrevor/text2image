import base64
import time
import logging
import os
from fastapi import APIRouter, HTTPException
<<<<<<< HEAD
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject, ImageEditRequest
from app.service import generate_image, edit_image
=======
from io import BytesIO
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject
from app.service import run_inference
>>>>>>> 6e298f99077d321e30a27d9fd2c4084df75c6129

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

DEFAULT_STEPS = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))
VALID_SIZES = {"864x1184", "1184x864", "768x1344", "1344x768", "1024x1024"}



@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest):
    """Generate an image from a text prompt via OpenAI-compatible API.

    Validates size, n, and response_format parameters, then calls the
    image generation service. Returns a base64-encoded image.

    :param request: The image generation request containing model, prompt,
        size, and optional tuning parameters.
    :returns: ImageGenerationResponse with a base64-encoded PNG image.
    :raises HTTPException 400: If size is invalid, n != 1, or prompt is empty.
    :raises HTTPException 400: If response_format is not 'b64_json' or 'url'.
    :raises HTTPException 501: If response_format='url' is requested.
    :raises HTTPException 500: If an unexpected error occurs during
        image generation.
    """
    logger.info(f"Received: model={request.model!r}, size={request.size!r}, "
                f"n={request.n}, response_format={request.response_format!r}, "
                f"prompt={request.prompt[:80]!r}")
    if request.size not in VALID_SIZES:
        raise HTTPException(400, f"size must be one of {VALID_SIZES}")
    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if request.response_format not in ("b64_json", "url"):
        raise HTTPException(400, "response_format must be 'b64_json' or 'url'")
    if request.response_format == "url":
        raise HTTPException(501, "response_format='url' is not implemented yet")

    w, h = map(int, request.size.split("x"))

    try:
<<<<<<< HEAD
        img_bytes, revised_prompt = await generate_image(
            model=request.model,
=======
        image, _ = await run_inference(
>>>>>>> 6e298f99077d321e30a27d9fd2c4084df75c6129
            prompt=request.prompt,
            width=w,
            height=h,
            steps=DEFAULT_STEPS,
            guidance=DEFAULT_GUIDANCE,
        )
        b64 = base64.b64encode(img_bytes).decode()
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageObject(b64_json=b64, revised_prompt=revised_prompt)]
        )
    except ValueError as e:
        logger.error(f"ValueError in generate_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, "Image generation failed")


<<<<<<< HEAD
@router.post("/images/edits", response_model=ImageGenerationResponse)
async def openai_edit(request: ImageEditRequest):
    """Edit an existing image based on a text prompt via OpenAI-compatible API.

    Validates size and n parameters, then calls the image edit service.
    Returns a base64-encoded edited image.

    :param request: The image edit request containing model, prompt,
        input image (base64), and size.
    :returns: ImageGenerationResponse with a base64-encoded PNG image.
    :raises HTTPException 400: If size is invalid or n != 1.
    :raises HTTPException 500: If an unexpected error occurs during
        image editing.
    """
    logger.info(f"Edit: model={request.model!r}, size={request.size!r}, "
                f"prompt={request.prompt[:80]!r}")
    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if request.size not in VALID_SIZES:
        raise HTTPException(400, f"size must be one of {VALID_SIZES}")

    try:
        img_bytes, revised_prompt = await edit_image(
            model=request.model,
            prompt=request.prompt,
            image_b64=request.image,
        )
        b64 = base64.b64encode(img_bytes).decode()
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageObject(b64_json=b64, revised_prompt=revised_prompt)]
        )
    except ValueError as e:
        logger.error(f"ValueError in edit_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(500, "Image edit failed")
=======
@router.get("/models")
def list_models():
    return {"object": "list", "data": [
        {"id": "flux-schnell", "object": "model",
         "created": int(time.time()), "owned_by": "local"}
    ]}
>>>>>>> 6e298f99077d321e30a27d9fd2c4084df75c6129
