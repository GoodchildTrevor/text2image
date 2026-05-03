# app/routers/openai_compat.py
import base64, time
from fastapi import APIRouter, HTTPException
from io import BytesIO
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject
from app.service import run_inference
import logging, os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

VALID_SIZES = {"256x256", "512x512", "1024x1024"}

DEFAULT_STEPS = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))

@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest) -> ImageGenerationResponse:
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
        image, _ = run_inference(
            prompt=request.prompt,
            width=w,
            height=h,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageObject(b64_json=b64, revised_prompt=request.prompt)],
        )
    except Exception as e:
        logger.error(f"OpenAI-compat generation error: {e}")
        raise HTTPException(500, f"Image generation failed: {e}")

@router.get("/models")
def list_models():
    return {"object": "list", "data": [
        {"id": "flux-schnell", "object": "model",
         "created": 1700000000, "owned_by": "local"}
    ]}
