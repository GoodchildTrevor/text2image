import base64, time
from fastapi import APIRouter, HTTPException
from io import BytesIO
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject
from app.service import run_inference
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

VALID_SIZES = {"256x256", "512x512", "1024x1024"}

@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest) -> ImageGenerationResponse:
    if request.size not in VALID_SIZES:
        raise HTTPException(400, f"size must be one of {VALID_SIZES}")
    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")

    w, h = map(int, request.size.split("x"))

    try:
        image, _ = run_inference(
            prompt=request.prompt,
            width=w,
            height=h,
            num_inference_steps=4,   # разумный дефолт для schnell
            guidance_scale=0.0,
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
