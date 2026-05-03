from fastapi import APIRouter, HTTPException, Response
from io import BytesIO
from app.config import TextToImageRequest
from app.service import run_inference
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_class=Response, responses={
    200: {"content": {"image/png": {}}},
})
async def generate_image(request: TextToImageRequest) -> Response:
    if not (256 <= request.height <= 1024 and request.height % 8 == 0):
        raise HTTPException(400, "Height must be 256-1024, divisible by 8")
    if not (256 <= request.width <= 1024 and request.width % 8 == 0):
        raise HTTPException(400, "Width must be 256-1024, divisible by 8")
    if not (1 <= request.num_inference_steps <= 50):
        raise HTTPException(400, "num_inference_steps must be in [1, 50]")

    try:
        image, _ = await run_inference(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, f"Image generation failed: {e}")
