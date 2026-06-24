from fastapi import APIRouter, HTTPException, Response
from app.config import TextToImageRequest
from app.service import generate_image
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_class=Response, responses={
    200: {"content": {"image/png": {}}},
})
async def generate_image_endpoint(request: TextToImageRequest) -> Response:
    """Generate an image from a text prompt using the specified model.

    Validates height and width parameters (must be between 256 and 1024,
    divisible by 8), then calls the image generation service. Returns
    a PNG image response.

    :param request: The image generation request containing model, prompt,
        dimensions, and optional tuning parameters (num_inference_steps,
        guidance_scale).
    :returns: A Response with the generated PNG image on success.
    :raises HTTPException 400: If dimensions are invalid or generation
        encounters a validation error.
    :raises HTTPException 500: If an unexpected error occurs during
        image generation.
    """
    if not (256 <= request.height <= 1024 and request.height % 8 == 0):
        raise HTTPException(400, "Height must be 256-1024, divisible by 8")
    if not (256 <= request.width <= 1024 and request.width % 8 == 0):
        raise HTTPException(400, "Width must be 256-1024, divisible by 8")

    try:
        img_bytes, revised = await generate_image(
            model=request.model,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            steps=request.num_inference_steps,
            guidance=request.guidance_scale,
        )
        if revised != request.prompt:
            logger.info(f"Prompt revised: {revised!r}")
        return Response(content=img_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, "Image generation failed")
    
