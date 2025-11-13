from pydantic import BaseModel
from typing import Optional


class TextToImageRequest(BaseModel):
    """Request model for text-to-image generation."""
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 5
    guidance_scale: Optional[float] = 1.0