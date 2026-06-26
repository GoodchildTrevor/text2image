from pydantic import BaseModel
import os
from typing import Optional


class TextToImageRequest(BaseModel):
    """Request model for text-to-image generation (direct/internal API)."""
    model: Optional[str] = os.getenv("LOCAL_MODEL")
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 5
    guidance_scale: Optional[float] = 1.0


class ImageEditRequest(BaseModel):
    """Request model for OpenAI-compatible image edits endpoint."""
    prompt: str
    model: Optional[str] = os.getenv("CLOUD_MODEL")
    image: str
    n: Optional[int] = 1
    # Legacy pixel size — validated against VALID_SIZES whitelist.
    size: Optional[str] = "1024x1024"
    # OpenRouter normalized parameters — take priority over size.
    resolution: Optional[str] = None   # "512" | "1K" | "2K" | "4K"
    aspect_ratio: Optional[str] = None # "1:1" | "16:9" | "9:16" | ...
    response_format: Optional[str] = "b64_json"


class ImageGenerationRequest(BaseModel):
    """Request model for OpenAI-compatible image generations endpoint."""
    prompt: str
    model: Optional[str] = os.getenv("LOCAL_MODEL")
    n: Optional[int] = 1
    # Legacy pixel size — validated against VALID_SIZES whitelist.
    size: Optional[str] = "1024x1024"
    # OpenRouter normalized parameters — take priority over size.
    resolution: Optional[str] = None   # "512" | "1K" | "2K" | "4K"
    aspect_ratio: Optional[str] = None # "1:1" | "16:9" | "9:16" | ...
    response_format: Optional[str] = "b64_json"
    quality: Optional[str] = None
    style: Optional[str] = None
    user: Optional[str] = None


class ImageObject(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageObject]
