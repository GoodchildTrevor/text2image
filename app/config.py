from pydantic import BaseModel
from typing import Optional, List


class TextToImageRequest(BaseModel):
    """Request model for text-to-image generation."""
    model: Optional[str] = "flux-schnell"
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 5
    guidance_scale: Optional[float] = 1.0


class ImageEditRequest(BaseModel):
    prompt: str
    model: Optional[str] = "google/gemini-3.1-flash-image-preview"
    image: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "flux-schnell"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    quality: Optional[str] = "standard"
    style: Optional[str] = None
    user: Optional[str] = None


class ImageObject(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageObject]
