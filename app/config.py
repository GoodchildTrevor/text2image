import os
from typing import Optional
from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    model: str = Field(
        default_factory=lambda: os.getenv("LOCAL_MODEL", "black-forest-labs/FLUX.1-schnell")
    )
    prompt: str
    n: int = 1
    size: Optional[str] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    quality: Optional[str] = None
    response_format: str = "b64_json"


class ImageEditRequest(BaseModel):
    model: str = Field(
        default_factory=lambda: os.getenv("LOCAL_MODEL", "black-forest-labs/FLUX.1-schnell")
    )
    prompt: str
    image: str  # base64-encoded input image
    n: int = 1
    size: Optional[str] = None
    resolution: Optional[str] = None
    aspect_ratio: Optional[str] = None
    response_format: str = "b64_json"


class ImageObject(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageObject]
