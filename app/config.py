import os
from typing import Optional
from pydantic import BaseModel, Field


# ── Application-wide constants (single source of truth) ───────────────

DEFAULT_STEPS: int = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE: float = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))
LOCAL_MODEL_ID: str = os.getenv("LOCAL_MODEL", "black-forest-labs/FLUX.1-schnell")

IMAGES_DIR_STR: str = "/app/static/images"

MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "25_000_000"))

DEFAULT_MAX_STORED_IMAGES: int = int(os.getenv("MAX_STORED_IMAGES", "500"))
MAX_EDIT_IMAGES: int = int(os.getenv("MAX_EDIT_IMAGES", "10"))


class TextToImageRequest(BaseModel):
    """Request model for the legacy POST /generate endpoint."""
    model: str = Field(
        default_factory=lambda: os.getenv("LOCAL_MODEL", "black-forest-labs/FLUX.1-schnell")
    )
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = Field(
        default_factory=lambda: int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
    )
    guidance_scale: float = Field(
        default_factory=lambda: float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))
    )


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
    image: Optional[str] = None        # base64-encoded input image (standard OpenAI format)
    image_urls: Optional[list[str]] = None  # OpenWebUI format: internal file paths or URLs
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
