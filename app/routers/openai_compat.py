import base64
import time
import logging
import os
from typing import Optional
import httpx
from fastapi import APIRouter, HTTPException, Header
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject, ImageEditRequest
from app.service import generate_image, edit_image, save_image_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

DEFAULT_STEPS = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))

# Base URL of the OpenWebUI instance, used to resolve internal file paths like
# /api/v1/files/<id>/content that OpenWebUI sends in image_urls.
OWUI_BASE_URL = os.getenv("OWUI_BASE_URL", "http://open-webui:8080")

_default_sizes = "1024x1024,864x1184,1184x864,768x1344,1344x768"
VALID_SIZES = set(
    s.strip()
    for s in os.getenv("VALID_SIZES", _default_sizes).split(",")
    if s.strip()
)

VALID_RESOLUTIONS = {"512", "1K", "2K", "4K"}


def _resolve_size_params(
    size: str | None,
    resolution: str | None,
    aspect_ratio: str | None,
) -> tuple[str | None, str | None, int | None, int | None]:
    """Validate and resolve size/resolution/aspect_ratio into canonical values.

    Priority: resolution > size-as-resolution > size-as-pixels.

    :returns: (resolved_resolution, resolved_aspect_ratio, width, height)
    :raises HTTPException 400: on invalid resolution or size value.
    """
    if resolution is not None:
        if resolution not in VALID_RESOLUTIONS:
            raise HTTPException(
                400,
                f"Invalid resolution {resolution!r}. Must be one of: {sorted(VALID_RESOLUTIONS)}"
            )
        return resolution, aspect_ratio, None, None

    if size is not None:
        if size in VALID_RESOLUTIONS:
            logger.info(f"size={size!r} promoted to resolution tier")
            return size, aspect_ratio, None, None

        if size not in VALID_SIZES:
            raise HTTPException(
                400,
                f"Invalid size {size!r}. "
                f"Must be one of: {sorted(VALID_SIZES)} or resolution tier: {sorted(VALID_RESOLUTIONS)}"
            )
        try:
            w, h = map(int, size.split("x"))
        except ValueError:
            raise HTTPException(400, f"Malformed size {size!r}, expected WxH format")
        return None, aspect_ratio, w, h

    return None, aspect_ratio, 1024, 1024


def _make_response(img_bytes: bytes, revised_prompt: str, response_format: str) -> ImageGenerationResponse:
    """Build ImageGenerationResponse from raw bytes."""
    if response_format == "url":
        url = save_image_bytes(img_bytes)
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[ImageObject(url=url, revised_prompt=revised_prompt)]
        )
    b64 = base64.b64encode(img_bytes).decode()
    return ImageGenerationResponse(
        created=int(time.time()),
        data=[ImageObject(b64_json=b64, revised_prompt=revised_prompt)]
    )


async def _resolve_image_b64(request: ImageEditRequest, authorization: Optional[str]) -> str:
    """Resolve image to base64 from either `image` or `image_urls`.

    1. ``request.image`` — base64 string, use as-is.
    2. ``request.image_urls[0]`` starts with ``/`` — internal OWUI path,
       prepend OWUI_BASE_URL and fetch with forwarded Authorization header.
    3. ``request.image_urls[0]`` — absolute URL, fetch directly.

    :raises HTTPException 400: if neither field provided or fetch fails.
    """
    if request.image:
        return request.image

    if request.image_urls:
        url = request.image_urls[0]
        if url.startswith("/"):
            url = f"{OWUI_BASE_URL.rstrip('/')}{url}"
            logger.info(f"Resolving OWUI internal image path -> {url}")
        else:
            logger.info(f"Fetching image from URL: {url}")

        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return base64.b64encode(resp.content).decode()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch image from {url}: HTTP {e.response.status_code}")
            raise HTTPException(400, f"Could not fetch image from {url}: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Failed to fetch image from {url}: {e}")
            raise HTTPException(400, f"Could not fetch image: {e}")

    raise HTTPException(400, "Either 'image' or 'image_urls' must be provided")


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest):
    """Generate an image from a text prompt via OpenAI-compatible API."""
    logger.info(
        f"Received: model={request.model!r}, size={request.size!r}, "
        f"resolution={request.resolution!r}, aspect_ratio={request.aspect_ratio!r}, "
        f"n={request.n}, response_format={request.response_format!r}, "
        f"prompt={request.prompt[:80]!r}"
    )

    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if request.response_format not in ("b64_json", "url"):
        raise HTTPException(400, "response_format must be 'b64_json' or 'url'")

    resolved_resolution, resolved_aspect_ratio, w, h = _resolve_size_params(
        request.size, request.resolution, request.aspect_ratio
    )

    try:
        img_bytes, revised_prompt = await generate_image(
            model=request.model,
            prompt=request.prompt,
            width=w,
            height=h,
            steps=DEFAULT_STEPS,
            guidance=DEFAULT_GUIDANCE,
            resolution=resolved_resolution,
            aspect_ratio=resolved_aspect_ratio,
            quality=request.quality,
            size=request.size if resolved_resolution is None else None,
        )
        return _make_response(img_bytes, revised_prompt, request.response_format)
    except ValueError as e:
        logger.error(f"ValueError in generate_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, "Image generation failed")


@router.post("/images/edits", response_model=ImageGenerationResponse)
async def openai_edit(
    request: ImageEditRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Edit an existing image based on a text prompt via OpenAI-compatible API.

    Accepts JSON body with either:
    - ``image``: base64-encoded image string (standard OpenAI format)
    - ``image_urls``: list of URLs/paths (OpenWebUI format)

    Internal OpenWebUI paths (``/api/v1/files/.../content``) are resolved
    against ``OWUI_BASE_URL`` env var with the forwarded Authorization header.
    """
    logger.info(
        f"Edit: model={request.model!r}, size={request.size!r}, "
        f"resolution={request.resolution!r}, aspect_ratio={request.aspect_ratio!r}, "
        f"image_urls={request.image_urls!r}, has_image={bool(request.image)}, "
        f"prompt={request.prompt[:80]!r}"
    )

    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if request.response_format not in ("b64_json", "url"):
        raise HTTPException(400, "response_format must be 'b64_json' or 'url'")

    image_b64 = await _resolve_image_b64(request, authorization)

    resolved_resolution, resolved_aspect_ratio, w, h = _resolve_size_params(
        request.size, request.resolution, request.aspect_ratio
    )

    try:
        img_bytes, revised_prompt = await edit_image(
            model=request.model,
            prompt=request.prompt,
            image_b64=image_b64,
            size=request.size if resolved_resolution is None else None,
            resolution=resolved_resolution,
            aspect_ratio=resolved_aspect_ratio,
        )
        return _make_response(img_bytes, revised_prompt, request.response_format)
    except ValueError as e:
        logger.error(f"ValueError in edit_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(500, "Image edit failed")
