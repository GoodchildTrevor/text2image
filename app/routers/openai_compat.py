import base64
import time
import logging
import os
from typing import Optional
import httpx
from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject
from app.service import generate_image, edit_image, save_image_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

DEFAULT_STEPS = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))

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


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest):
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


@router.post("/images/edits")
async def openai_edit(request: Request):
    """Edit an existing image via multipart/form-data.

    Parses the form manually to support both ``image`` (standard OpenAI)
    and ``image[]`` (OpenWebUI) field names, as FastAPI's File/alias
    mechanism does not support bracket-notation field names.
    """
    try:
        form = await request.form()
    except Exception as e:
        logger.error(f"Edit: failed to parse multipart form: {e}")
        raise HTTPException(
            400,
            "Malformed multipart request — image body is empty or truncated. "
            "The image URL may have expired before the pipe could download it."
        )

    prompt = form.get("prompt")
    if not prompt:
        raise HTTPException(400, "Field 'prompt' is required")

    model = form.get("model") or os.getenv("LOCAL_MODEL", "black-forest-labs/FLUX.1-schnell")
    n = int(form.get("n", 1))
    size = form.get("size") or None
    resolution = form.get("resolution") or None
    aspect_ratio = form.get("aspect_ratio") or None
    response_format = form.get("response_format") or "b64_json"

    logger.info(
        f"Edit: model={model!r}, size={size!r}, resolution={resolution!r}, "
        f"aspect_ratio={aspect_ratio!r}, prompt={prompt[:80]!r}"
    )

    if n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if response_format not in ("b64_json", "url"):
        raise HTTPException(400, "response_format must be 'b64_json' or 'url'")

    # Find image file under any field name (image, image[], etc.)
    image_file: Optional[UploadFile] = None
    for key, val in form.multi_items():
        if isinstance(val, UploadFile):
            image_file = val
            logger.info(f"Edit: image field name={key!r}, filename={val.filename!r}")
            break

    if image_file is None:
        raise HTTPException(400, "No image file provided")

    raw = await image_file.read()
    if not raw:
        raise HTTPException(400, "Image file is empty — the URL may have expired or upload failed")

    logger.info(f"Edit: image_bytes={len(raw)} bytes")
    image_b64 = base64.b64encode(raw).decode()

    resolved_resolution, resolved_aspect_ratio, w, h = _resolve_size_params(
        size, resolution, aspect_ratio
    )

    try:
        img_bytes, revised_prompt = await edit_image(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            size=size if resolved_resolution is None else None,
            resolution=resolved_resolution,
            aspect_ratio=resolved_aspect_ratio,
        )
        return _make_response(img_bytes, revised_prompt, response_format)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        if status == 403:
            msg = (
                f"Model {model!r} does not support image editing on this provider. "
                f"Try recraft/recraft-v4 or bytedance-seed/seedream-4.5 instead."
            )
            logger.warning(f"Edit 403 for model={model!r}: {detail}")
            raise HTTPException(400, msg)
        logger.error(f"Edit HTTP {status} for model={model!r}: {detail}")
        raise HTTPException(400, f"Provider error {status}: {detail}")
    except ValueError as e:
        logger.error(f"ValueError in edit_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(500, "Image edit failed")
