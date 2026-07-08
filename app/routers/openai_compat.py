import base64
import time
import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.config import (
    ImageGenerationRequest,
    ImageEditRequest,
    ImageGenerationResponse,
    ImageObject,
)
from app.service import generate_image, edit_image, save_image_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

DEFAULT_STEPS = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))

_default_sizes = "1024x1024,864x1184,1184x864,768x1344,1344x768"
VALID_SIZES = set(
    s.strip()
    for s in os.getenv("VALID_SIZES", _default_sizes).split(",")
    if s.strip()
)

VALID_RESOLUTIONS = {"512", "1K", "2K", "4K"}

# Field names OpenWebUI may use for the image file
_IMAGE_FIELD_CANDIDATES = ("image", "image[]", "image_url", "file")


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


def _to_data_url(raw: bytes, mime: str | None) -> str:
    mime = (mime or "image/png").split(";")[0].strip()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


async def _parse_edit_request(http_request: Request) -> ImageEditRequest:
    """Parse /v1/images/edits body as JSON or multipart/form-data.

    Handles OpenWebUI quirk: image file is sent as 'image[]' (array notation),
    not 'image'. We try all known field name variants.
    """
    content_type = (http_request.headers.get("content-type") or "").lower()
    logger.info(f"[edit] content-type: {content_type!r}")

    try:
        if "application/json" in content_type:
            payload = await http_request.json()
            return ImageEditRequest.model_validate(payload)

        if "multipart/form-data" in content_type:
            form = await http_request.form()
            logger.info(f"[edit] multipart form keys: {list(form.keys())}")

            # Find image among known field name variants (image, image[], image_url, file)
            image_value: str | None = None
            for field_name in _IMAGE_FIELD_CANDIDATES:
                image_part = form.get(field_name)
                if image_part is None:
                    continue
                if isinstance(image_part, UploadFile):
                    raw = await image_part.read()
                    if raw:
                        image_value = _to_data_url(raw, image_part.content_type)
                        logger.info(f"[edit] image from field {field_name!r}: {len(raw)} bytes")
                    break
                elif isinstance(image_part, str) and image_part:
                    image_value = image_part
                    logger.info(f"[edit] image from field {field_name!r}: string len={len(image_part)}")
                    break

            raw_payload: dict = {
                "prompt": form.get("prompt"),
                "image": image_value,
                "n": int(form.get("n") or 1),
                "response_format": form.get("response_format") or "b64_json",
            }
            for field in ("model", "size", "resolution", "aspect_ratio", "quality"):
                val = form.get(field)
                if val is not None:
                    raw_payload[field] = val

            return ImageEditRequest.model_validate(raw_payload)

        raise HTTPException(
            415,
            f"Unsupported Content-Type for /v1/images/edits: {content_type!r}. "
            "Use 'application/json' or 'multipart/form-data'."
        )

    except HTTPException:
        raise
    except ValidationError as e:
        raise RequestValidationError(e.errors()) from e


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


@router.post("/images/edits", response_model=ImageGenerationResponse)
async def openai_edit(http_request: Request):
    request = await _parse_edit_request(http_request)

    img_info = f"<b64 len={len(request.image)}>" if request.image else "None"
    logger.info(
        f"Edit: model={request.model!r}, size={request.size!r}, "
        f"resolution={request.resolution!r}, aspect_ratio={request.aspect_ratio!r}, "
        f"prompt={request.prompt[:80]!r}, image={img_info}"
    )

    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if request.response_format not in ("b64_json", "url"):
        raise HTTPException(400, "response_format must be 'b64_json' or 'url'")
    if not request.image:
        raise HTTPException(400, "Field 'image' (file upload or base64) is required for edits")

    resolved_resolution, resolved_aspect_ratio, w, h = _resolve_size_params(
        request.size, request.resolution, request.aspect_ratio
    )

    try:
        img_bytes, revised_prompt = await edit_image(
            model=request.model,
            prompt=request.prompt,
            image_b64=request.image,
            size=request.size if resolved_resolution is None else None,
            resolution=resolved_resolution,
            aspect_ratio=resolved_aspect_ratio,
        )
        return _make_response(img_bytes, revised_prompt, request.response_format)

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        if status == 403:
            msg = (
                f"Model {request.model!r} does not support image editing on this provider. "
                f"Try recraft/recraft-v4 or bytedance-seed/seedream-4.5 instead."
            )
            logger.warning(f"Edit 403 for model={request.model!r}: {detail}")
            raise HTTPException(400, msg)
        logger.error(f"Edit HTTP {status} for model={request.model!r}: {detail}")
        raise HTTPException(400, f"Provider error {status}: {detail}")
    except ValueError as e:
        logger.error(f"ValueError in edit_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(500, "Image edit failed")
