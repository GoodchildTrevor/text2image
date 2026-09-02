import base64
import io
import time
import logging
import os
from typing import Annotated, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError, DecompressionBombError
from app.config import (
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
    LOCAL_MODEL_ID,
    MAX_UPLOAD_BYTES,
    MAX_IMAGE_PIXELS,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
)
from app.service import generate_image, edit_image, save_image_bytes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _make_response(
    img_bytes: bytes, 
    revised_prompt: str, 
    response_format: str,
) -> ImageGenerationResponse:
    """Make a response object.
    :param img_bytes: The image bytes.
    :param revised_prompt: The revised prompt.
    :param response_format: The format of the response. Must be 'b64_json' or 'url'.
    :return: The response object.
    """
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
    """Convert raw image bytes to a data URL.
    :param raw: The raw image bytes.
    :param mime: The MIME type of the image.
    :return: The data URL.
    :raises HTTPException: If the image is not a valid image.
    """
    mime = (mime or "image/png").split(";")[0].strip()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def _normalize_image(raw: bytes) -> tuple[bytes, str]:
    """Re-encode any uploaded image to PNG so unsupported formats
    (BMP, TIFF, HEIC, ICO, GIF, ...) never reach the cloud provider as-is.
    Providers like Gemini reject e.g. image/bmp outright."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "Uploaded file is not a valid image")

    if img.mode not in ("RGB", "RGBA"):
        has_alpha = img.mode in ("P", "LA", "PA") and "transparency" in img.info
        img = img.convert("RGBA" if has_alpha else "RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest):
    """
    OpenAI-compatible generation endpoint.
    :param request: The request object.
    :return: The response.
    :raises HTTPException: If the request is invalid.
    :raises ValueError: If the request is invalid.
    :raises Exception: If the request is invalid.
    :raises httpx.HTTPStatusError: If the request is invalid.
    :raises httpx.ConnectError: If the request is invalid.
    :raises httpx.TimeoutException: If the request is invalid.
    :raises httpx.RequestException: If the request is invalid.
    :raises httpx.ResponseException: If the request is invalid.
    :raises httpx.ClientError: If the request is invalid.
    :raises httpx.ServerError: If the request is invalid.
    """
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

    try:
        img_bytes, revised_prompt = await generate_image(
            model=request.model,
            prompt=request.prompt,
            steps=DEFAULT_STEPS,
            guidance=DEFAULT_GUIDANCE,
            resolution=request.resolution,
            aspect_ratio=request.aspect_ratio,
            quality=request.quality,
            size=request.size,
        )
        return _make_response(img_bytes, revised_prompt, request.response_format)
    except ValueError as e:
        logger.exception(f"ValueError in generate_image")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Generation error")
        raise HTTPException(500, "Image generation failed")


@router.post("/images/edits", response_model=ImageGenerationResponse)
async def openai_edit(
    prompt: Annotated[str, Form()],
    # OpenWebUI sends a single image as 'image[]' (array notation), standard clients use 'image'
    image: Annotated[Optional[UploadFile], File(alias="image[]")] = None,
    image_single: Annotated[Optional[UploadFile], File(alias="image")] = None,
    model: Annotated[Optional[str], Form()] = None,
    n: Annotated[int, Form()] = 1,
    size: Annotated[Optional[str], Form()] = None,
    resolution: Annotated[Optional[str], Form()] = None,
    aspect_ratio: Annotated[Optional[str], Form()] = None,
    quality: Annotated[Optional[str], Form()] = None,
    response_format: Annotated[str, Form()] = "b64_json",
):
    """
    OpenAI-compatible edit endpoint.
    :param prompt: The prompt to edit.
    :param image: The image to edit.
    :param image_single: The image to edit.
    :param model: The model to use.
    :param n: The number of images to generate.
    :param size: The size of the image.
    :param resolution: The resolution of the image.
    :param aspect_ratio: The aspect ratio of the image.
    :param quality: The quality of the image.
    :param response_format: The format of the response. Must be 'b64_json' or 'url'.
    :return: The response.
    :raises HTTPException: If the request is invalid.
    :raises ValueError: If the request is invalid.
    :raises Exception: If the request is invalid.
    :raises httpx.HTTPStatusError: If the request is invalid.
    :raises httpx.ConnectError: If the request is invalid.
    :raises httpx.TimeoutException: If the request is invalid.
    :raises httpx.RequestException: If the request is invalid.
    :raises httpx.ResponseException: If the request is invalid.
    :raises httpx.ClientError: If the request is invalid.
    :raises httpx.ServerError: If the request is invalid.
    """
    upload = image or image_single

    logger.info(
        f"Edit: model={model!r}, size={size!r}, resolution={resolution!r}, "
        f"prompt={prompt[:80]!r}, upload={'yes' if upload else 'None'}"
    )

    if n != 1:
        raise HTTPException(400, "Only n=1 is supported")
    if response_format not in ("b64_json", "url"):
        raise HTTPException(400, "response_format must be 'b64_json' or 'url'")
    if upload is None:
        raise HTTPException(400, "Field 'image' or 'image[]' (file upload) is required for edits")

    raw = await upload.read()
    if not raw:
        raise HTTPException(400, "Uploaded image file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        logger.warning(f"Edit upload too large: {len(raw)} bytes (max={MAX_UPLOAD_BYTES})")
        raise HTTPException(413, f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes ({MAX_UPLOAD_BYTES // 1024 // 1024} MiB)")
    original_mime = upload.content_type
    raw, mime = _normalize_image(raw)
    image_b64 = _to_data_url(raw, mime)
    logger.info(f"[edit] image read: {len(raw)} bytes, original_mime={original_mime!r}, normalized_mime={mime!r}")

    if model is None:
        model = os.getenv("DEFAULT_MODEL") or LOCAL_MODEL_ID

    try:
        img_bytes, revised_prompt = await edit_image(
            model=model,
            prompt=prompt,
            image_b64=image_b64,
            size=size,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )
        return _make_response(img_bytes, revised_prompt, response_format)

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        if status == 403:
            logger.warning(f"Edit 403 for model={model!r}: {detail}")
            raise HTTPException(400, f"Model {model!r} does not support image editing on this provider.")
        logger.error(f"Edit HTTP {status} for model={model!r}: {detail}")
        raise HTTPException(400, f"Provider error {status}: {detail}")
    except ValueError as e:
        logger.exception(f"ValueError in edit_image")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Edit error")
        raise HTTPException(500, "Image edit failed")
