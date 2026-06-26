import base64
import time
import logging
import os
from fastapi import APIRouter, HTTPException
from app.config import ImageGenerationRequest, ImageGenerationResponse, ImageObject, ImageEditRequest
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


def _resolve_size_params(
    size: str | None,
    resolution: str | None,
    aspect_ratio: str | None,
) -> tuple[str | None, str | None, int | None, int | None]:
    """Validate and resolve size/resolution/aspect_ratio into canonical values.

    Priority: resolution + aspect_ratio > size.

    :returns: (resolved_resolution, resolved_aspect_ratio, width, height)
              width/height are set only when size is used (local pipeline needs them).
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
        if size not in VALID_SIZES:
            raise HTTPException(
                400,
                f"Invalid size {size!r}. Must be one of: {sorted(VALID_SIZES)}"
            )
        try:
            w, h = map(int, size.split("x"))
        except ValueError:
            raise HTTPException(400, f"Malformed size {size!r}, expected WxH format")
        return None, aspect_ratio, w, h

    # Nothing provided — default
    default = "1024x1024"
    w, h = map(int, default.split("x"))
    return None, aspect_ratio, w, h


def _make_response(img_bytes: bytes, revised_prompt: str, response_format: str) -> ImageGenerationResponse:
    """Build ImageGenerationResponse from raw bytes.

    When ``response_format`` is ``"url"``, the image is saved to disk and
    a public URL is returned. Otherwise base64-encodes the bytes.

    :param img_bytes: Raw PNG image bytes.
    :param revised_prompt: The prompt used/revised by the provider.
    :param response_format: ``"b64_json"`` or ``"url"``.
    :returns: :class:`ImageGenerationResponse` with one item.
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


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def openai_generate(request: ImageGenerationRequest):
    """Generate an image from a text prompt via OpenAI-compatible API.

    Accepts either ``resolution`` + ``aspect_ratio`` (OpenRouter normalized
    parameters) or legacy ``size`` (pixel dimensions). When both are present,
    ``resolution`` takes priority and ``size`` is ignored.

    :param request: ImageGenerationRequest with model, prompt, and size params.
    :returns: ImageGenerationResponse with image as b64_json or url.
    :raises HTTPException 400: On invalid params or unknown model.
    :raises HTTPException 500: On unexpected generation failure.
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
async def openai_edit(request: ImageEditRequest):
    """Edit an existing image based on a text prompt via OpenAI-compatible API.

    Accepts either ``resolution`` + ``aspect_ratio`` or legacy ``size``.
    When both are present, ``resolution`` takes priority.

    :param request: ImageEditRequest with model, prompt, input image, and size params.
    :returns: ImageGenerationResponse with image as b64_json or url.
    :raises HTTPException 400: On invalid params or unknown model.
    :raises HTTPException 500: On unexpected edit failure.
    """
    logger.info(
        f"Edit: model={request.model!r}, size={request.size!r}, "
        f"resolution={request.resolution!r}, aspect_ratio={request.aspect_ratio!r}, "
        f"prompt={request.prompt[:80]!r}"
    )

    if request.n != 1:
        raise HTTPException(400, "Only n=1 is supported")

    resolved_resolution, resolved_aspect_ratio, w, h = _resolve_size_params(
        request.size, request.resolution, request.aspect_ratio
    )

    try:
        img_bytes, revised_prompt = await edit_image(
            model=request.model,
            prompt=request.prompt,
            image_b64=request.image,
            resolution=resolved_resolution,
            aspect_ratio=resolved_aspect_ratio,
            size=request.size if resolved_resolution is None else None,
        )
        return _make_response(img_bytes, revised_prompt, request.response_format)
    except ValueError as e:
        logger.error(f"ValueError in edit_image: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(500, "Image edit failed")
