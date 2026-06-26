import asyncio
import os
import uuid
import pathlib
from io import BytesIO
import time
import torch
from functools import lru_cache
from diffusers import FluxPipeline
from PIL import Image
import logging

from app.providers import PROVIDERS

logger = logging.getLogger(__name__)

_inference_lock = asyncio.Lock()

LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL", "black-forest-labs/FLUX.1-schnell")
_LOCAL_DEFAULT_STEPS = int(os.getenv("FLUX_DEFAULT_STEPS", "4"))
_LOCAL_DEFAULT_GUIDANCE = float(os.getenv("FLUX_DEFAULT_GUIDANCE", "1.0"))

IMAGES_DIR = pathlib.Path("/app/static/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Public base URL for serving static images
# e.g. "http://imgen:8010" or "https://imgen.example.com"
PUBLIC_BASE_URL = os.getenv("IMGEN_PUBLIC_URL", "http://imgen:8010")

# Mapping from OpenRouter resolution tier to pixel dimensions for local pipeline
_RESOLUTION_TO_PIXELS: dict[str, tuple[int, int]] = {
    "512": (512, 512),
    "1K":  (1024, 1024),
    "2K":  (2048, 2048),
    "4K":  (4096, 4096),
}


def save_image_bytes(img_bytes: bytes) -> str:
    """Save PNG bytes to static/images and return public URL.

    :param img_bytes: Raw PNG image bytes.
    :returns: Public URL string like ``http://imgen:8010/static/images/abc123.png``.
    """
    fname = f"{uuid.uuid4().hex}.png"
    (IMAGES_DIR / fname).write_bytes(img_bytes)
    return f"{PUBLIC_BASE_URL}/static/images/{fname}"


@lru_cache(maxsize=1)
def get_pipeline() -> FluxPipeline:
    """Load and cache the local diffusion pipeline.

    The model is taken from the ``LOCAL_MODEL`` environment variable
    (default: ``black-forest-labs/FLUX.1-schnell``). The pipeline is
    loaded once and cached via ``@lru_cache``. Applies VRAM-saving
    optimizations (slicing, tiling, attention slicing).

    :returns: A configured :class:`FluxPipeline` instance.
    """
    model_id = LOCAL_MODEL_ID
    logger.info(f"Loading model '{model_id}'...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing(1)
    logger.info("Model loaded and cached.")
    return pipe


async def run_inference(
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> tuple[Image.Image, float]:
    """Run local diffusion pipeline inference.

    Serialized via :data:`_inference_lock` to prevent concurrent CUDA
    errors. Logs elapsed time and peak VRAM usage after each run.

    :param prompt: The text prompt describing the desired image.
    :param width: The requested image width in pixels.
    :param height: The requested image height in pixels.
    :param num_inference_steps: Number of denoising steps.
    :param guidance_scale: Classifier-free guidance scale.
    :returns: A tuple of (generated PIL Image, elapsed seconds).
    """
    async with _inference_lock:
        pipe = get_pipeline()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        elapsed = time.perf_counter() - start
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"Inference: {elapsed:.2f}s | Peak VRAM: {peak_gb:.2f} GB")
        return output.images[0], elapsed


async def generate_image(
    model: str,
    prompt: str,
    width: int | None = None,
    height: int | None = None,
    steps: int = None,
    guidance: float = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    **kwargs,
) -> tuple[bytes, str]:
    """Generate an image from a text prompt using the specified provider.

    Routes to a local diffusion pipeline (when ``provider is None``) or
    to a cloud provider. Returns PNG bytes and the final prompt used.

    For the local pipeline, dimensions are resolved in this order:
    1. Explicit ``width`` + ``height``
    2. ``resolution`` tier mapped via ``_RESOLUTION_TO_PIXELS``
    3. Default 1024x1024

    For cloud providers, ``resolution``, ``aspect_ratio``, and any extra
    ``**kwargs`` (e.g. ``size``, ``quality``) are forwarded as-is.

    :param model: The model identifier to look up in the provider registry.
    :param prompt: The text prompt describing the desired image.
    :param width: Pixel width — used by local pipeline only.
    :param height: Pixel height — used by local pipeline only.
    :param steps: Inference steps (local pipeline only).
    :param guidance: Guidance scale (local pipeline only).
    :param resolution: OpenRouter resolution tier (``"512"``, ``"1K"``, ``"2K"``, ``"4K"``).
    :param aspect_ratio: OpenRouter aspect ratio string (``"1:1"``, ``"16:9"``, ...).
    :returns: A tuple of (PNG image bytes, prompt used for generation).
    :raises ValueError: If ``model`` is not registered in the provider registry.
    """
    if model not in PROVIDERS:
        raise ValueError(f"Unknown model '{model}'")
    provider = PROVIDERS[model]

    if provider is None:
        # Resolve pixel dimensions for local pipeline
        if not (width and height) and resolution:
            width, height = _RESOLUTION_TO_PIXELS.get(resolution, (1024, 1024))
        width = width or 1024
        height = height or 1024

        image, _ = await run_inference(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps or _LOCAL_DEFAULT_STEPS,
            guidance_scale=guidance or _LOCAL_DEFAULT_GUIDANCE,
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), prompt
    else:
        return await provider.generate(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            **kwargs,
        )


async def edit_image(
    model: str,
    prompt: str,
    image_b64: str,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    **kwargs,
) -> tuple[bytes, str]:
    """Edit an existing image using the specified provider.

    Only cloud providers support image editing; the local pipeline
    will raise ``ValueError``.

    :param model: The model identifier to look up in the provider registry.
    :param prompt: The text prompt describing the desired edit.
    :param image_b64: Base64-encoded input image.
    :param resolution: OpenRouter resolution tier.
    :param aspect_ratio: OpenRouter aspect ratio string.
    :returns: A tuple of (edited PNG image bytes, revised prompt).
    :raises ValueError: If ``model`` is unknown or does not support editing.
    """
    if model not in PROVIDERS:
        raise ValueError(f"Unknown model '{model}'")
    provider = PROVIDERS[model]
    if provider is None:
        raise ValueError(f"Model '{model}' does not support image editing")
    return await provider.edit(
        model=model,
        prompt=prompt,
        image_b64=image_b64,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        **kwargs,
    )
