import asyncio
import os
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
    width: int,
    height: int,
    steps: int = None,
    guidance: float = None,
) -> tuple[bytes, str]:
    """Generate an image from a text prompt using the specified provider.

    Routes to a local diffusion pipeline (when ``provider is None``) or
    to a cloud provider (e.g. RouterAI). Returns PNG bytes and the final
    prompt used.

    :param model: The model identifier to look up in the provider registry.
    :param prompt: The text prompt describing the desired image.
    :param width: The requested image width in pixels.
    :param height: The requested image height in pixels.
    :param steps: Number of inference steps (falls back to FLUX_DEFAULT_STEPS).
    :param guidance: Guidance scale (falls back to FLUX_DEFAULT_GUIDANCE).
    :returns: A tuple of (PNG image bytes, prompt used for generation).
    :raises ValueError: If ``model`` is not registered in the provider registry.
    """
    if model not in PROVIDERS:
        raise ValueError(f"Unknown model '{model}'")
    provider = PROVIDERS[model]

    if provider is None:
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
        )


async def edit_image(
    model: str,
    prompt: str,
    image_b64: str,
) -> tuple[bytes, str]:
    """Edit an existing image using the specified provider.

    Only cloud providers support image editing; the local pipeline
    will raise ``ValueError``.

    :param model: The model identifier to look up in the provider registry.
    :param prompt: The text prompt describing the desired edit.
    :param image_b64: Base64-encoded input image.
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
    )
