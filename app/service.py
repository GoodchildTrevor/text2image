import asyncio
import time
import torch
from functools import lru_cache
from diffusers import FluxPipeline
from PIL import Image
import logging

logger = logging.getLogger(__name__)

_inference_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def get_pipeline() -> FluxPipeline:
    model_id = "black-forest-labs/FLUX.1-schnell"
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
    """Run FLUX inference. Serialized via asyncio.Lock to prevent concurrent CUDA errors."""
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
