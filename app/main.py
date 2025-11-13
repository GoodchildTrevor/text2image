import os
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from io import BytesIO
from diffusers import FluxPipeline
import logging

from app.config import TextToImageRequest

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

LOG_PATH = "summarizer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_pipeline(model_name: str = "FLUX.1-schnell") -> FluxPipeline:
    """
    Load and cache the FluxPipeline model.
    :param model_name 
    :return: FluxPipeline
    :raise:RuntimeError if model fails to load.
    """
    model_id = "black-forest-labs/FLUX.1-schnell"

    try:
        load_start = time.perf_counter()
        logger.info(f"✅ Model '{model_name}' start loading")
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
        load_time = time.perf_counter() - load_start
        logger.info(f"✅ Model '{model_name}' loaded in {load_time:.2f} sec")
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

app = FastAPI(title="FLUX.1-schnell Text-to-Image API")

@app.post("/generate", response_class=Response, responses={
    200: {
        "content": {"image/png": {}},
        "description": "Generated image in PNG format.",
    },
    400: {"description": "Invalid input."},
    500: {"description": "Model inference failed."},
})
async def generate_image(request: TextToImageRequest) -> Response:
    """
    Generate an image from a text prompt using FLUX.1-schnell.
    :param request : TextToImageRequest
    Input request with prompt and generation parameters.
    :return: Response. PNG image as binary response.
    :raises: HTTPException. If generation fails or inputs are invalid.
    """
    # Validate dimensions (optional but recommended)
    if not (256 <= request.height <= 1024 and request.height % 8 == 0):
        raise HTTPException(status_code=400, detail="Height must be between 256–1024 and divisible by 8.")
    if not (256 <= request.width <= 1024 and request.width % 8 == 0):
        raise HTTPException(status_code=400, detail="Width must be between 256–1024 and divisible by 8.")
    if request.num_inference_steps < 1 or request.num_inference_steps > 50:
        raise HTTPException(status_code=400, detail="num_inference_steps must be in [1, 50].")

    try:
        pipe = get_pipeline()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        with torch.no_grad():
            output = pipe(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
        image: Image.Image = output.images[0]

        inference_time = time.perf_counter() - start_time
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(
            f"Inference time: {inference_time:.2f}s | 📈 Peak VRAM: {peak_mem_gb:.2f} GB"
        )

        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return Response(content=img_bytes.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.get("/health")
def health_check() -> dict:
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
