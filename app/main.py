import os, torch, logging
from fastapi import FastAPI
from app.routers import direct_image, openai_compat

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("text2image.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

app = FastAPI(title="FLUX.1-schnell Text-to-Image API")

app.include_router(direct_image.router)
app.include_router(openai_compat.router)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }