import os
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers.direct_image import router as direct_router
from app.routers.openai_compat import router as openai_router
from app.service import get_pipeline

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optional model warmup on startup. Enable via PRELOAD_MODEL=1."""
    if os.getenv("PRELOAD_MODEL", "0") == "1":
        get_pipeline()
    yield


app = FastAPI(title="Text-to-Image API", lifespan=lifespan)

app.include_router(direct_router)    # POST /generate
app.include_router(openai_router)    # POST /v1/images/generations


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
