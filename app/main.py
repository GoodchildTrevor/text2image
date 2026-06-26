import os
import pathlib
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers.direct_image import router as direct_router
from app.routers.openai_compat import router as openai_router
from app.service import get_pipeline

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("imgen.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

IMAGES_DIR = pathlib.Path("/app/static/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optional model warmup on startup. Enable via PRELOAD_MODEL=1."""
    if os.getenv("PRELOAD_MODEL", "0") == "1":
        get_pipeline()
    yield


app = FastAPI(title="imgen API", lifespan=lifespan)

# Serve generated images: GET /images/{filename}
app.mount("/images", StaticFiles(directory="/app/static/images"), name="images")

app.include_router(direct_router)   # POST /generate
app.include_router(openai_router)   # POST /v1/images/generations


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
