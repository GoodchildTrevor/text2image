import os
import pathlib
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.routers.direct_image import router as direct_router
from app.routers.openai_compat import router as openai_router
from app.service import get_pipeline
from app import openrouter_caps

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

logger = logging.getLogger(__name__)

IMAGES_DIR = pathlib.Path("/app/static/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: fetch live OpenRouter model capabilities, then optionally warm up local pipeline."""
    await openrouter_caps.refresh_caps()
    if os.getenv("PRELOAD_MODEL", "0") == "1":
        get_pipeline()
    yield


app = FastAPI(title="imgen API", lifespan=lifespan)


@app.middleware("http")
async def log_edits_request(request: Request, call_next):
    """Temporarily log raw body of /v1/images/edits for debugging."""
    if request.url.path == "/v1/images/edits":
        body = await request.body()
        ct = request.headers.get("content-type", "<none>")
        logger.info(f"[DEBUG edits] content-type: {ct}")
        logger.info(f"[DEBUG edits] body[:1000]: {body[:1000]}")
    return await call_next(request)


# Serve generated images: GET /images/{filename}
app.mount("/images", StaticFiles(directory="/app/static/images"), name="images")

app.include_router(direct_router)   # POST /generate
app.include_router(openai_router)   # POST /v1/images/generations, /v1/images/edits


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
