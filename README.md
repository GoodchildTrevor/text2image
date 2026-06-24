# text2image

A self-hosted **text-to-image REST API** powered by [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) from Black Forest Labs. Built with FastAPI and packaged as a Docker container with full NVIDIA GPU support.

***

## Features

- **Two API modes** — a simple direct endpoint and a drop-in **OpenAI-compatible** endpoint (`/v1/images/generations`)
- **FLUX.1-schnell** model — fast, high-quality image generation via `diffusers`
- **Memory-efficient** — uses VAE slicing, tiling, and attention slicing; serializes requests via async lock to prevent CUDA conflicts
- **Lazy model loading** — model is loaded on first request (or eagerly via `PRELOAD_MODEL=1`)
- **Persistent model cache** — Hugging Face cache is mounted as a Docker volume, so the model is not re-downloaded on container restart
- **Health endpoint** — reports CUDA availability and GPU count

***

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with CUDA 12.8 support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Hugging Face account with access to `black-forest-labs/FLUX.1-schnell`

***

## Quick Start

**1. Clone the repository:**

```bash
git clone https://github.com/GoodchildTrevor/text2image.git
cd text2image
```

**2. Set your Hugging Face token:**

```bash
export HF_TOKEN=hf_your_token_here
```

Or create a `.env` file:

```env
HF_TOKEN=hf_your_token_here
```

**3. Build and run:**

```bash
docker compose up --build
```

The API will be available at `http://localhost:8010`.

***

## API Reference

### `GET /health`

Returns the service status and CUDA device info.

```json
{
  "status": "ok",
  "cuda_available": true,
  "device_count": 1
}
```

***

### `POST /generate`

Generate an image and receive it as a raw PNG binary.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | required | Text description of the image |
| `width` | `int` | `512` | Image width in pixels (256–1024, divisible by 8) |
| `height` | `int` | `512` | Image height in pixels (256–1024, divisible by 8) |
| `num_inference_steps` | `int` | `5` | Denoising steps (1–50) |
| `guidance_scale` | `float` | `1.0` | CFG guidance scale |

**Example:**

```bash
curl -X POST http://localhost:8010/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat sitting on a neon-lit rooftop, cyberpunk style", "width": 512, "height": 512}' \
  --output result.png
```

***

### `POST /v1/images/generations`

OpenAI-compatible endpoint. Drop-in replacement for `openai.images.generate`.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | required | Text description of the image |
| `model` | `string` | `"flux-schnell"` | Model name (informational) |
| `n` | `int` | `1` | Number of images (only `1` supported) |
| `size` | `string` | `"512x512"` | One of `256x256`, `512x512`, `1024x1024` |
| `response_format` | `string` | `"b64_json"` | Only `b64_json` is supported |

**Example (Python with OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8010/v1"
)

response = client.images.generate(
    model="flux-schnell",
    prompt="a futuristic city at sunset, watercolor style",
    size="512x512",
)

import base64
image_data = base64.b64decode(response.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_data)
```

***

### `GET /v1/models`

Returns the list of available models.

***

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Hugging Face API token for downloading gated model |
| `HF_HOME` | `/root/.cache/huggingface` | Path for Hugging Face model cache |
| `FLUX_DEFAULT_STEPS` | `4` | Default inference steps for the OpenAI-compat endpoint |
| `FLUX_DEFAULT_GUIDANCE` | `1.0` | Default guidance scale for the OpenAI-compat endpoint |
| `PRELOAD_MODEL` | `0` | Set to `1` to load the model at startup instead of on first request |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs to expose to the container |

***

## Project Structure

```
text2image/
├── app/
│   ├── main.py            # FastAPI app, lifespan, router registration
│   ├── config.py          # Pydantic request/response models
│   ├── service.py         # Model loading (lru_cache) and inference logic
│   └── routers/
│       ├── direct_image.py   # POST /generate → raw PNG
│       └── openai_compat.py  # POST /v1/images/generations (OpenAI-compatible)
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh          # HF login + app startup
└── requirements.txt
```

***

## Notes

- Inference requests are **serialized** with an `asyncio.Lock` to avoid concurrent CUDA memory errors.
- The model is cached with `lru_cache(maxsize=1)` — it loads once and stays in GPU memory.
- `TORCHDYNAMO_DISABLE=1` and `expandable_segments:True` are set to improve stability and VRAM efficiency.
- Logs are written to both stdout and `text2image.log`.

***

## License

This project does not include an explicit license. The FLUX.1-schnell model is subject to its own [license on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
