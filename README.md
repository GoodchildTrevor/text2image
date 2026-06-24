# imgen

A self-hosted **image generation API** with an OpenAI-compatible interface. Routes requests to a local [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) diffusion pipeline or to any OpenAI-compatible cloud provider — all controlled through environment variables, no code changes needed.

Built with FastAPI. Packaged as a Docker container with NVIDIA GPU support.

---

## Features

- **OpenAI-compatible API** — drop-in replacement for `openai.images.generate` and `openai.images.edit`
- **Multi-provider routing** — local FLUX pipeline or cloud models selected per-request by `model` field
- **Any OpenAI-compatible backend** — point `CLOUD_API_BASE_URL` at any provider (OpenAI, self-hosted, proxy, etc.)
- **Zero hardcode** — all model names, sizes, and inference defaults are configured via env vars
- **Memory-efficient local inference** — VAE slicing/tiling, attention slicing, async lock to prevent concurrent CUDA errors
- **Lazy or eager model loading** — loads on first request, or at startup with `PRELOAD_MODEL=true`
- **Persistent model cache** — HuggingFace cache mounted as Docker volume, no re-downloads on restart
- **Image editing** — `POST /v1/images/edits` for image-to-image via cloud providers
- **Health endpoint** — reports CUDA availability and GPU count

---

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with CUDA 12.8 support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- HuggingFace account with access to `black-forest-labs/FLUX.1-schnell` *(local mode only)*

---

## Quick Start

**1. Clone the repository:**

```bash
git clone https://github.com/GoodchildTrevor/imgen.git
cd imgen
```

**2. Create your `.env` file:**

```bash
cp .env.example .env
```

Edit `.env` — at minimum set `HF_TOKEN` for local mode, or `API_KEY` + `CLOUD_API_BASE_URL` + `CLOUD_MODELS` for cloud mode.

**3. Build and run:**

```bash
docker compose up --build
```

The API will be available at `http://localhost:8010`.

---

## Configuration

All configuration is done via environment variables. See [`.env.example`](.env.example) for the full list with descriptions.

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token for downloading gated models (local mode) |
| `LOCAL_MODEL` | `black-forest-labs/FLUX.1-schnell` | HuggingFace model ID for the local diffusion pipeline |
| `LOCAL_MODELS` | value of `LOCAL_MODEL` | Comma-separated short names that route to the local pipeline |
| `API_KEY` | — | Bearer token for the cloud provider |
| `CLOUD_API_BASE_URL` | — | Base URL of the OpenAI-compatible cloud endpoint |
| `CLOUD_MODELS` | — | Comma-separated cloud model IDs routed to the cloud provider |
| `FLUX_DEFAULT_STEPS` | `4` | Default denoising steps for local inference |
| `FLUX_DEFAULT_GUIDANCE` | `1.0` | Default guidance scale for local inference |
| `VALID_SIZES` | `1024x1024,864x1184,...` | Comma-separated allowed output sizes (`WxH`) |
| `PRELOAD_MODEL` | `false` | Set to `true` to load the local model at container startup |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs to expose to the container |

### Example: local-only setup

```env
HF_TOKEN=hf_your_token_here
LOCAL_MODEL=black-forest-labs/FLUX.1-schnell
LOCAL_MODELS=flux-schnell
```

### Example: cloud-only setup

```env
API_KEY=your_api_key
CLOUD_API_BASE_URL=https://api.openai.com/v1
CLOUD_MODELS=gpt-image-1
```

### Example: mixed setup

```env
HF_TOKEN=hf_your_token_here
LOCAL_MODEL=black-forest-labs/FLUX.1-schnell
LOCAL_MODELS=flux-schnell
API_KEY=your_api_key
CLOUD_API_BASE_URL=https://your-proxy.example.com/api/v1
CLOUD_MODELS=gpt-image-1,google/gemini-2.0-flash-image-preview
```

---

## API Reference

### `GET /health`

Returns service status and CUDA device info.

```json
{
  "status": "ok",
  "cuda_available": true,
  "device_count": 1
}
```

---

### `POST /generate`

Generate an image and receive it as a raw PNG binary.

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | required | Text description of the image |
| `model` | `string` | `LOCAL_MODEL` env | Model name to use |
| `width` | `int` | `512` | Image width in pixels |
| `height` | `int` | `512` | Image height in pixels |
| `num_inference_steps` | `int` | `5` | Denoising steps |
| `guidance_scale` | `float` | `1.0` | CFG guidance scale |

```bash
curl -X POST http://localhost:8010/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat on a neon-lit rooftop, cyberpunk style", "width": 1024, "height": 1024}' \
  --output result.png
```

---

### `POST /v1/images/generations`

OpenAI-compatible image generation. Drop-in replacement for `openai.images.generate`.

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | required | Text description of the image |
| `model` | `string` | `LOCAL_MODEL` env | Model name — routes to local or cloud provider |
| `n` | `int` | `1` | Number of images (only `1` supported) |
| `size` | `string` | `1024x1024` | Output size — must be in `VALID_SIZES` |
| `response_format` | `string` | `b64_json` | Only `b64_json` supported |

```python
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://localhost:8010/v1")

# Local model
response = client.images.generate(
    model="flux-schnell",
    prompt="a futuristic city at sunset, watercolor style",
    size="1024x1024",
)

# Cloud model
response = client.images.generate(
    model="gpt-image-1",
    prompt="a futuristic city at sunset, watercolor style",
    size="1024x1024",
)

import base64
with open("output.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

---

### `POST /v1/images/edits`

OpenAI-compatible image editing. Requires a cloud model — local pipeline does not support editing.

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | `string` | required | Text description of the desired edit |
| `model` | `string` | `CLOUD_MODEL` env | Must be a registered cloud model |
| `image` | `string` | required | Base64-encoded input image |
| `n` | `int` | `1` | Number of images (only `1` supported) |
| `size` | `string` | `1024x1024` | Output size — must be in `VALID_SIZES` |

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://localhost:8010/v1")

with open("input.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.images.edit(
    model="gpt-image-1",
    image=image_b64,
    prompt="make it look like a painting",
)

with open("edited.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

---

### `GET /v1/models`

Returns the list of all registered models (local + cloud).

---

## Project Structure

```
imgen/
├── app/
│   ├── main.py            # FastAPI app, lifespan, router registration
│   ├── config.py          # Pydantic request/response models
│   ├── providers.py       # Provider registry — local pipeline and cloud
│   ├── service.py         # Inference logic and provider routing
│   └── routers/
│       ├── direct_image.py   # POST /generate → raw PNG
│       └── openai_compat.py  # POST /v1/images/generations and /v1/images/edits
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
└── requirements.txt
```

---

## Notes

- Inference requests are **serialized** via `asyncio.Lock` to prevent concurrent CUDA memory errors.
- The local pipeline is cached with `lru_cache(maxsize=1)` — loaded once, stays in GPU memory.
- `TORCHDYNAMO_DISABLE=1` and `expandable_segments:True` are set in the container for VRAM stability.
- Logs go to stdout and `imgen.log`.

---

## License

No explicit license. The FLUX.1-schnell model is subject to its own [license on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
