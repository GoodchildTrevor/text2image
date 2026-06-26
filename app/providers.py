import os
import base64
import httpx
from abc import ABC, abstractmethod
from typing import Optional
import json
import re
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Per-model size constraints ───────────────────────────────────────────────
# Maps model slug prefix → list of allowed WxH strings (largest first).
# When the requested size is not in the list, the closest smaller one is used.
# Models NOT in this map pass the size through as-is.
_MODEL_ALLOWED_SIZES: dict[str, list[str]] = {
    # GPT-5 Image family — supports square + two landscape/portrait orientations
    "openai/gpt-5-image": [
        "1792x1024", "1024x1792", "1024x1024",
    ],
    "openai/gpt-5-image-mini": [
        "1792x1024", "1024x1792", "1024x1024",
    ],
    "openai/gpt-5.4-image-2": [
        "1792x1024", "1024x1792", "1024x1024",
    ],
    # Gemini image family — only 1:1 1024 confirmed via OpenRouter
    "google/gemini-3.1-flash-image-preview": ["1024x1024"],
    "google/gemini-3-pro-image-preview":     ["1024x1024"],
    "google/gemini-2.5-flash-image":         ["1024x1024"],
}


def _clamp_size(model: str, requested: str | None) -> str | None:
    """Return the best allowed size for *model* given *requested* ``WxH``.

    Strategy:
    - If model has no size constraints → return *requested* unchanged.
    - If model has constraints and *requested* is in the list → keep it.
    - Otherwise pick the largest allowed size whose long side ≤ requested long side.
    - If nothing is smaller → return the smallest allowed size.

    :param model: OpenRouter model slug.
    :param requested: Pixel string ``"WxH"`` or ``None``.
    :returns: Clamped or original size string, or ``None`` if input is ``None``.
    """
    if requested is None:
        return None

    allowed = _MODEL_ALLOWED_SIZES.get(model)
    if not allowed:
        return requested  # no constraints for this model

    if requested in allowed:
        return requested

    try:
        req_w, req_h = map(int, requested.split("x"))
    except ValueError:
        return allowed[0]  # malformed — use largest allowed

    req_long = max(req_w, req_h)

    # Find largest allowed whose long side <= requested long side
    for candidate in allowed:  # already sorted largest-first
        try:
            c_w, c_h = map(int, candidate.split("x"))
        except ValueError:
            continue
        # Match orientation: landscape→landscape, portrait→portrait, square→square
        req_landscape = req_w > req_h
        c_landscape = c_w > c_h
        req_portrait = req_h > req_w
        c_portrait = c_h > c_w
        same_orientation = (
            (req_landscape and c_landscape)
            or (req_portrait and c_portrait)
            or (req_w == req_h and c_w == c_h)
        )
        if same_orientation and max(c_w, c_h) <= req_long:
            logger.info(
                f"size clamp: model={model!r} {requested!r} → {candidate!r}"
            )
            return candidate

    # Fallback: square 1024x1024 or last in list
    fallback = "1024x1024" if "1024x1024" in allowed else allowed[-1]
    logger.info(f"size clamp fallback: model={model!r} {requested!r} → {fallback!r}")
    return fallback


class BaseProvider(ABC):
    """Abstract base class for image generation providers."""

    @abstractmethod
    async def generate(
        self,
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
        """Generate an image from a text prompt.

        :returns: A tuple of (image_bytes, revised_prompt).
        """

    @abstractmethod
    async def edit(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Edit an existing image based on a text prompt.

        :returns: A tuple of (image_bytes, revised_prompt).
        """


class CloudProvider(BaseProvider, ABC):
    """Abstract base for cloud image providers.

    Handles common auth and HTTP plumbing shared by all cloud backends.

    :param api_key: Bearer token for the provider API.
    :param base_url: Base URL of the provider endpoint (trailing slash stripped).
    """

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}


class OpenAICompatProvider(CloudProvider):
    """Image generation and editing via an OpenAI-compatible /chat/completions API.

    Communicates with any provider that implements the OpenAI chat/completions
    interface and returns images in the response (data URI, inlineData, or URL).

    ``resolution`` and ``aspect_ratio`` are accepted for interface compatibility
    but ignored — the upstream provider controls output dimensions.

    :param api_key: Bearer token. Falls back to ``API_KEY`` env var.
    :param base_url: Base URL. Falls back to ``CLOUD_API_BASE_URL`` env var.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(
            api_key=api_key or os.getenv("API_KEY", ""),
            base_url=base_url or os.getenv("CLOUD_API_BASE_URL", ""),
        )

    def _parse_response(self, data: dict) -> tuple[bytes, str]:
        """Parse an OpenAI-compatible chat/completions response and extract image bytes.

        Handles all known image response formats:
        - ``message.images[].image_url.url`` — data URI or HTTP URL
        - ``message.parts[].inlineData`` — Gemini-style base64
        - ``message.parts[].fileData.fileUri``
        - ``message.content`` — plain data URI or HTTP URL

        :param data: The JSON response dict from the chat/completions endpoint.
        :returns: A tuple of (image_bytes, text_content).
        :raises ValueError: If no image is found in the response.
        """
        message = data["choices"][0]["message"]
        content = message.get("content")
        parts = message.get("parts") or []
        images = message.get("images") or []

        for img in images:
            url = (img.get("image_url") or {}).get("url", "")
            if url.startswith("data:image"):
                logger.info("cloud: image found in message.images (data URI)")
                b64 = re.sub(r"^data:image/.+;base64,", "", url)
                return base64.b64decode(b64), content or ""
            if url.startswith("http"):
                raise ValueError("HTTP image URL in message.images is not supported in _parse_response")

        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                logger.info("cloud: image found in inlineData")
                return base64.b64decode(inline["data"]), content or ""
            url = (
                (part.get("fileData") or {}).get("fileUri")
                or (part.get("image_url") or {}).get("url", "")
            )
            if url.startswith("data:image"):
                logger.info("cloud: image found in parts (data URI)")
                b64 = re.sub(r"^data:image/.+;base64,", "", url)
                return base64.b64decode(b64), content or ""

        if isinstance(content, str):
            if content.startswith("data:image"):
                logger.info("cloud: image found in content (data URI)")
                b64 = re.sub(r"^data:image/.+;base64,", "", content)
                return base64.b64decode(b64), content
            if content.startswith("http"):
                raise ValueError("URL-only content response is not supported")

        raise ValueError(
            f"No image found in response. "
            f"message keys: {list(message.keys())}, "
            f"content preview: {str(content)[:100]}"
        )

    async def _chat_completions(
        self, client: httpx.AsyncClient, model: str, messages: list
    ) -> dict:
        resp = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._auth_headers(),
            json={"model": model, "messages": messages},
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"cloud response: {json.dumps(data, ensure_ascii=False)[:500]}")
        return data

    async def generate(
        self,
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
        """Text-to-image via OpenAI-compatible chat/completions.

        ``resolution``, ``aspect_ratio``, ``width``, and ``height`` are
        accepted for interface compatibility but ignored.
        """
        async with httpx.AsyncClient(timeout=120) as client:
            messages = [{"role": "user", "content": prompt}]
            data = await self._chat_completions(client, model, messages)
            return self._parse_response(data)

    async def edit(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Image-to-image edit via OpenAI-compatible chat/completions."""
        if not image_b64.startswith("data:"):
            image_b64 = f"data:image/png;base64,{image_b64}"

        async with httpx.AsyncClient(timeout=120) as client:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }]
            data = await self._chat_completions(client, model, messages)
            return self._parse_response(data)


class OpenRouterProvider(CloudProvider):
    """Image generation and editing via the OpenRouter dedicated Image API.

    Uses ``POST /api/v1/images`` (not /chat/completions).
    Returns images as ``data[0].b64_json``.

    Parameter priority for output dimensions:
    1. ``size`` (``WxH`` string) — clamped to model-specific allowed list, then forwarded.
    2. ``width`` + ``height``    — combined into ``WxH`` size string, then clamped.
    3. ``resolution`` + ``aspect_ratio`` — forwarded directly (legacy/fallback path).

    Docs: https://openrouter.ai/docs/guides/overview/multimodal/image-generation

    :param api_key: Bearer token. Falls back to ``OPENROUTER_API_KEY`` env var.
    :param base_url: Base URL. Defaults to ``https://openrouter.ai``.
    """

    OPENROUTER_BASE = "https://openrouter.ai"

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL", self.OPENROUTER_BASE),
        )

    def _parse_image_response(self, data: dict) -> tuple[bytes, str]:
        """Parse OpenRouter /api/v1/images response.

        Expected shape::

            {
                "created": 1748372400,
                "data": [{"b64_json": "<base64>"}],
                "usage": {...}
            }

        :param data: JSON response dict.
        :returns: A tuple of (image_bytes, revised_prompt).
        :raises ValueError: If ``data[0].b64_json`` is missing.
        """
        items = data.get("data") or []
        if not items:
            raise ValueError(f"OpenRouter returned empty data array: {json.dumps(data)[:300]}")
        b64 = items[0].get("b64_json")
        if not b64:
            raise ValueError(f"OpenRouter response missing b64_json: {json.dumps(items[0])[:300]}")
        logger.info("openrouter: image received via b64_json")
        return base64.b64decode(b64), ""

    def _build_size_payload(
        self,
        model: str,
        resolution: str | None,
        aspect_ratio: str | None,
        size: str | None,
        width: int | None,
        height: int | None,
    ) -> dict:
        """Resolve dimension params into OpenRouter payload keys.

        Priority: size > width+height > resolution/aspect_ratio.
        ``size`` and derived ``WxH`` strings are clamped via ``_clamp_size``.
        """
        payload = {}
        if size:
            clamped = _clamp_size(model, size)
            payload["size"] = clamped
        elif width and height:
            raw = f"{width}x{height}"
            clamped = _clamp_size(model, raw)
            payload["size"] = clamped
        elif resolution:
            payload["resolution"] = resolution
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
        elif aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        return payload

    async def generate(
        self,
        model: str,
        prompt: str,
        width: int | None = None,
        height: int | None = None,
        steps: int = None,
        guidance: float = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        n: int | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Text-to-image via OpenRouter /api/v1/images.

        :param model: OpenRouter model slug, e.g. ``openai/gpt-5-image``.
        :param prompt: Text prompt.
        :param size: Pixel string ``"WxH"`` — clamped to model-specific limits.
        :param resolution: Tier string (``"1K"`` etc.) forwarded only when ``size`` absent.
        :param aspect_ratio: Ratio string forwarded alongside ``resolution``.
        :param quality: ``auto``, ``low``, ``medium``, or ``high``.
        :param n: Number of images to generate (1-10).
        :returns: A tuple of (PNG image bytes, revised_prompt).
        """
        payload: dict = {"model": model, "prompt": prompt}
        payload.update(
            self._build_size_payload(model, resolution, aspect_ratio, size, width, height)
        )

        if quality:
            payload["quality"] = quality
        if n and n > 1:
            payload["n"] = n

        logger.info(
            f"openrouter generate: model={model!r} "
            f"resolution={resolution!r} aspect_ratio={aspect_ratio!r} "
            f"size={size!r} → payload_size={payload.get('size')!r}"
        )

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/images",
                headers=self._auth_headers(),
                json=payload,
            )
            if not resp.is_success:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text
                logger.error(
                    f"openrouter {resp.status_code}: model={model!r} "
                    f"payload={json.dumps(payload, ensure_ascii=False)} "
                    f"response={json.dumps(err_body, ensure_ascii=False)[:500]}"
                )
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"openrouter response keys: {list(data.keys())}")
            return self._parse_image_response(data)

    async def edit(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        n: int | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Image-to-image edit via OpenRouter /api/v1/images with input_references.

        :param image_b64: Base64-encoded image — raw base64 or data URI.
        """
        if not image_b64.startswith("data:"):
            image_b64 = f"data:image/png;base64,{image_b64}"

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "input_references": [
                {"type": "image_url", "image_url": {"url": image_b64}}
            ],
        }
        payload.update(
            self._build_size_payload(model, resolution, aspect_ratio, size, None, None)
        )

        if quality:
            payload["quality"] = quality
        if n and n > 1:
            payload["n"] = n

        logger.info(
            f"openrouter edit: model={model!r} resolution={resolution!r} "
            f"size={size!r} → payload_size={payload.get('size')!r}"
        )

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/images",
                headers=self._auth_headers(),
                json=payload,
            )
            if not resp.is_success:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text
                logger.error(
                    f"openrouter edit {resp.status_code}: model={model!r} "
                    f"response={json.dumps(err_body, ensure_ascii=False)[:500]}"
                )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_image_response(data)


def build_providers() -> dict[str, Optional[BaseProvider]]:
    """Build the provider registry from environment variables.

    Environment variables:
        LOCAL_MODEL           — HuggingFace model ID for the local diffusion pipeline.
                                Also used as fallback when LOCAL_MODELS is not set.
        LOCAL_MODELS          — Comma-separated short names mapped to the local pipeline
                                (provider = None). Defaults to LOCAL_MODEL.
        API_KEY               — Bearer token for the OpenAI-compat cloud provider.
        CLOUD_API_BASE_URL    — Base URL of the OpenAI-compat cloud endpoint.
        CLOUD_MODELS          — Comma-separated model IDs → OpenAICompatProvider.
        OPENROUTER_API_KEY    — Bearer token for OpenRouter.
        OPENROUTER_BASE_URL   — Override OpenRouter base URL (default: https://openrouter.ai).
        OPENROUTER_MODELS     — Comma-separated model slugs → OpenRouterProvider.

    :returns: Dict mapping model name to provider instance (or None for local).
    """
    providers: dict[str, Optional[BaseProvider]] = {}

    # ── Local models ────────────────────────────────────────────────────────
    local_default = os.getenv("LOCAL_MODEL", "flux-schnell")
    for model_name in os.getenv("LOCAL_MODELS", local_default).split(","):
        model_name = model_name.strip()
        if model_name:
            providers[model_name] = None
            logger.info(f"Registered local model: {model_name!r}")

    # ── OpenAI-compat cloud provider ────────────────────────────────────────
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("CLOUD_API_BASE_URL", "").rstrip("/")
    if api_key and base_url:
        cloud = OpenAICompatProvider(api_key=api_key, base_url=base_url)
        for model_name in os.getenv("CLOUD_MODELS", "").split(","):
            model_name = model_name.strip()
            if model_name:
                providers[model_name] = cloud
                logger.info(f"Registered cloud model: {model_name!r} → {base_url}")
    elif api_key and not base_url:
        logger.warning("API_KEY set but CLOUD_API_BASE_URL missing — cloud models unavailable")

    # ── OpenRouter provider ──────────────────────────────────────────────────
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        openrouter = OpenRouterProvider(api_key=or_key)
        for model_name in os.getenv("OPENROUTER_MODELS", "").split(","):
            model_name = model_name.strip()
            if model_name:
                providers[model_name] = openrouter
                logger.info(f"Registered OpenRouter model: {model_name!r}")
    else:
        logger.info("OPENROUTER_API_KEY not set — OpenRouter models disabled")

    return providers


PROVIDERS = build_providers()
