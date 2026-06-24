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


class BaseProvider(ABC):
    """Abstract base class for image generation providers."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        steps: int = None,
        guidance: float = None,
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
        width: int,
        height: int,
        steps: int = None,
        guidance: float = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Text-to-image via OpenAI-compatible chat/completions."""
        async with httpx.AsyncClient(timeout=120) as client:
            messages = [{"role": "user", "content": prompt}]
            data = await self._chat_completions(client, model, messages)
            return self._parse_response(data)

    async def edit(
        self,
        model: str,
        prompt: str,
        image_b64: str,
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

    async def generate(
        self,
        model: str,
        prompt: str,
        width: int = None,
        height: int = None,
        steps: int = None,
        guidance: float = None,
        size: str = None,
        quality: str = None,
        n: int = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Text-to-image via OpenRouter /api/v1/images.

        :param model: OpenRouter model slug, e.g. ``openai/gpt-image-1``.
        :param prompt: Text prompt.
        :param width: Optional width — combined with height into ``size`` if provided.
        :param height: Optional height — combined with width into ``size`` if provided.
        :param size: Explicit size string, e.g. ``"1024x1024"`` or ``"2K"``.
                     Takes precedence over width/height.
        :param quality: ``auto``, ``low``, ``medium``, or ``high``.
        :param n: Number of images to generate (1-10).
        :returns: A tuple of (PNG image bytes, revised_prompt).
        """
        payload: dict = {"model": model, "prompt": prompt}

        resolved_size = size
        if not resolved_size and width and height:
            resolved_size = f"{width}x{height}"
        if resolved_size:
            payload["size"] = resolved_size

        if quality:
            payload["quality"] = quality
        if n and n > 1:
            payload["n"] = n

        logger.info(f"openrouter generate: model={model!r} size={resolved_size!r}")

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/images",
                headers=self._auth_headers(),
                json=payload,
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
        size: str = None,
        quality: str = None,
        n: int = None,
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
        if size:
            payload["size"] = size
        if quality:
            payload["quality"] = quality
        if n and n > 1:
            payload["n"] = n

        logger.info(f"openrouter edit: model={model!r}")

        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/images",
                headers=self._auth_headers(),
                json=payload,
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

    # ── Local models ─────────────────────────────────────────────────────────
    local_default = os.getenv("LOCAL_MODEL", "flux-schnell")
    for model_name in os.getenv("LOCAL_MODELS", local_default).split(","):
        model_name = model_name.strip()
        if model_name:
            providers[model_name] = None
            logger.info(f"Registered local model: {model_name!r}")

    # ── OpenAI-compat cloud provider ─────────────────────────────────────────
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

    # ── OpenRouter provider ───────────────────────────────────────────────────
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
