import os
import base64
import httpx
from typing import Optional
import json
import re
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BaseProvider:
    """Abstract base class for image generation providers.

    Subclasses must implement :meth:`generate` and optionally
    :meth:`edit` to support text-to-image and image-edit operations.
    """

    async def generate(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        steps: int = None,
        guidance: float = None,
    ) -> tuple[bytes, str]:
        """Generate an image from a text prompt.

        :param model: The model identifier to use for generation.
        :param prompt: The text prompt describing the desired image.
        :param width: The requested image width in pixels.
        :param height: The requested image height in pixels.
        :param steps: Number of inference steps (optional).
        :param guidance: Guidance scale for generation (optional).
        :returns: A tuple of (image_bytes, revised_prompt).
        :raises NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    async def edit(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        **kwargs,
    ) -> tuple[bytes, str]:
        """Edit an existing image based on a text prompt.

        :param model: The model identifier to use for editing.
        :param prompt: The text prompt describing the desired edit.
        :param image_b64: Base64-encoded input image.
        :returns: A tuple of (image_bytes, revised_prompt).
        :raises NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError


class OpenAICompatProvider(BaseProvider):
    """Image generation and editing via an OpenAI-compatible chat/completions API.

    Communicates with any provider that implements the OpenAI chat/completions
    interface and returns images in the response (data URI, inlineData, or URL).

    :param api_key: Bearer token for the provider API.
    :param base_url: Base URL of the OpenAI-compatible endpoint.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize OpenAICompatProvider.

        :param api_key: Bearer token. Falls back to ``API_KEY`` env var.
        :param base_url: API base URL. Falls back to ``CLOUD_API_BASE_URL`` env var.
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("CLOUD_API_BASE_URL", "").rstrip("/")

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

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
                logger.info("cloud: image found in message.images (URL) — not supported in sync parse")
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
        """Send a chat/completions request to the cloud provider.

        :param client: The httpx.AsyncClient instance.
        :param model: The model identifier.
        :param messages: List of message dicts for the conversation.
        :returns: The JSON response dict from the API.
        :raises httpx.HTTPStatusError: If the API returns an error status.
        """
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
        """Text-to-image via OpenAI-compatible chat/completions.

        Width, height, steps, and guidance are accepted for interface
        compatibility but ignored — the cloud provider controls output size.

        :param model: The model identifier.
        :param prompt: The text prompt describing the desired image.
        :param width: Requested width (passed for interface compatibility, ignored).
        :param height: Requested height (passed for interface compatibility, ignored).
        :param steps: Number of inference steps (ignored by cloud provider).
        :param guidance: Guidance scale (ignored by cloud provider).
        :returns: A tuple of (image_bytes, revised_prompt).
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
        **kwargs,
    ) -> tuple[bytes, str]:
        """Image-to-image edit via OpenAI-compatible chat/completions.

        :param image_b64: Base64-encoded image — either a data URI
                          (``data:image/png;base64,...``) or raw base64 string.
        """
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


def build_providers() -> dict[str, Optional[BaseProvider]]:
    """Build the provider registry from environment variables.

    Environment variables:
        LOCAL_MODEL        — HuggingFace model ID for the local diffusion pipeline.
                             Also used as fallback when LOCAL_MODELS is not set.
        LOCAL_MODELS       — Comma-separated short model names that map to the local
                             pipeline (provider = None). Defaults to LOCAL_MODEL.
        API_KEY            — Bearer token for the cloud provider API.
        CLOUD_API_BASE_URL — Base URL of the OpenAI-compatible cloud endpoint.
        CLOUD_MODELS       — Comma-separated cloud model IDs routed to the cloud provider.

    :returns: Dict mapping model name to provider instance (or None for local).
    """
    providers: dict[str, Optional[BaseProvider]] = {}

    local_default = os.getenv("LOCAL_MODEL", "flux-schnell")
    for model_name in os.getenv("LOCAL_MODELS", local_default).split(","):
        model_name = model_name.strip()
        if model_name:
            providers[model_name] = None
            logger.info(f"Registered local model: {model_name!r}")

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
        logger.warning("API_KEY is set but CLOUD_API_BASE_URL is missing — cloud models unavailable")
    else:
        logger.info("API_KEY not set — cloud models disabled")

    return providers


PROVIDERS = build_providers()
