import os
import base64
import httpx
from typing import Dict, Optional, Tuple
import json
import re
import logging

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
    ) -> Tuple[bytes, str]:
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
    ) -> Tuple[bytes, str]:
        """Edit an existing image based on a text prompt.

        :param model: The model identifier to use for editing.
        :param prompt: The text prompt describing the desired edit.
        :param image_b64: Base64-encoded input image.
        :returns: A tuple of (image_bytes, revised_prompt).
        :raises NotImplementedError: This is an abstract method.
        """


class RouterAIProvider(BaseProvider):
    """Image generation and editing provider via RouterAI API (https://routerai.ru).

    Text-to-image and image-edit operations are performed through the
    RouterAI chat/completions endpoint.

    :param api_key: API key for RouterAI. Falls back to ROUTERAI_API_KEY env var.
    """

    def __init__(self, api_key: str = None):
        """Initialize RouterAIProvider.

        :param api_key: API key for RouterAI. Falls back to
            ``ROUTERAI_API_KEY`` environment variable if not provided.
        """
        self.api_key = api_key or os.getenv("ROUTERAI_API_KEY")
        self.base_url = "https://routerai.ru/api/v1"

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _parse_response(self, data: dict) -> Tuple[bytes, str]:
        """Parse RouterAI chat/completions response and extract image bytes.

        Handles all known response formats:
        - ``message.images[].image_url.url`` — data URI or HTTP URL
        - ``message.parts[].inlineData`` — Gemini-style base64
        - ``message.parts[].fileData.fileUri``
        - ``message.content`` — plain data URI or HTTP URL

        :param data: The JSON response dict from RouterAI chat/completions.
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
                logger.info("RouterAI: image found in message.images (data URI)")
                b64 = re.sub(r"^data:image/.+;base64,", "", url)
                return base64.b64decode(b64), content or ""
            if url.startswith("http"):
                logger.info("RouterAI: image found in message.images (URL) — not supported in sync parse")
                raise ValueError("HTTP image URL in message.images is not supported in _parse_response")

        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                logger.info("RouterAI: image found in inlineData")
                return base64.b64decode(inline["data"]), content or ""
            url = (
                (part.get("fileData") or {}).get("fileUri")
                or (part.get("image_url") or {}).get("url", "")
            )
            if url.startswith("data:image"):
                logger.info("RouterAI: image found in parts (data URI)")
                b64 = re.sub(r"^data:image/.+;base64,", "", url)
                return base64.b64decode(b64), content or ""

        if isinstance(content, str):
            if content.startswith("data:image"):
                logger.info("RouterAI: image found in content (data URI)")
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
        """Send a chat/completions request to RouterAI API.

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
        logger.info(f"RouterAI response: {json.dumps(data, ensure_ascii=False)[:500]}")
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
    ) -> Tuple[bytes, str]:
        """Text-to-image via RouterAI chat/completions.

        Width and height are accepted for interface compatibility but
        ignored — RouterAI does not support explicit size parameters.

        :param model: The model identifier.
        :param prompt: The text prompt describing the desired image.
        :param width: Requested width (ignored by RouterAI).
        :param height: Requested height (ignored by RouterAI).
        :param steps: Number of inference steps (ignored by RouterAI).
        :param guidance: Guidance scale (ignored by RouterAI).
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
    ) -> Tuple[bytes, str]:
        """Image-to-image edit via RouterAI chat/completions.

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


def build_providers() -> Dict[str, Optional[BaseProvider]]:
    """Build the provider registry from environment variables.

    Environment variables:
        API_KEY — API key from your provider.
        CLOUD_MODELS — Comma-separated model names to register from your provider.

    :returns: Dict mapping model name to provider instance (or None for local models).
    """
    providers: Dict[str, Optional[BaseProvider]] = {
        "flux-schnell": None,
        "google/gemini-3.1-flash-image-preview": None,
        "google/gemini-3-pro-image-preview": None,
        "openai/gpt-5-image": None,
        "openai/gpt-5.4-image-2": None,
    }

    router_key = os.getenv("API_KEY")
    if router_key:
        router = RouterAIProvider(api_key=router_key)
        for model_name in os.getenv("CLOUD_MODELS", "").split(","):
            model_name = model_name.strip()
            if model_name:
                providers[model_name] = router
    else:
        logger.warning("ROUTERAI_API_KEY is not set — RouterAI models unavailable")

    return providers


PROVIDERS = build_providers()
