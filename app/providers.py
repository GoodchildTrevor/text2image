import os
import base64
import httpx
from abc import ABC, abstractmethod
from typing import Optional
import json
import re
import logging
from dotenv import load_dotenv
from app import openrouter_caps

load_dotenv()

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    @abstractmethod
    async def generate(self, model: str, prompt: str, **kwargs) -> tuple[bytes, str]: ...

    @abstractmethod
    async def edit(self, model: str, prompt: str, image_b64: str, **kwargs) -> tuple[bytes, str]: ...


class CloudProvider(BaseProvider, ABC):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}


class FallbackProvider(BaseProvider):
    """Tries ``primary``, falls back to ``secondary`` on any HTTP/network error."""

    def __init__(self, primary: BaseProvider, secondary: BaseProvider):
        self.primary = primary
        self.secondary = secondary

    async def generate(self, model: str, prompt: str, **kwargs) -> tuple[bytes, str]:
        try:
            result = await self.primary.generate(model=model, prompt=prompt, **kwargs)
            logger.info(f"FallbackProvider.generate: primary OK for model={model!r}")
            return result
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException, ValueError) as e:
            logger.warning(
                f"FallbackProvider.generate: primary failed for model={model!r} "
                f"({type(e).__name__}: {e}), switching to secondary"
            )
            return await self.secondary.generate(model=model, prompt=prompt, **kwargs)

    async def edit(self, model: str, prompt: str, image_b64: str, **kwargs) -> tuple[bytes, str]:
        try:
            result = await self.primary.edit(model=model, prompt=prompt, image_b64=image_b64, **kwargs)
            logger.info(f"FallbackProvider.edit: primary OK for model={model!r}")
            return result
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException, ValueError) as e:
            logger.warning(
                f"FallbackProvider.edit: primary failed for model={model!r} "
                f"({type(e).__name__}: {e}), switching to secondary"
            )
            return await self.secondary.edit(model=model, prompt=prompt, image_b64=image_b64, **kwargs)


class OpenAICompatProvider(CloudProvider):
    """Image generation and editing via an OpenAI-compatible /chat/completions API."""

    def __init__(self, api_key: str = None, base_url: str = None):
        super().__init__(
            api_key=api_key or os.getenv("API_KEY", ""),
            base_url=base_url or os.getenv("CLOUD_API_BASE_URL", ""),
        )

    def _parse_response(self, data: dict) -> tuple[bytes, str]:
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
                raise ValueError("HTTP image URL in message.images is not supported")

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

    async def _chat_completions(self, client: httpx.AsyncClient, model: str, messages: list) -> dict:
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
        self, model: str, prompt: str,
        width: int | None = None, height: int | None = None,
        steps: int = None, guidance: float = None,
        resolution: str | None = None, aspect_ratio: str | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=120) as client:
            data = await self._chat_completions(client, model, [{"role": "user", "content": prompt}])
            return self._parse_response(data)

    async def edit(
        self, model: str, prompt: str, image_b64: str,
        resolution: str | None = None, aspect_ratio: str | None = None,
        **kwargs,
    ) -> tuple[bytes, str]:
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
    """Image generation and editing via an OpenRouter-compatible /api/v1/images endpoint.

    ``base_url`` is **required** — read from ``OPENROUTER_BASE_URL`` env var.
    No hardcoded URL.
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        resolved_url = base_url or os.getenv("OPENROUTER_BASE_URL", "").rstrip("/")
        if not resolved_url:
            raise ValueError(
                "OpenRouterProvider requires OPENROUTER_BASE_URL env var to be set. "
                "Example: OPENROUTER_BASE_URL=https://routerai.ru"
            )
        super().__init__(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            base_url=resolved_url,
        )

    def _parse_image_response(self, data: dict) -> tuple[bytes, str]:
        items = data.get("data") or []
        if not items:
            raise ValueError(f"Empty data array: {json.dumps(data)[:300]}")
        b64 = items[0].get("b64_json")
        if not b64:
            raise ValueError(f"Missing b64_json: {json.dumps(items[0])[:300]}")
        logger.info("openrouter: image received via b64_json")
        return base64.b64decode(b64), ""

    def _build_size_payload(
        self, model: str,
        resolution: str | None, aspect_ratio: str | None,
        size: str | None, width: int | None, height: int | None,
    ) -> dict:
        payload = {}
        if size:
            payload["size"] = openrouter_caps.clamp_size(model, size)
        elif width and height:
            payload["size"] = openrouter_caps.clamp_size(model, f"{width}x{height}")
        elif resolution:
            payload["resolution"] = resolution
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
        elif aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        return payload

    async def generate(
        self, model: str, prompt: str,
        width: int | None = None, height: int | None = None,
        steps: int = None, guidance: float = None,
        resolution: str | None = None, aspect_ratio: str | None = None,
        size: str | None = None, quality: str | None = None,
        n: int | None = None, **kwargs,
    ) -> tuple[bytes, str]:
        payload: dict = {"model": model, "prompt": prompt}
        payload.update(self._build_size_payload(model, resolution, aspect_ratio, size, width, height))
        if quality:
            payload["quality"] = quality
        if n and n > 1:
            payload["n"] = n

        logger.info(
            f"openrouter generate: model={model!r} base_url={self.base_url!r} "
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
            return self._parse_image_response(resp.json())

    async def edit(
        self, model: str, prompt: str, image_b64: str,
        resolution: str | None = None, aspect_ratio: str | None = None,
        size: str | None = None, quality: str | None = None,
        n: int | None = None, **kwargs,
    ) -> tuple[bytes, str]:
        if not image_b64.startswith("data:"):
            image_b64 = f"data:image/png;base64,{image_b64}"

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "input_references": [{"type": "image_url", "image_url": {"url": image_b64}}],
        }
        payload.update(self._build_size_payload(model, resolution, aspect_ratio, size, None, None))
        if quality:
            payload["quality"] = quality
        if n and n > 1:
            payload["n"] = n

        logger.info(
            f"openrouter edit: model={model!r} base_url={self.base_url!r} "
            f"resolution={resolution!r} size={size!r} → payload_size={payload.get('size')!r}"
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
            return self._parse_image_response(resp.json())


def build_providers() -> dict[str, Optional[BaseProvider]]:
    """Build the provider registry from environment variables.

    Required env vars when using OpenRouter models:
        OPENROUTER_API_KEY      — bearer token
        OPENROUTER_BASE_URL     — base URL (e.g. https://routerai.ru), NO default

    Optional fallback:
        OPENROUTER_FALLBACK_URL — secondary base URL
        OPENROUTER_FALLBACK_KEY — secondary key (defaults to OPENROUTER_API_KEY)
    """
    providers: dict[str, Optional[BaseProvider]] = {}

    # ── Local models ──────────────────────────────────────────────────
    local_default = os.getenv("LOCAL_MODEL", "flux-schnell")
    for model_name in os.getenv("LOCAL_MODELS", local_default).split(","):
        model_name = model_name.strip()
        if model_name:
            providers[model_name] = None
            logger.info(f"Registered local model: {model_name!r}")

    # ── OpenAI-compat cloud provider ───────────────────────────────────
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

    # ── OpenRouter provider ────────────────────────────────────────────
    or_key = os.getenv("OPENROUTER_API_KEY")
    or_url = os.getenv("OPENROUTER_BASE_URL", "").rstrip("/")
    or_models = [m.strip() for m in os.getenv("OPENROUTER_MODELS", "").split(",") if m.strip()]

    if or_models and not or_key:
        logger.error("OPENROUTER_MODELS is set but OPENROUTER_API_KEY is missing — models will be unavailable")
    elif or_models and not or_url:
        logger.error(
            "OPENROUTER_MODELS is set but OPENROUTER_BASE_URL is missing — "
            "set OPENROUTER_BASE_URL (e.g. https://routerai.ru)"
        )
    elif or_key and or_url:
        try:
            primary = OpenRouterProvider(api_key=or_key, base_url=or_url)
        except ValueError as e:
            logger.error(f"Failed to create OpenRouterProvider: {e}")
            return providers

        fallback_url = os.getenv("OPENROUTER_FALLBACK_URL", "").rstrip("/")
        fallback_key = os.getenv("OPENROUTER_FALLBACK_KEY") or or_key

        if fallback_url:
            secondary = OpenRouterProvider(api_key=fallback_key, base_url=fallback_url)
            provider: BaseProvider = FallbackProvider(primary=primary, secondary=secondary)
            logger.info(
                f"OpenRouter: primary={or_url!r} fallback={fallback_url!r}"
            )
        else:
            provider = primary
            logger.info(f"OpenRouter: single endpoint {or_url!r}")

        for model_name in or_models:
            providers[model_name] = provider
            logger.info(f"Registered OpenRouter model: {model_name!r}")

    return providers


PROVIDERS = build_providers()
