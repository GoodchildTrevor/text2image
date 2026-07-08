import re
import asyncio
import base64
import logging
import httpx
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_HIGH_RES_MODELS: frozenset[str] = frozenset(
    {
        "bytedance-seed/seedream-4.5",
    }
)

_DEFAULT_EDIT_PROMPT = "Edit this image."


def _format_error_detail(detail) -> str:
    if isinstance(detail, list):
        parts = []
        for item in detail:
            loc = ".".join(str(x) for x in item.get("loc", []))
            msg = item.get("msg", "")
            parts.append(f"{loc}: {msg}")
        return "; ".join(parts) if parts else str(detail)
    return str(detail)


class EventEmitter:
    def __init__(self, event_emitter):
        self.emit = event_emitter


class Pipe:
    class Valves(BaseModel):
        IMGEN_API_URL: str = Field(
            default="http://imgen:8010/v1/images/generations",
            description="OpenAI-compatible image generation endpoint",
        )
        IMGEN_TIMEOUT: int = Field(default=180)
        MAX_ATTEMPTS: int = Field(default=2)
        DEBUG: bool = Field(default=True)

    class UserValves(BaseModel):
        RESOLUTION: str = Field(
            default="1K",
            json_schema_extra={
                "input": {
                    "type": "select",
                    "options": [
                        {"label": "1K  (~1024 px) — все модели", "value": "1K"},
                        {
                            "label": "2K  (~2048 px) — только Seedream 4.5",
                            "value": "2K",
                        },
                        {
                            "label": "4K  (~4096 px) — только Seedream 4.5",
                            "value": "4K",
                        },
                    ],
                }
            },
        )
        ASPECT_RATIO: str = Field(
            default="1:1",
            description="Соотношение сторон",
            json_schema_extra={
                "input": {
                    "type": "select",
                    "options": [
                        {"label": "⬜ 1:1   — квадрат", "value": "1:1"},
                        {"label": "🖼️  4:3   — пейзаж", "value": "4:3"},
                        {"label": "📱 3:4   — портрет", "value": "3:4"},
                        {"label": "🎬 16:9  — широкий", "value": "16:9"},
                        {"label": "📲 9:16  — вертикальный", "value": "9:16"},
                        {"label": "📷 3:2   — фото landscape", "value": "3:2"},
                        {"label": "📸 2:3   — фото portrait", "value": "2:3"},
                        {"label": "🎞️  21:9  — ultrawide", "value": "21:9"},
                    ],
                }
            },
        )
        IMGEN_MODEL: str = Field(
            default="google/gemini-3.1-flash-image-preview",
            description="Модель генерации",
            json_schema_extra={
                "input": {
                    "type": "select",
                    "options": [
                        {
                            "label": "🍌 Nano Banana 2",
                            "value": "google/gemini-3.1-flash-image-preview",
                        },
                        {
                            "label": "🎨 Nano Banana Pro",
                            "value": "google/gemini-3-pro-image-preview",
                        },
                        {"label": "⚡ GPT Image", "value": "openai/gpt-5-image"},
                        {"label": "🧠 GPT Image 2", "value": "openai/gpt-5.4-image-2"},
                        {"label": "🖼️ Recraft V4", "value": "recraft/recraft-v4"},
                        {
                            "label": "✨ Recraft V4 Pro",
                            "value": "recraft/recraft-v4-pro",
                        },
                        {
                            "label": "☁️ Seedream 4.5  [2K/4K]",
                            "value": "bytedance-seed/seedream-4.5",
                        },
                        {
                            "label": "🏊 Riverflow V2 Fast",
                            "value": "sourceful/riverflow-v2-fast",
                        },
                        {
                            "label": "🌊 Riverflow V2 Pro",
                            "value": "sourceful/riverflow-v2-pro",
                        },
                    ],
                }
            },
        )

    def __init__(self):
        self.valves = self.Valves()

    def _extract_last_user_content(self, messages: list) -> tuple[str, str]:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                text, image = "", ""
                for part in content:
                    if part.get("type") == "text":
                        text += part.get("text", "")
                    elif part.get("type") == "image_url" and not image:
                        image = part.get("image_url", {}).get("url", "")
                return text.strip(), image
            return str(content).strip(), ""
        return "", ""

    def _extract_last_generated_image(self, messages: list) -> str:
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            url_match = re.search(r"(https?://\S+\.(?:png|jpg|jpeg|webp))", content)
            if url_match:
                return url_match.group(1)
            b64_match = re.search(r"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)", content)
            if b64_match:
                return b64_match.group(1)
        return ""

    async def _emit_status(self, emitter: EventEmitter, description: str, done: bool) -> None:
        if emitter.emit:
            await emitter.emit(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    def _clamp_resolution(self, model: str, requested: str) -> tuple[str, bool]:
        if requested != "1K" and model not in _HIGH_RES_MODELS:
            return "1K", True
        return requested, False

    async def _get_image_b64(self, image: str, timeout: int) -> str:
        """Return image as base64 string (with data: prefix)."""
        if image.startswith("data:image"):
            return image  # already b64
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(image)
            resp.raise_for_status()
            raw = resp.content
        mime = "image/png"
        ct = resp.headers.get("content-type", "")
        if ct.startswith("image/"):
            mime = ct.split(";")[0].strip()
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}"

    async def _post_json(self, api_url: str, payload: dict, timeout: int) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                api_url, json=payload, headers={"Content-Type": "application/json"}
            )

    def _parse_success(self, resp: httpx.Response) -> str:
        item = resp.json()["data"][0]
        img_url = item.get("url")
        if img_url:
            return f"![Generated Image]({img_url})"
        b64 = item.get("b64_json", "")
        return f"![Generated Image](data:image/png;base64,{b64})"

    def _parse_error(self, resp: httpx.Response) -> str:
        try:
            raw_detail = resp.json().get("detail", resp.text)
        except Exception:
            raw_detail = resp.text
        return f"HTTP {resp.status_code}: {_format_error_detail(raw_detail)}"

    async def pipe(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> str:
        emitter = EventEmitter(__event_emitter__)
        uv = (__user__ or {}).get("valves") or self.UserValves()

        aspect_ratio: str = uv.ASPECT_RATIO
        model: str = uv.IMGEN_MODEL
        resolution, was_clamped = self._clamp_resolution(model, uv.RESOLUTION)

        messages = body.get("messages", [])
        raw_prompt, user_image = self._extract_last_user_content(messages)
        image = user_image or self._extract_last_generated_image(messages)

        prompt = raw_prompt.strip() if raw_prompt else ""

        if not prompt and not image:
            await self._emit_status(emitter, "❌ Нет текста для генерации", True)
            return "❌ Нет текста для генерации изображения."

        clamp_note = " (модель не поддерживает выше 1K)" if was_clamped else ""
        is_edit = bool(image)

        # Скачиваем байты и конвертируем в b64 один раз до retry-цикла
        image_b64: Optional[str] = None
        if is_edit:
            try:
                image_b64 = await self._get_image_b64(image, self.valves.IMGEN_TIMEOUT)
                if self.valves.DEBUG:
                    logger.info(f"[imgen pipe] image_b64 len={len(image_b64)} downloaded")
            except Exception as e:
                await self._emit_status(emitter, "❌ Не удалось загрузить изображение", True)
                return f"❌ Не удалось загрузить исходное изображение: {type(e).__name__}: {e}"

        if is_edit:
            effective_prompt = prompt if prompt else _DEFAULT_EDIT_PROMPT
            api_url = self.valves.IMGEN_API_URL.replace("/images/generations", "/images/edits")
            payload = {
                "model": model,
                "prompt": effective_prompt,
                "image": image_b64,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "response_format": "url",
            }
            status_msg = f"🖼️ Редактирование {resolution} ({aspect_ratio}){clamp_note}…"
            success_msg = "✅ Изображение отредактировано!"
        else:
            effective_prompt = prompt
            api_url = self.valves.IMGEN_API_URL
            payload = {
                "model": model,
                "prompt": effective_prompt,
                "n": 1,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "response_format": "url",
            }
            status_msg = f"🎨 Генерация {resolution} ({aspect_ratio}){clamp_note}…"
            success_msg = "✅ Изображение сгенерировано!"

        if self.valves.DEBUG:
            logger.info(
                f"[imgen pipe] is_edit={is_edit} raw_prompt={raw_prompt!r} "
                f"effective_prompt={effective_prompt!r} api_url={api_url} "
                f"model={model!r} resolution={resolution} aspect_ratio={aspect_ratio}"
            )

        await self._emit_status(emitter, status_msg, False)

        last_err = None
        for attempt in range(self.valves.MAX_ATTEMPTS):
            try:
                resp = await self._post_json(api_url, payload, self.valves.IMGEN_TIMEOUT)

                if resp.status_code == 200:
                    await self._emit_status(emitter, success_msg, True)
                    return self._parse_success(resp)

                last_err = self._parse_error(resp)
                if self.valves.DEBUG:
                    logger.warning(f"[imgen pipe] attempt={attempt} error={last_err}")

                if attempt < self.valves.MAX_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                    continue

                await self._emit_status(emitter, f"❌ Ошибка: {last_err}", True)
                return f"❌ Ошибка генерации: {last_err}"

            except httpx.TimeoutException:
                if attempt < self.valves.MAX_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                else:
                    await self._emit_status(emitter, "⏱️ Таймаут генерации", True)
                    return "❌ Таймаут генерации изображения."

            except httpx.ConnectError:
                if attempt < self.valves.MAX_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                else:
                    await self._emit_status(emitter, "❌ Нет соединения с сервером", True)
                    return "❌ Не удалось подключиться к серверу генерации."

            except Exception as e:
                if attempt < self.valves.MAX_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                    continue
                await self._emit_status(emitter, f"❌ Ошибка: {e}", True)
                return f"❌ Ошибка: {type(e).__name__}: {str(e)}"

        return "❌ Исчерпаны попытки генерации."
