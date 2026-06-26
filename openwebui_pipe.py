import re
import asyncio
import httpx
from typing import Optional
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter):
        self.emit = event_emitter


class Pipe:
    class Valves(BaseModel):
        IMGEN_API_URL: str = Field(
            default="http://imgen:8010/v1/images/generations",
            description="OpenAI-compatible image generation endpoint",
        )
        IMGEN_TIMEOUT: int = Field(
            default=120,
            description="Maximum seconds for image generation",
        )
        MAX_ATTEMPTS: int = Field(
            default=2,
            description="Number of attempts to generate the image",
        )

    class UserValves(BaseModel):
        RESOLUTION: str = Field(
            default="1K",
            description="Разрешение изображения",
            json_schema_extra={
                "input": {
                    "type": "select",
                    "options": [
                        {"label": "1K  (~1024 px)", "value": "1K"},
                        {"label": "2K  (~2048 px)", "value": "2K"},
                        {"label": "4K  (~4096 px)", "value": "4K"},
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
                        {"label": "⬜ 1:1   — квадрат",         "value": "1:1"},
                        {"label": "🖼️  4:3   — пейзаж",          "value": "4:3"},
                        {"label": "📱 3:4   — портрет",          "value": "3:4"},
                        {"label": "🎬 16:9  — широкий",          "value": "16:9"},
                        {"label": "📲 9:16  — вертикальный",     "value": "9:16"},
                        {"label": "📷 3:2   — фото landscape",   "value": "3:2"},
                        {"label": "📸 2:3   — фото portrait",    "value": "2:3"},
                        {"label": "🎞️  21:9  — ultrawide",       "value": "21:9"},
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
                        {"label": "🌀 Gemini Flash Image", "value": "google/gemini-3.1-flash-image-preview"},
                        {"label": "🎨 Gemini Pro Image",   "value": "google/gemini-3-pro-image-preview"},
                        {"label": "🧠 GPT-5 Image",        "value": "openai/gpt-5-image"},
                        {"label": "⚡ GPT-5.4 Image 2",   "value": "openai/gpt-5.4-image-2"},
                    ],
                }
            },
        )

    def __init__(self):
        self.valves = self.Valves()

    def _extract_last_user_content(self, messages: list) -> tuple[str, str]:
        """
        Returns (text_prompt, image_data_url_or_empty).
        OpenWebUI sends images as multipart content arrays.
        """
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
        """
        Scans assistant messages in reverse for a previously generated image URL
        (http/https link ending in .png/.jpg/.webp) or base64 data URL.
        Prefers URL over base64 to avoid re-sending large payloads.
        """
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # Prefer static URL from imgen
            url_match = re.search(
                r"(https?://[^\s)"']+\.(?:png|jpg|jpeg|webp))", content
            )
            if url_match:
                return url_match.group(1)
            # Fallback: base64
            b64_match = re.search(
                r"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)", content
            )
            if b64_match:
                return b64_match.group(1)
        return ""

    async def _emit_status(self, emitter: EventEmitter, description: str, done: bool) -> None:
        if emitter.emit:
            await emitter.emit(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    async def pipe(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> str:
        emitter = EventEmitter(__event_emitter__)
        uv = (__user__ or {}).get("valves") or self.UserValves()

        resolution   = uv.RESOLUTION
        aspect_ratio = uv.ASPECT_RATIO
        model        = uv.IMGEN_MODEL

        messages = body.get("messages", [])
        prompt, user_image = self._extract_last_user_content(messages)
        image = user_image or self._extract_last_generated_image(messages)

        if not prompt and not image:
            await self._emit_status(emitter, "❌ Нет текста для генерации", True)
            return "❌ Нет текста для генерации изображения."

        size_params = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "response_format": "url",  # avoid base64 in DOM
        }

        if image:
            url = self.valves.IMGEN_API_URL.replace("/images/generations", "/images/edits")
            payload = {
                "model": model,
                "prompt": prompt or "Edit this image",
                "image": image,
                "n": 1,
                **size_params,
            }
            status_msg  = f"🖼️ Редактирование ({resolution} · {aspect_ratio})…"
        else:
            url = self.valves.IMGEN_API_URL
            payload = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                **size_params,
            }
            status_msg  = f"🎨 Генерация ({resolution} · {aspect_ratio})…"

        await self._emit_status(emitter, status_msg, False)

        for attempt in range(self.valves.MAX_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=self.valves.IMGEN_TIMEOUT) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                if resp.status_code == 200:
                    data = resp.json()["data"][0]
                    img_url = data.get("url")
                    if img_url:
                        return f"![Generated Image]({img_url})"
                    # fallback to b64 if server returned it anyway
                    b64 = data.get("b64_json", "")
                    return f"![Generated Image](data:image/png;base64,{b64})"

                err = resp.json().get("detail", resp.text)
                if attempt < self.valves.MAX_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                    continue
                await self._emit_status(emitter, f"❌ Ошибка: {err}", True)
                return f"❌ Ошибка генерации: {err}"

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
