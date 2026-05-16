# SPDX-License-Identifier: Apache-2.0
"""Audio processing utilities for VLM with audio modality (e.g. Gemma 4 nano).

Mirrors ``omlx/utils/image.py``: parse OpenAI ``input_audio`` content parts out
of chat messages, decode their bytes, and return ``(text_messages, audios)``
suitable for ``mlx_vlm.utils.prepare_inputs(..., audio=audios)``.

mlx-vlm's ``read_audio`` accepts a file path, a URL, or raw bytes; passing
bytes avoids a tempfile round-trip when the request carries base64 data.
"""

import base64
import logging
import urllib.request
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _decode_audio_part(payload: Dict[str, Any]) -> bytes | None:
    """Decode an ``input_audio`` payload to raw bytes.

    Accepts either ``{"data": "<base64>", "format": "wav"}`` (OpenAI native),
    ``{"url": "data:audio/wav;base64,..."}`` (data URI),
    or ``{"url": "http(s)://..."}`` / local file path.
    """
    if not isinstance(payload, dict):
        return None
    data_b64 = payload.get("data")
    if data_b64:
        try:
            return base64.b64decode(data_b64)
        except Exception as e:
            logger.warning("Failed to base64-decode input_audio.data: %s", e)
            return None
    url = payload.get("url")
    if not url:
        return None
    if url.startswith("data:"):
        try:
            _, body = url.split(",", 1)
            return base64.b64decode(body)
        except Exception as e:
            logger.warning("Failed to decode data URI input_audio: %s", e)
            return None
    if url.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            logger.warning("Failed to fetch input_audio URL: %s", e)
            return None
    try:
        with open(url, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("Failed to read input_audio path %s: %s", url, e)
        return None


def extract_audios_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[bytes]]:
    """Extract ``input_audio`` parts from OpenAI-format messages.

    Returns ``(text_messages, audios)`` where ``text_messages`` has every
    audio part removed (text parts merged into a single string) and
    ``audios`` is the ordered list of audio byte payloads.

    Symmetric with ``extract_images_from_messages`` so the engine can chain
    image and audio extraction without round-tripping through the API layer.
    """
    text_messages: List[Dict[str, Any]] = []
    audios: List[bytes] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            new_msg = {"role": role, "content": content or ""}
            for key in msg:
                if key not in ("role", "content"):
                    new_msg[key] = msg[key]
            text_messages.append(new_msg)
            continue

        text_parts: List[str] = []
        kept_non_text: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    text_parts.append(text)
            elif part_type in ("input_audio", "audio"):
                payload_key = "input_audio" if part_type == "input_audio" else "audio"
                payload = (
                    part.get(payload_key) if isinstance(part, dict)
                    else getattr(part, payload_key, None)
                )
                if hasattr(payload, "model_dump"):
                    payload = payload.model_dump()
                audio_bytes = _decode_audio_part(payload if isinstance(payload, dict) else {"url": payload})
                if audio_bytes is not None:
                    audios.append(audio_bytes)
            else:
                # Preserve image_url / input_image / etc. for downstream
                # extract_images_from_messages to handle.
                if isinstance(part, dict):
                    kept_non_text.append(part)

        if kept_non_text:
            # Audio extracted, image-bearing parts still need to flow through
            # extract_images_from_messages downstream. Keep them in a list and
            # prepend collected text as a text part for shape consistency.
            new_content: List[Dict[str, Any]] = []
            if text_parts:
                new_content.append({"type": "text", "text": "\n".join(text_parts)})
            new_content.extend(kept_non_text)
            new_msg = {"role": role, "content": new_content}
        else:
            new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}

        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, audios
