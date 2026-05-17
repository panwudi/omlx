# SPDX-License-Identifier: Apache-2.0
"""
Audio API routes for oMLX.

This module provides OpenAI-compatible audio endpoints:
- POST /v1/audio/transcriptions  - Speech-to-Text
- POST /v1/audio/speech          - Text-to-Speech
- POST /v1/audio/process         - Speech-to-Speech / audio processing
"""

import base64
import logging
import math
import os
import re
import tempfile
from typing import AsyncIterator, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from ..engine.audio_utils import wav_bytes_to_pcm_frames, wav_header
from ..server_metrics import get_server_metrics
from .audio_models import AudioSpeechRequest, AudioTranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum upload size for audio files (100 MB).
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024

# Maximum base64-encoded ref_audio size (~15 MB raw audio, enough for ~60s).
MAX_REF_AUDIO_BASE64_BYTES = 20 * 1024 * 1024

# Default native TTS chunk cadence. Keep this below the mlx-audio default to
# improve TTFT while still letting the model process the full input at once.
DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.2
MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.01

# Video container extensions that should be routed through ffmpeg decoding.
# mlx-audio only recognises audio-specific extensions (m4a, aac, ogg, opus),
# so we remap video containers to .m4a before handing off. ffmpeg detects the
# actual format from file content, not the extension.
_VIDEO_CONTAINERS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"}


# ---------------------------------------------------------------------------
# Engine pool accessor — patched in tests via omlx.api.audio_routes._get_engine_pool
# ---------------------------------------------------------------------------


def _get_engine_pool():
    """Return the active EnginePool from server state.

    Imported lazily to avoid a circular import at module load time.
    Can be replaced in tests via patch('omlx.api.audio_routes._get_engine_pool').
    """
    # Import here to avoid circular imports at module load
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve a model alias to its real model ID.

    Delegates to the same resolve_model_id used by LLM/chat endpoints,
    ensuring audio endpoints handle aliases consistently.
    """
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _get_settings_manager():
    """Return the active ModelSettingsManager from server state, or None.

    Lazy import + defensive guard so the audio router stays usable in tests
    that don't bring up the full server state.
    """
    try:
        from omlx.server import _server_state
    except Exception:
        return None
    return getattr(_server_state, "settings_manager", None)


def _fmt_srt_time(t: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(t: float) -> str:
    """WebVTT timestamp: HH:MM:SS.mmm"""
    return _fmt_srt_time(t).replace(",", ".")


def _group_words_into_cues(
    words: list[dict],
    max_chars: int = 40,
    max_duration: float = 5.0,
) -> list[dict]:
    """Pack consecutive word-level entries into subtitle cues.

    A cue is closed when any of these holds:
      * accumulated text reaches ``max_chars``
      * cue duration reaches ``max_duration`` seconds
      * the current word ends with a sentence-terminating punctuation
        (Chinese 。！？ or Western .!?)

    Each input word entry should have ``word`` / ``start`` / ``end`` keys
    (the shape produced by the Whisper word-timestamps path and the
    Qwen3-ForcedAligner auto-chain).
    """
    if not words:
        return []
    cues: list[dict] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end = 0.0
    for w in words:
        token = (w.get("word") or "").strip("\n")
        if not token:
            continue
        if cur_start is None:
            cur_start = float(w.get("start", 0.0))
        cur_text += token
        cur_end = float(w.get("end", cur_end))
        last = token[-1:] if token else ""
        cue_full = (
            len(cur_text) >= max_chars
            or (cur_end - cur_start) >= max_duration
            or last in "。！？.!?"
        )
        if cue_full:
            cues.append({
                "start": cur_start,
                "end": max(cur_end, cur_start + 0.05),
                "text": cur_text.strip(),
            })
            cur_text = ""
            cur_start = None
            cur_end = 0.0
    if cur_text.strip() and cur_start is not None:
        cues.append({
            "start": cur_start,
            "end": max(cur_end, cur_start + 0.05),
            "text": cur_text.strip(),
        })
    return cues


def _build_subtitle(result: dict, fmt: str = "srt") -> str:
    """Build SRT or WebVTT body from an STTEngine.transcribe() result.

    Prefers word-level entries (segments[*].words from word_timestamps /
    Forced-Aligner chain). Falls back to segment-level entries when no
    word data is present (one cue per ASR segment).
    """
    segments = result.get("segments") or []
    cues: list[dict] = []
    # 1) Word-level path
    all_words: list[dict] = []
    for seg in segments:
        ws = seg.get("words") or []
        if ws:
            all_words.extend(ws)
    if all_words:
        cues = _group_words_into_cues(all_words)
    # 2) Fall back to segment-level
    if not cues and segments:
        cues = [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip(),
            }
            for s in segments
            if (s.get("text") or "").strip()
        ]
    # 3) Final fallback: a single cue from the whole text + duration
    if not cues and (result.get("text") or "").strip():
        cues = [{
            "start": 0.0,
            "end": float(result.get("duration") or 0.0),
            "text": result["text"].strip(),
        }]

    fmt = fmt.lower()
    fmt_time = _fmt_vtt_time if fmt == "vtt" else _fmt_srt_time
    lines: list[str] = []
    if fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
    for idx, cue in enumerate(cues, start=1):
        if fmt == "srt":
            lines.append(str(idx))
        lines.append(f"{fmt_time(cue['start'])} --> {fmt_time(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")
    return "\n".join(lines)


def _record_audio_request(model_id: str) -> None:
    """Record audio request count without treating bytes/chars as tokens."""
    try:
        get_server_metrics().record_request_complete(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model_id=model_id,
        )
    except Exception as exc:
        logger.warning("Failed to record audio metrics for %s: %s", model_id, exc)


async def _read_upload(file: UploadFile) -> bytes:
    """Read an uploaded file in chunks, bailing early if it exceeds the limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio file exceeds maximum allowed size "
                    f"({MAX_AUDIO_UPLOAD_BYTES} bytes)"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_ref_audio_base64(request: AudioSpeechRequest) -> Optional[bytes]:
    """Validate and decode optional base64 ref_audio from a TTS request."""
    if request.ref_audio is None:
        return None

    if not request.ref_text:
        raise HTTPException(
            status_code=400,
            detail="'ref_text' is required when 'ref_audio' is provided "
            "(must be the transcript of the reference audio)",
        )
    if len(request.ref_audio) > MAX_REF_AUDIO_BASE64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"ref_audio exceeds maximum allowed size "
                f"({MAX_REF_AUDIO_BASE64_BYTES} bytes base64, "
                f"~60 seconds of audio)"
            ),
        )
    try:
        return base64.b64decode(request.ref_audio, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 encoding in 'ref_audio' field",
        )


def _write_ref_audio_tempfile(audio_bytes: Optional[bytes]) -> Optional[str]:
    """Persist decoded ref audio to a temp file if present."""
    if audio_bytes is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(audio_bytes)
        return tmp.name
    finally:
        tmp.close()


def _cleanup_tempfile(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _resolve_tts_streaming_interval(request: AudioSpeechRequest) -> float:
    """Return a native TTS streaming interval that is safe for mlx-audio."""
    if request.streaming_interval is None:
        return DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS

    interval = request.streaming_interval
    if (
        not math.isfinite(interval)
        or interval < MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "'streaming_interval' must be at least "
                f"{MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS} seconds"
            ),
        )
    return interval


def _split_tts_text(text: str, max_chars: int = 300) -> list[str]:
    """Split TTS input into conservative sentence-like chunks."""
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush_current()
            parts = re.split(r"(?<=[,;:，；：])\s*", sentence)
            parts = [p.strip() for p in parts if p and p.strip()]
            buffer = ""
            for part in parts or [sentence]:
                while len(part) > max_chars:
                    if buffer:
                        chunks.append(buffer.strip())
                        buffer = ""
                    chunks.append(part[:max_chars].strip())
                    part = part[max_chars:].strip()
                if not part:
                    continue
                candidate = f"{buffer} {part}".strip() if buffer else part
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = part
            if buffer:
                chunks.append(buffer.strip())
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            flush_current()
            current = sentence
        else:
            current = candidate

    flush_current()
    return chunks or [text]


async def _stream_speech_response(
    engine,
    request: AudioSpeechRequest,
    ref_audio_path: Optional[str],
    streaming_interval: float,
) -> AsyncIterator[bytes]:
    """Stream sentence-level TTS as a single WAV header plus PCM chunks."""
    try:
        if (
            hasattr(engine, "supports_native_tts_streaming")
            and engine.supports_native_tts_streaming()
            and hasattr(engine, "stream_synthesize_pcm")
        ):
            logger.info(
                "TTS native streaming start: model=%s, text_len=%d, voice=%s",
                request.model, len(request.input), request.voice,
            )
            stream_format: Optional[tuple[int, int, int]] = None
            try:
                async for sample_rate, channels, sample_width, pcm_bytes in engine.stream_synthesize_pcm(
                    request.input,
                    voice=request.voice,
                    speed=request.speed,
                    instructions=request.instructions,
                    ref_audio=ref_audio_path,
                    ref_text=request.ref_text,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    max_tokens=request.max_tokens,
                    streaming_interval=streaming_interval,
                ):
                    fmt = (sample_rate, channels, sample_width)
                    if stream_format is None:
                        stream_format = fmt
                        yield wav_header(
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_width=sample_width,
                        )
                    elif fmt != stream_format:
                        raise RuntimeError(
                            "Inconsistent native streaming PCM format: "
                            f"expected {stream_format}, got {fmt}"
                        )
                    if pcm_bytes:
                        yield pcm_bytes
            except NotImplementedError:
                if stream_format is not None:
                    raise
                logger.info(
                    "TTS native streaming unavailable at runtime; falling back "
                    "to segmented synthesis: model=%s",
                    request.model,
                )
            else:
                return

        segments = _split_tts_text(request.input)
        logger.info(
            "TTS streaming start: model=%s, text_len=%d, segments=%d, voice=%s",
            request.model, len(request.input), len(segments), request.voice,
        )

        stream_format: Optional[tuple[int, int, int]] = None
        for idx, segment in enumerate(segments, start=1):
            wav_bytes = await engine.synthesize(
                segment,
                voice=request.voice,
                speed=request.speed,
                instructions=request.instructions,
                ref_audio=ref_audio_path,
                ref_text=request.ref_text,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                max_tokens=request.max_tokens,
            )
            sample_rate, channels, sample_width, pcm_bytes = wav_bytes_to_pcm_frames(wav_bytes)
            fmt = (sample_rate, channels, sample_width)
            if stream_format is None:
                stream_format = fmt
                yield wav_header(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
            elif fmt != stream_format:
                raise RuntimeError(
                    "Inconsistent WAV format across TTS segments: "
                    f"expected {stream_format}, got {fmt}"
                )
            logger.debug(
                "TTS streaming segment %d/%d: text_len=%d, pcm_bytes=%d",
                idx, len(segments), len(segment), len(pcm_bytes),
            )
            if pcm_bytes:
                yield pcm_bytes
    finally:
        _cleanup_tempfile(ref_audio_path)


async def _stream_with_prefetched_chunk(
    first_chunk: bytes,
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Yield a chunk fetched before response headers, then the rest of the stream."""
    try:
        yield first_chunk
        async for chunk in stream:
            yield chunk
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    min_p: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    repetition_context_size: Optional[int] = Form(None),
    max_tokens: Optional[int] = Form(None),
    word_timestamps: bool = Form(False),
    n: int = Form(1),
):
    """OpenAI-compatible audio transcription endpoint (Speech-to-Text).

    ``prompt`` is forwarded to the underlying STT backend as a transcription-
    biasing context (Whisper's ``initial_prompt``, Qwen2-Audio's ``prompt``,
    etc.) — useful for steering domain vocabulary, proper nouns, product
    names or stylistic preferences. Backends without any prompt-style kwarg
    drop it silently.

    ``text`` is a reference transcript for forced-alignment models like
    Qwen3-ForcedAligner. Calling this endpoint with ``model=<aligner>`` +
    ``text=<known transcript>`` returns word/char-level timestamps. Without
    a ``text`` the aligner model returns 400.

    ``word_timestamps`` is an oMLX extension. For Whisper-family models it
    exposes mlx-audio's native word-level alignment (each segment in the
    response gets a ``words`` array of ``{word, start, end, probability}``).
    For Qwen3-ASR and other backends without native word-level alignment,
    if the model has ``aligner_model`` set in its ModelSettings, the server
    automatically runs that aligner (e.g. Qwen3-ForcedAligner-0.6B-4bit)
    on the (audio, ASR transcript) pair and merges the result into
    ``segments[0].words``. Default False preserves the existing response
    shape for every current caller.

    ``temperature``, ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``,
    ``repetition_context_size`` are forwarded to the backend's sampler when
    non-default. Useful for breaking decoder degenerate loops on long
    silent/repetitive audio — e.g. Qwen3-ASR on mono channels can emit
    hundreds of repeated tokens; ``repetition_penalty=1.2`` with
    ``repetition_context_size=64`` is a typical fix. Qwen3-ASR and
    Qwen2-Audio accept all of these natively; Whisper has ``**decode_options``
    so unknown kwargs are silently ignored.

    ``response_format`` supports ``json`` (default), ``verbose_json``,
    ``text``, ``srt`` and ``vtt``. SRT/VTT subtitle output auto-enables
    word_timestamps so the cue pacing isn't a single giant block; cues
    are packed by sentence-ending punctuation, 40-char text limit, or
    5-second duration cap, whichever comes first. When no word-level
    data is available the output falls back to one cue per ASR segment.

    ``n`` must be 1. Multi-candidate transcription is rejected with 400.

    ``max_tokens`` is an oMLX extension that raises the underlying model's
    output cap. Useful for long audio with models like VibeVoice-ASR whose
    mlx-audio default (8192) truncates ~24 min files. When omitted, the
    model's own default applies.
    """
    if n != 1:
        raise HTTPException(
            status_code=400,
            detail="n must be 1; multi-candidate transcription is not supported.",
        )
    response_format = (response_format or "json").lower().strip()
    if response_format not in ("json", "verbose_json", "text", "srt", "vtt"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown response_format='{response_format}'. "
                "Supported: json, verbose_json, text, srt, vtt."
            ),
        )
    # SRT/VTT need word-level timestamps to produce useful subtitle pacing —
    # otherwise we'd emit one giant cue per ASR segment. Auto-enable when
    # the caller asked for subtitles unless they explicitly opted out.
    if response_format in ("srt", "vtt") and not word_timestamps:
        word_timestamps = True
    from omlx.engine.stt import STTEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, STTEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a speech-to-text model",
        )

    # Save uploaded file to a temp path so the engine can open it by path.
    # Remap video container extensions to .m4a so mlx-audio routes them
    # through ffmpeg instead of miniaudio (which can't decode containers).
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    if suffix.lower() in _VIDEO_CONTAINERS:
        suffix = ".m4a"
    tmp_path = None
    try:
        content = await _read_upload(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        # Effective max_tokens precedence: request > per-model setting (if any) >
        # model's own ``generate(max_tokens=...)`` default. The per-model lookup
        # mirrors how chat completions reads ModelSettings.max_tokens for LLMs;
        # for STT, settings.json's ``max_tokens`` (e.g. raised to 65536 for
        # VibeVoice-ASR) becomes the durable default for that model.
        effective_max_tokens = max_tokens
        if effective_max_tokens is None:
            sm = _get_settings_manager()
            if sm is not None:
                try:
                    ms = sm.get_settings(resolved_model)
                    if ms is not None and getattr(ms, "max_tokens", None) is not None:
                        effective_max_tokens = ms.max_tokens
                except Exception:
                    pass

        transcribe_kwargs: dict = {"language": language}
        if effective_max_tokens is not None:
            transcribe_kwargs["max_tokens"] = effective_max_tokens
        if word_timestamps:
            transcribe_kwargs["word_timestamps"] = True
        if prompt:
            transcribe_kwargs["prompt"] = prompt
        # Forward `temperature` only when > 0. Most STT backends interpret
        # temperature=0 as greedy (their default), and several reject the
        # kwarg entirely; passing 0 just adds noise. >0 is the case that
        # matters — used to break decoder degenerate loops (e.g. Qwen3-ASR
        # on silent/quasi-silent mono input emits hundreds of repeated "嗯").
        if temperature is not None and temperature > 0:
            transcribe_kwargs["temperature"] = temperature
        # Sampling tail (Qwen3-ASR / Qwen2-Audio accept all of these natively;
        # Whisper has **decode_options so unknown kwargs are silently dropped).
        # repetition_penalty + repetition_context_size is the typical fix for
        # mono-channel long-silence degeneration on Qwen3-ASR.
        if top_p is not None:
            transcribe_kwargs["top_p"] = top_p
        if top_k is not None:
            transcribe_kwargs["top_k"] = top_k
        if min_p is not None:
            transcribe_kwargs["min_p"] = min_p
        if repetition_penalty is not None:
            transcribe_kwargs["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            transcribe_kwargs["repetition_context_size"] = repetition_context_size
        if text is not None:
            transcribe_kwargs["text"] = text

        # Detect aligner model so we can either run alignment (text given) or
        # reject cleanly (text missing) instead of crashing on a missing
        # positional argument deep inside mlx-audio.
        is_aligner = False
        try:
            inner = getattr(getattr(engine, "_model", None), "_model", None)
            is_aligner = type(inner).__name__ == "ForcedAlignerModel"
        except Exception:
            pass
        if is_aligner and text is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{resolved_model}' is a forced aligner and requires "
                    "a 'text' form field with the reference transcript. To get "
                    "word-level timestamps automatically from audio alone, call "
                    "an ASR model with word_timestamps=true; the server will "
                    "auto-chain its configured aligner."
                ),
            )

        result = await engine.transcribe(tmp_path, **transcribe_kwargs)

        # Auto-chain: when caller asked for word_timestamps on an ASR that has
        # an aligner companion configured (ModelSettings.aligner_model), run
        # the aligner on the (same audio, ASR transcript) pair and graft the
        # word-level timestamps into segments[0].words.
        if (
            word_timestamps
            and not is_aligner
            and result.get("text")
            and not (result.get("segments") and result["segments"][0].get("words"))
        ):
            aligner_name = None
            sm = _get_settings_manager()
            if sm is not None:
                try:
                    ms = sm.get_settings(resolved_model)
                    if ms is not None:
                        aligner_name = getattr(ms, "aligner_model", None)
                except Exception:
                    pass
            if aligner_name:
                try:
                    aligner_resolved = _resolve_model(aligner_name)
                    aligner_engine = await pool.get_engine(aligner_resolved)
                    if isinstance(aligner_engine, STTEngine):
                        align_kwargs: dict = {"language": language}
                        if effective_max_tokens is not None:
                            align_kwargs["max_tokens"] = effective_max_tokens
                        align_kwargs["text"] = result["text"]
                        align_result = await aligner_engine.transcribe(
                            tmp_path, **align_kwargs
                        )
                        words = align_result.get("words") or []
                        if words:
                            segments = result.get("segments") or []
                            if not segments:
                                segments = [{
                                    "text": result["text"],
                                    "language": result.get("language"),
                                    "start": float(min(
                                        (w["start"] for w in words), default=0.0
                                    )),
                                    "end": float(max(
                                        (w["end"] for w in words),
                                        default=result.get("duration", 0.0),
                                    )),
                                }]
                            segments[0] = {**segments[0], "words": words}
                            result["segments"] = segments
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Aligner companion %s failed for %s: %s — returning "
                        "transcript without word timestamps",
                        aligner_name, resolved_model, exc,
                    )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    _record_audio_request(resolved_model)

    # response_format=text: OpenAI spec returns just the transcript body as
    # text/plain with no envelope, segments, or metadata.
    if response_format == "text":
        return PlainTextResponse(
            content=result.get("text", ""),
            media_type="text/plain; charset=utf-8",
        )

    # SRT / VTT — assemble subtitle cues from segments/words.
    if response_format in ("srt", "vtt"):
        body = _build_subtitle(result, fmt=response_format)
        return PlainTextResponse(
            content=body,
            media_type=("text/srt" if response_format == "srt"
                        else "text/vtt") + "; charset=utf-8",
        )

    # json (default) and verbose_json both return the same shape today —
    # OpenAI's verbose_json is a superset of json with segments/duration,
    # and we always include those when the backend exposes them. Clients
    # asking for plain json simply ignore the extra fields.
    segments = result.get("segments") or None

    return AudioTranscriptionResponse(
        text=result.get("text", ""),
        language=result.get("language"),
        duration=result.get("duration"),
        segments=segments,
    )


@router.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """OpenAI-compatible text-to-speech endpoint."""
    from omlx.engine.tts import TTSEngine
    from omlx.exceptions import ModelNotFoundError

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="'input' field must not be empty")
    streaming_interval = DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    if request.stream:
        if request.response_format not in (None, "wav"):
            raise HTTPException(
                status_code=400,
                detail="Streaming TTS currently only supports response_format='wav'",
            )
        streaming_interval = _resolve_tts_streaming_interval(request)

    audio_bytes = _decode_ref_audio_base64(request)

    pool = _get_engine_pool()
    resolved_model = _resolve_model(request.model)

    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, TTSEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a text-to-speech model",
        )

    ref_audio_path = _write_ref_audio_tempfile(audio_bytes)

    if request.stream:
        stream = _stream_speech_response(
            engine,
            request,
            ref_audio_path,
            streaming_interval,
        )
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration as exc:
            raise HTTPException(
                status_code=500,
                detail="TTS streaming produced no audio output",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return StreamingResponse(
            _stream_with_prefetched_chunk(first_chunk, stream),
            media_type="audio/wav",
        )

    try:
        wav_bytes = await engine.synthesize(
            request.input,
            voice=request.voice,
            speed=request.speed,
            instructions=request.instructions,
            ref_audio=ref_audio_path,
            ref_text=request.ref_text,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup_tempfile(ref_audio_path)

    _record_audio_request(resolved_model)

    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/v1/audio/process")
async def process_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """Audio processing endpoint (speech enhancement, source separation, STS).

    Accepts a multipart audio file upload and a model identifier, processes
    the audio through an STS engine (e.g. DeepFilterNet, MossFormer2,
    SAMAudio, LFM2.5-Audio), and returns WAV bytes of the processed audio.
    """
    from omlx.engine.sts import STSEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, STSEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a speech-to-speech / audio processing model",
        )

    # Save uploaded file to a temp path so the engine can open it by path.
    # Remap video container extensions to .m4a so mlx-audio routes them
    # through ffmpeg instead of miniaudio (which can't decode containers).
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    if suffix.lower() in _VIDEO_CONTAINERS:
        suffix = ".m4a"
    tmp_path = None
    try:
        content = await _read_upload(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        wav_bytes = await engine.process(tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    _record_audio_request(resolved_model)

    return Response(content=wav_bytes, media_type="audio/wav")
