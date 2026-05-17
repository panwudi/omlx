# SPDX-License-Identifier: Apache-2.0
"""Speaker diarization backends for the STT pipeline.

Two backends are planned:

* ``energy`` — stereo channels carry one speaker each (FreeSWITCH 2-leg
  call recording, podcasts with separate mics). Compares L/R RMS energy
  per word to assign label. Zero model, zero deps beyond numpy. Implemented
  here.

* ``pyannote`` — mono or multi-speaker input where channel-based labelling
  doesn't apply (conference recordings, single-mic interviews). Wraps
  pyannote.audio's speaker-diarization-3.1 pipeline. Phase 2, separate
  session — model download + HF license + 16kHz resample + glue.

Both produce the same output shape: a list of word dicts with a
``speaker`` field populated on each entry. The audio_routes layer then
threads this label into the SRT/VTT formatter and the verbose_json
response.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def energy_diarize_words(
    stereo_audio: np.ndarray,
    sample_rate: int,
    words: list[dict],
    left_speaker: str,
    right_speaker: str,
    diarize_threshold: float = 1.3,
    pad_seconds: float = 0.05,
    overlap_label: str = "overlap",
) -> list[dict]:
    """Label each word with a speaker based on stereo L / R RMS ratio.

    Each input word has ``word`` / ``start`` / ``end`` (seconds). For
    every word we slice the stereo buffer over [start-pad, end+pad],
    compute RMS on each channel, and pick the louder side — unless the
    ratio is below ``diarize_threshold`` (interpreted as cross-talk or
    silent gap), in which case we emit ``overlap_label``.

    Args:
        stereo_audio: shape ``(samples, 2)``, L = column 0, R = column 1.
        sample_rate: original sample rate of the buffer (Hz).
        words: ASR word entries.
        left_speaker / right_speaker: label string for L / R speakers.
        diarize_threshold: ``max(L, R) / min(L, R)`` below this is
            considered indeterminate. Default 1.3.
        pad_seconds: widen the per-word window by this on each side
            before measuring energy. Default 0.05.
        overlap_label: label to assign when the ratio is too close.

    Returns:
        A new list of word dicts with a ``speaker`` field added. The input
        list is not mutated.
    """
    if stereo_audio.ndim != 2 or stereo_audio.shape[1] != 2:
        raise ValueError(
            "energy_diarize requires stereo (samples, 2); got shape "
            f"{stereo_audio.shape}"
        )
    L = stereo_audio[:, 0]
    R = stereo_audio[:, 1]
    n_samples = L.shape[0]
    pad_n = max(1, int(pad_seconds * sample_rate))

    out: list[dict] = []
    for w in words:
        try:
            start_s = float(w.get("start", 0.0))
            end_s = float(w.get("end", start_s))
        except (TypeError, ValueError):
            out.append({**w, "speaker": overlap_label})
            continue

        s = max(0, int(start_s * sample_rate) - pad_n)
        e = min(n_samples, int(end_s * sample_rate) + pad_n)
        if e <= s:
            out.append({**w, "speaker": overlap_label})
            continue

        L_rms = float(np.sqrt(np.mean(L[s:e] ** 2)))
        R_rms = float(np.sqrt(np.mean(R[s:e] ** 2)))
        hi = max(L_rms, R_rms)
        lo = max(min(L_rms, R_rms), 1e-9)
        ratio = hi / lo

        if ratio < diarize_threshold:
            label = overlap_label
        elif L_rms >= R_rms:
            label = left_speaker
        else:
            label = right_speaker
        out.append({**w, "speaker": label})

    return out


def mono_mix_for_asr(stereo_audio: np.ndarray) -> np.ndarray:
    """Down-mix stereo to mono with clip protection.

    Simple ``(L + R) / 2`` averaging. Avoids the +6 dB clipping that
    naive ``L + R`` would cause if both channels are loud at the same
    moment. The diarization layer keeps the original stereo buffer for
    RMS analysis; only the ASR-bound copy is mixed down.
    """
    if stereo_audio.ndim == 1:
        return stereo_audio
    if stereo_audio.ndim != 2 or stereo_audio.shape[1] not in (1, 2):
        raise ValueError(
            f"mono_mix expects (samples,) or (samples, 1|2); got shape "
            f"{stereo_audio.shape}"
        )
    if stereo_audio.shape[1] == 1:
        return stereo_audio[:, 0]
    return (stereo_audio[:, 0] + stereo_audio[:, 1]) * 0.5
