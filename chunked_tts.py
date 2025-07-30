#!/usr/bin/env python
"""Chunked TTS converter for long papers.

Usage
-----
python chunked_tts.py --text-file extracted.txt [--out saved_paper.wav]

This script assumes you already extracted the paper text (for example using
Gemini like in `paper2audio_gemini.ipynb`).  It then:
1. Splits the text into ~12 k-character chunks without cutting mid-line.
2. Calls the Gemini 2.5 TTS model for each chunk.
3. Concatenates the raw 24 kHz/16-bit PCM and writes a single WAV file.

Requires that the GEMINI_API_KEY environment variable is set.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import wave
from pathlib import Path

from google import genai
from google.genai import types

VOICE = "Kore"  # change if you want a different voice
MODEL = "gemini-2.5-flash-preview-tts"
CHUNK_CHAR_LIMIT = 10_000  # safe limit well below model context size


def wave_file(filename: str | Path, pcm: bytes, *, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    """Write PCM data to a simple mono 24 kHz WAV file."""
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def chunk_text_by_lines(text: str, char_limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """Return chunks no longer than *char_limit*, never splitting a line."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.splitlines():
        extra = len(line) + 1  # account for re-inserted newline
        if current and length + extra > char_limit:
            chunks.append("\n".join(current))
            current, length = [line], extra
        else:
            current.append(line)
            length += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def tts_chunk(client: genai.Client, chunk: str) -> bytes:
    """Generate PCM for a single chunk and return it."""
    resp = client.models.generate_content(
        model=MODEL,
        contents="Read:" + chunk,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
                )
            ),
        ),
    )
    print ('Completed TTS chunk')
    return b"".join(
        p.inline_data.data
        for p in resp.candidates[0].content.parts
        if getattr(p, "inline_data", None) and p.inline_data.data
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked Gemini TTS for long documents")
    parser.add_argument("--text-file", required=True, type=Path, help="Plain-text file to read")
    parser.add_argument("--out", default="saved_paper.wav", type=Path, help="Output WAV file")
    args = parser.parse_args()

    if not args.text_file.exists():
        sys.exit(f"Input text file '{args.text_file}' not found")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY environment variable not set")

    client = genai.Client()

    text = args.text_file.read_text()
    chunks = chunk_text_by_lines(text)
    print(f"Splitting input into {len(chunks)} chunks…")

    pcm_all = b""
    for idx, chunk in enumerate(chunks, 1):
        print(f"→ Processing chunk {idx}/{len(chunks)} (chars={len(chunk)})")
        pcm_all += tts_chunk(client, chunk)

    wave_file(args.out, pcm_all)
    print(f"Wrote {args.out} (size={len(pcm_all)/24000/2:.1f} s)")


if __name__ == "__main__":
    main() 