#!/usr/bin/env python
"""Chunked TTS converter for long papers.

Usage
-----
python chunked_tts.py --text-file extracted.txt [--out saved_paper.wav]
python chunked_tts.py --text-file extracted.txt --out saved_paper.mp3

This script assumes you already extracted the paper text (for example using
Gemini like in `paper2audio_gemini.ipynb`).  It then:
1. Splits the text into ~8 k-character chunks without cutting mid-line.
2. Calls the TTS model TTS model for each chunk.
3. Concatenates the raw 24 kHz/16-bit PCM and writes a single WAV or MP3 file.

To run from script requires that the GEMINI_API_KEY environment variable is set.
Note: For MP3 output, requires pydub library (pip install pydub) and ffmpeg.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave
import io
from pathlib import Path

import openai
from google import genai
from google.genai import types
# Only import pydub when MP3 output is requested to avoid unnecessary dependency errors
try:
    from pydub import AudioSegment  # type: ignore
except ModuleNotFoundError:
    AudioSegment = None  # will raise later if MP3 is requested

GEMINI_VOICE = "Kore"  # change if you want a different voice
GEMINI_MODEL = "gemini-2.5-flash-preview-tts"
CHUNK_CHAR_LIMIT = 8_000  # safe limit well below model context size

OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_VOICE = "ash"

def save_audio_file(filename: str | Path, pcm: bytes, *, fmt: str, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    """Write PCM data to WAV or MP3 based on *fmt* ("wav" or "mp3")."""

    filename = Path(filename)

    if fmt == "mp3":
        if AudioSegment is None:
            raise RuntimeError("pydub is required for MP3 output. Install with `pip install pydub` and ensure ffmpeg is available.")

        audio = AudioSegment(
            data=pcm,
            sample_width=sample_width,
            frame_rate=rate,
            channels=channels,
        )
        audio.export(str(filename.with_suffix('.mp3')), format="mp3", bitrate="192k")
    else:  # wav
        with wave.open(str(filename.with_suffix('.wav')), "wb") as wf:
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


def tts_chunk(client: genai.Client | openai.OpenAI, chunk: str) -> bytes:
    """Generate PCM for a single chunk and return it."""
    
    if isinstance(client, genai.Client):
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Read:" + chunk,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
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
    elif isinstance(client, openai.OpenAI):
        resp = client.audio.speech.with_raw_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_VOICE,
            input=chunk,
            instructions="You are reading a scientific journal article. Speak in a professional and engaging tone.",
            speed=1.0,
            response_format="wav",
        )

        # The raw response is a complete WAV container. Extract only the PCM frames
        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            pcm_data = wf.readframes(wf.getnframes())

        print("Completed TTS chunk")
        return pcm_data
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked Gemini TTS for long documents")
    parser.add_argument("--text-file", required=True, type=Path, help="Plain-text file to read")
    parser.add_argument("--out", default="saved_paper", type=Path, help="Output filename without extension")
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav", help="Audio output format")
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

    save_audio_file(args.out, pcm_all, fmt=args.format)
    print(f"Wrote {args.out.with_suffix('.'+args.format)} (size={len(pcm_all)/24000/2:.1f} s)")


if __name__ == "__main__":
    main() 