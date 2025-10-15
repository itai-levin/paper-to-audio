from __future__ import annotations

"""Utility functions for extracting narratable text from a PDF with caching.

The main helper ``extract_text_from_pdf`` will:
1. Derive a companion ``.txt`` file path from a target output audio filename (or allow one to be
   specified explicitly).
2. Check if that ``.txt`` file already exists. If so, it just returns its contents, avoiding an
   expensive LLM call.
3. Otherwise, it calls the OpenAI chat completion endpoint with the provided prompt and PDF bytes,
   saves the returned text, and returns the text so that downstream code (e.g. TTS conversion) can
   continue unchanged.

Example
-------
```python
from pathlib import Path
from text_extraction import extract_text_from_pdf

pdf = Path("~/Downloads/paper.pdf").expanduser()
audio_out = Path("paper_reading_20251014_120000.wav")

prompt = (
    "Print out all of the text in the paper that a narrator would read aloud. "
    "Include the title and section headings. Do not include citation numbers. "
    "Do not include acknowledgements, bibliography, reporting summary, or competing interests. "
    "Do not preface with any text other than: This is an automated voice reading <The title of the paper> "
    "by <The first author> et al."
)

text = extract_text_from_pdf(pdf, audio_out, prompt)
```
"""

from pathlib import Path
import base64
from typing import Optional
import openai

# Add default narrator extraction prompt
DEFAULT_PROMPT: str = (
    "Print out all of the text in the paper that a narrator would read aloud. "
    "Include the title and section headings. Do not include citation numbers. "
    "Do not include acknowledgements, bibliography, reporting summary, or competing interests. "
    "Do not preface with any text other than: This is an automated voice reading <The title of the paper> "
    "by <The first author> et al."
)

__all__ = ["extract_text_from_pdf", "DEFAULT_PROMPT"]


def _default_txt_path(output_audio: Path) -> Path:
    """Return the *.txt* path derived from the audio filename."""
    return output_audio.with_suffix(".txt")


def extract_text_from_pdf(
    pdf_path: str | Path,
    output_audio_file: str | Path,
    prompt: str | None = None,
    *,
    model: str = "gpt-4.1-mini-2025-04-14",
    client: Optional[openai.OpenAI] = None,
    txt_path: str | Path | None = None,
) -> str:
    """Return narratable text for *pdf_path*, caching to a .txt file.

    Parameters
    ----------
    pdf_path
        Path to the PDF.
    output_audio_file
        Intended audio output filename. Determines default location of the cached ``.txt``.
    prompt
        User prompt instructing the model how to extract text.
    model
        Text extraction model to use.
    client
        Optional pre-constructed ``openai.OpenAI`` instance.
    txt_path
        Optional explicit txt path. If *None*, derives it by replacing the suffix of
        *output_audio_file* with ``.txt``.
    """
    pdf_path = Path(pdf_path)
    if txt_path is None:
        txt_path = _default_txt_path(Path(output_audio_file))
    txt_path = Path(txt_path)

    if txt_path.exists():
        print(f"Using cached extracted text: {txt_path}")
        return txt_path.read_text(encoding="utf-8")

    # Fall back to the library default prompt if none supplied
    if prompt is None:
        prompt = DEFAULT_PROMPT

    if client is None:
        client = openai.OpenAI()

    with pdf_path.open("rb") as fh:
        pdf_bytes = fh.read()

    b64_data = base64.b64encode(pdf_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{b64_data}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    extracted_text: str = response.choices[0].message.content
    txt_path.write_text(extracted_text, encoding="utf-8")
    print(f"Saved extracted text to {txt_path}")
    return extracted_text
