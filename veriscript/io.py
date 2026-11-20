"""Text extraction from .txt, .pdf and .docx inputs (module ``io.extract`` in Table 5.3)."""

from __future__ import annotations

import io as _io
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


class ExtractionError(ValueError):
    """Raised when text cannot be extracted from an input."""


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("could not decode text file")


def _pdf_text(data: bytes) -> str:
    import pdfplumber

    try:
        with pdfplumber.open(_io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
    except Exception as exc:  # pdfplumber raises many types
        raise ExtractionError(f"unreadable PDF: {exc}") from exc
    return "\n\n".join(pages)


def _docx_text(data: bytes) -> str:
    import docx

    try:
        document = docx.Document(_io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"unreadable DOCX: {exc}") from exc
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_bytes(data: bytes, filename: str) -> str:
    """Extract text from in-memory file contents; the suffix of ``filename`` selects the parser."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md", ""):
        return _decode(data)
    if suffix == ".pdf":
        return _pdf_text(data)
    if suffix == ".docx":
        return _docx_text(data)
    raise ExtractionError(f"unsupported file type: {suffix}")


def extract_text(path: str | Path) -> str:
    """Extract UTF-8 plain text from a file on disk."""
    p = Path(path)
    if not p.is_file():
        raise ExtractionError(f"file not found: {p}")
    return extract_bytes(p.read_bytes(), p.name)


def iter_folder(folder: str | Path):
    """Yield (doc_id, text_or_None, error_or_None) for supported files in a folder."""
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                yield p.name, extract_text(p), None
            except ExtractionError as exc:
                yield p.name, None, str(exc)
