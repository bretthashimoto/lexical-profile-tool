"""Extract plain text from uploaded files of various formats.

Supports .txt (decoded as UTF-8), .docx (via python-docx), and .pdf (via
pypdf). The heavier parsing libraries are imported lazily inside each
extractor so that just importing this module stays cheap.

Ported verbatim from webapp/text_extract.py (the Streamlit app) -- this
logic has no Streamlit dependency, so it's shared as-is.
"""

from __future__ import annotations

import io


def extract_text(name: str, data: bytes) -> str:
    """Extract plain text from raw file bytes, based on `name`'s extension.

    Raises ValueError (with a plain-language message) for an unsupported
    extension or a file that can't be parsed as its claimed type.
    """
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "txt":
        return data.decode("utf-8", errors="ignore")
    if ext == "docx":
        return _extract_docx(name, data)
    if ext == "pdf":
        return _extract_pdf(name, data)
    raise ValueError(
        f"Can't use '{name}': unsupported file type '.{ext}'. Supported "
        f"types are .txt, .docx, and .pdf."
    )


def _extract_docx(name: str, data: bytes) -> str:
    import docx  # python-docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Couldn't read '{name}' as a .docx file: {e}") from e
    return "\n".join(p.text for p in document.paragraphs)


def _extract_pdf(name: str, data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise ValueError(f"Couldn't read '{name}' as a .pdf file: {e}") from e
