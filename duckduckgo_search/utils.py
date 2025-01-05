# utils.py (replace entire file)

from __future__ import annotations
import re
from html import unescape
from typing import Any
from urllib.parse import unquote
import os
import csv
import io
import orjson
import pdfplumber
import docx
from .exceptions import DuckDuckGoSearchException

REGEX_STRIP_TAGS = re.compile("<.*?>")

def json_dumps(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()

def json_loads(obj: str | bytes) -> Any:
    return orjson.loads(obj)

def _extract_vqd(html_bytes: bytes, keywords: str) -> str:
    for c1, c1_len, c2 in (
        (b'vqd="', 5, b'"'),
        (b"vqd=", 4, b"&"),
        (b"vqd='", 5, b"'"),
    ):
        try:
            start = html_bytes.index(c1) + c1_len
            end = html_bytes.index(c2, start)
            return html_bytes[start:end].decode()
        except ValueError:
            pass
    raise DuckDuckGoSearchException(f"Could not extract vqd for {keywords}")

def _normalize(raw_html: str) -> str:
    return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""

def _normalize_url(url: str) -> str:
    return unquote(url).replace(" ", "+") if url else ""

def _expand_proxy_tb_alias(proxy: str | None) -> str | None:
    return "socks5://127.0.0.1:9150" if proxy == "tb" else proxy

def _parse_pdf(path: str) -> str:
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks)

def _parse_docx(path: str) -> str:
    doc = docx.Document(path)
    chunks = []
    for para in doc.paragraphs:
        if para.text:
            chunks.append(para.text)
    return "\n".join(chunks)

def _parse_doc(path: str) -> str:
    return _parse_docx(path)

def _parse_csv(path: str) -> str:
    content = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            line = " ".join(row)
            content.append(line)
    return "\n".join(content)

def parse_document(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pdf":
        return _parse_pdf(path)
    elif ext == ".docx":
        return _parse_docx(path)
    elif ext == ".doc":
        return _parse_doc(path)
    elif ext == ".csv":
        return _parse_csv(path)
    else:
        raise DuckDuckGoSearchException("Unsupported file format")
