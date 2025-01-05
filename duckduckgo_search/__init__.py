"""Duckduckgo_search.

Search for text, documents, images, news, and perform AI analysis using
both local files and the DuckDuckGo.com search engine.
"""

import logging

from .document_processor import DocumentProcessor
from .duckduckgo_search import DDGS
from .file_system import FileSystemTracker
from .search_index import SearchIndex
from .unified_search import UnifiedSearch
from .version import __version__

__all__ = [
    "DDGS",
    "DocumentProcessor",
    "FileSystemTracker",
    "SearchIndex",
    "UnifiedSearch",
    "__version__",
    "cli"
]

# Configure null logging handler
logging.getLogger("duckduckgo_search").addHandler(logging.NullHandler())