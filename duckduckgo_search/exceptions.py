class DuckDuckGoSearchException(Exception):
    """Base exception class for duckduckgo_search."""

class RatelimitException(DuckDuckGoSearchException):
    """Raised for rate limit exceeded errors during API requests."""

class TimeoutException(DuckDuckGoSearchException):
    """Raised for timeout errors during API requests."""

class ConversationLimitException(DuckDuckGoSearchException):
    """Raised for conversation limit during API requests to AI endpoint."""

class FileSystemException(DuckDuckGoSearchException):
    """Raised for file system related errors."""

class DocumentProcessingException(DuckDuckGoSearchException):
    """Raised for errors during document processing."""

class SearchIndexException(DuckDuckGoSearchException):
    """Raised for errors related to search indexing."""

class UnifiedSearchException(DuckDuckGoSearchException):
    """Raised for errors in unified search operations."""

class AIAnalysisException(DuckDuckGoSearchException):
    """Raised for errors during AI analysis of search results."""