from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Optional, TypeVar, Union

from .document_processor import DocumentProcessor, ProcessedDocument
from .duckduckgo_search import DDGS
from .exceptions import UnifiedSearchException
from .file_system import FileSystemTracker, FileInfo
from .search_index import SearchIndex
from .utils import json_dumps, json_loads

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class SearchResult:
    """Unified search result combining local and web results."""
    source: str  # 'local' or 'web'
    title: str
    content: str
    url: Optional[str] = None
    file_path: Optional[Path] = None
    score: float = 0.0
    metadata: dict[str, Any] = None
    highlights: Optional[str] = None
    timestamp: datetime = None

class UnifiedSearch:
    """Unified search combining local and web search capabilities with AI integration."""
    
    def __init__(
        self,
        monitored_paths: list[str | Path],
        index_path: str | Path,
        allowed_extensions: Optional[set[str]] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        headers: Optional[dict[str, str]] = None,
        proxy: Optional[str] = None,
        timeout: int = 10,
        max_workers: int = 4
    ) -> None:
        """Initialize unified search.
        
        Args:
            monitored_paths: List of paths to monitor for files
            index_path: Path to store search index
            allowed_extensions: Set of allowed file extensions
            max_file_size: Maximum file size to process
            headers: Custom headers for web requests
            proxy: Proxy for web requests
            timeout: Request timeout
            max_workers: Maximum worker threads
        """
        self.max_workers = max_workers
        
        # Initialize components
        self.file_tracker = FileSystemTracker(
            monitored_paths,
            allowed_extensions,
            max_file_size
        )
        
        self.document_processor = DocumentProcessor()
        
        self.search_index = SearchIndex(
            index_path,
            create_if_missing=True
        )
        
        self.web_search = DDGS(
            headers=headers,
            proxy=proxy,
            timeout=timeout
        )
        
        # Start file monitoring
        self.file_tracker.start_monitoring()
        
    def index_existing_files(
        self,
        callback: Optional[Callable[[str, int, int], None]] = None
    ) -> None:
        """Index all existing files in monitored paths.
        
        Args:
            callback: Optional progress callback function(file_path, current, total)
        """
        try:
            # Scan for files
            self.file_tracker.scan_existing_files()
            
            # Process and index files
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                total_files = len(self.file_tracker._tracked_files)
                
                for i, (file_path, file_info) in enumerate(
                    self.file_tracker._tracked_files.items(), 1
                ):
                    futures.append(
                        executor.submit(self._process_and_index_file, file_info)
                    )
                    
                    if callback:
                        callback(str(file_path), i, total_files)
                        
                # Wait for completion
                for future in as_completed(futures):
                    future.result()  # Raise any exceptions
                    
        except Exception as e:
            raise UnifiedSearchException(f"Failed to index existing files: {e}")
            
    def _process_and_index_file(self, file_info: FileInfo) -> None:
        """Process and index a single file."""
        try:
            # Process document
            processed_doc = self.document_processor.process_document(file_info)
            
            # Add to search index
            self.search_index.add_document(processed_doc)
            
        except Exception as e:
            logger.error(f"Error processing file {file_info.path}: {e}")
            
    def search(
        self,
        query: str,
        search_type: str = 'combined',
        include_web: bool = True,
        max_local_results: int = 10,
        max_web_results: int = 10,
        min_score: float = 0.1,
        **filters: Any
    ) -> Iterator[SearchResult]:
        """Perform unified search across local and web content.
        
        Args:
            query: Search query
            search_type: Type of local search ('keyword', 'semantic', 'combined')
            include_web: Whether to include web results
            max_local_results: Maximum local results
            max_web_results: Maximum web results
            min_score: Minimum relevance score
            **filters: Additional search filters
            
        Returns:
            Iterator of SearchResult objects
        """
        try:
            # Start local and web searches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                # Local search
                futures.append(
                    executor.submit(
                        self._local_search,
                        query,
                        search_type,
                        max_local_results,
                        min_score,
                        filters
                    )
                )
                
                # Web search
                if include_web:
                    futures.append(
                        executor.submit(
                            self._web_search,
                            query,
                            max_web_results
                        )
                    )
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        for result in results:
                            yield result
                    except Exception as e:
                        logger.error(f"Error processing search results: {e}")
                        
        except Exception as e:
            raise UnifiedSearchException(f"Search failed: {e}")
            
    def _local_search(
        self,
        query: str,
        search_type: str,
        max_results: int,
        min_score: float,
        filters: dict[str, Any]
    ) -> list[SearchResult]:
        """Perform local search."""
        try:
            results = []
            search_results = self.search_index.search(
                query,
                search_type=search_type,
                limit=max_results,
                min_score=min_score,
                **filters
            )
            
            for result in search_results:
                results.append(SearchResult(
                    source='local',
                    title=Path(result['path']).name,
                    content=result.get('content', ''),
                    file_path=Path(result['path']),
                    score=result.get('score', 0.0),
                    metadata=json_loads(result.get('metadata', '{}')),
                    highlights=result.get('highlights'),
                    timestamp=result.get('modified_date')
                ))
                
            return results
            
        except Exception as e:
            logger.error(f"Local search error: {e}")
            return []
            
    def _web_search(
        self,
        query: str,
        max_results: int
    ) -> list[SearchResult]:
        """Perform web search."""
        try:
            results = []
            web_results = self.web_search.text(
                keywords=query,
                max_results=max_results
            )
            
            for result in web_results:
                results.append(SearchResult(
                    source='web',
                    title=result['title'],
                    content=result['body'],
                    url=result['href'],
                    score=1.0,  # Web results don't have scores
                    timestamp=datetime.now()  # Use current time
                ))
                
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
            
    def analyze_results(
        self,
        query: str,
        results: list[SearchResult]
    ) -> dict[str, Any]:
        """Analyze and enhance search results using AI."""
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(query, results)
            
            # Get AI analysis
            analysis = self.web_search.chat(
                keywords=context,
                model="claude-3-haiku"
            )
            
            # Parse AI response
            try:
                return json_loads(analysis)
            except Exception:
                # If not valid JSON, return as text
                return {"summary": analysis}
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {}
            
    def _prepare_ai_context(
        self,
        query: str,
        results: list[SearchResult]
    ) -> str:
        """Prepare context for AI analysis."""
        context = [
            "Please analyze these search results and provide insights. ",
            f"Query: {query}\n\n",
            "Results:\n"
        ]
        
        for i, result in enumerate(results, 1):
            context.extend([
                f"{i}. {'File' if result.source == 'local' else 'Web'}: ",
                f"{result.title}\n",
                f"Content: {result.content[:500]}...\n\n"
            ])
            
        context.append(
            "Please provide analysis in JSON format with: "
            "summary, key_points, sources_quality, suggested_queries"
        )
        
        return "".join(context)