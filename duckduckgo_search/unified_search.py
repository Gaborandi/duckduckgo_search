return "".join(context)
        
    def suggest_queries(
        self,
        query: str,
        max_suggestions: int = 5
    ) -> list[str]:
        """Generate query suggestions based on search history."""
        try:
            # Get related terms from index
            related_terms = set()
            results = self.search_index.search(
                query,
                search_type='keyword',
                limit=10
            )
            
            for result in results:
                # Extract keywords
                keywords = result.get('keywords', '').split(',')
                related_terms.update(keywords)
                
                # Extract terms from content
                content = result.get('content', '')
                words = re.findall(r'\w+', content.lower())
                related_terms.update(words)
            
            # Filter and sort suggestions
            suggestions = []
            base_terms = set(re.findall(r'\w+', query.lower()))
            
            for term in related_terms:
                if term not in base_terms and len(term) > 2:
                    suggestions.append(f"{query} {term}")
                    
            return sorted(suggestions, key=len)[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
            
    def get_document_preview(
        self,
        file_path: str | Path,
        max_length: int = 1000
    ) -> Optional[dict[str, Any]]:
        """Get a preview of a document's content."""
        try:
            doc = self.search_index.get_document(file_path)
            if not doc:
                return None
                
            content = doc.get('content', '')
            if len(content) > max_length:
                content = content[:max_length] + '...'
                
            return {
                'title': Path(doc['path']).name,
                'content': content,
                'summary': doc.get('summary', ''),
                'keywords': doc.get('keywords', '').split(','),
                'metadata': json_loads(doc.get('metadata', '{}')),
                'modified_date': doc.get('modified_date')
            }
            
        except Exception as e:
            logger.error(f"Error getting document preview: {e}")
            return None
            
    def watch_query(
        self,
        query: str,
        callback: Callable[[SearchResult], None],
        interval: int = 300  # 5 minutes
    ) -> None:
        """Watch for new results matching a query.
        
        Args:
            query: Search query to watch
            callback: Callback function for new results
            interval: Check interval in seconds
        """
        import threading
        import time
        
        def watch_worker():
            seen_results = set()
            
            while True:
                try:
                    # Search for new results
                    results = list(self.search(
                        query,
                        include_web=True,
                        max_local_results=50,
                        max_web_results=50
                    ))
                    
                    # Check for new results
                    for result in results:
                        result_id = f"{result.source}:{result.title}"
                        if result_id not in seen_results:
                            seen_results.add(result_id)
                            callback(result)
                            
                    # Limit seen results cache
                    if len(seen_results) > 1000:
                        seen_results.clear()
                        
                except Exception as e:
                    logger.error(f"Error in watch worker: {e}")
                    
                time.sleep(interval)
                
        # Start watch thread
        thread = threading.Thread(
            target=watch_worker,
            daemon=True
        )
        thread.start()
        
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the search system."""
        try:
            return {
                'index_stats': self.search_index.get_statistics(),
                'monitored_paths': [
                    str(p) for p in self.file_tracker.root_paths
                ],
                'file_count': len(self.file_tracker._tracked_files),
                'allowed_extensions': self.file_tracker.allowed_extensions,
                'max_file_size': self.file_tracker.max_file_size
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
            
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop file monitoring
            self.file_tracker.stop_monitoring()
            
            # Optimize index
            self.search_index.optimize()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def __enter__(self) -> UnifiedSearch:
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()from __future__ import annotations

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
from .file_system import FileSystemTracker
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