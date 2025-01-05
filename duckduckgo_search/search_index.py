from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from whoosh import writing
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, DATETIME, KEYWORD, NUMERIC, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.index import FileIndex, create_in, exists_in, open_dir
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import Term

from .document_processor import ProcessedDocument
from .exceptions import SearchIndexException
from .utils import json_dumps

logger = logging.getLogger(__name__)

class SearchIndex:
    """Manages document indexing and searching using Whoosh and BM25."""
    
    def __init__(
        self,
        index_path: str | Path,
        create_if_missing: bool = True,
        vector_cache_size: int = 1000
    ) -> None:
        """Initialize search index.
        
        Args:
            index_path: Path to store the search index
            create_if_missing: Create index if it doesn't exist
            vector_cache_size: Size of vector cache for similarity search
        """
        self.index_path = Path(index_path)
        self.vector_cache_size = vector_cache_size
        
        # Create schema for document index
        self.schema = Schema(
            path=ID(stored=True, unique=True),
            filename=ID(stored=True),
            content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            file_type=ID(stored=True),
            extension=ID(stored=True),
            mime_type=ID(stored=True),
            size=NUMERIC(stored=True),
            created_date=DATETIME(stored=True),
            modified_date=DATETIME(stored=True),
            processed_date=DATETIME(stored=True),
            keywords=KEYWORD(stored=True, commas=True, lowercase=True),
            language=ID(stored=True),
            metadata=TEXT(stored=True),
            summary=TEXT(stored=True)
        )
        
        # Initialize index
        self._initialize_index(create_if_missing)
        
        # Initialize vector search components
        self._vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            max_features=10000
        )
        self._vector_cache: dict[str, np.ndarray] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._document_lookup: dict[int, str] = {}
        
    def _initialize_index(self, create_if_missing: bool) -> None:
        """Initialize or open the search index."""
        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize storage
            self.storage = FileStorage(str(self.index_path))
            
            if not exists_in(str(self.index_path)):
                if create_if_missing:
                    self.index = create_in(
                        str(self.index_path),
                        self.schema
                    )
                else:
                    raise SearchIndexException(
                        f"Index not found at {self.index_path}"
                    )
            else:
                self.index = open_dir(str(self.index_path))
                
        except Exception as e:
            raise SearchIndexException(f"Failed to initialize index: {e}")
            
    def add_document(self, processed_doc: ProcessedDocument) -> None:
        """Add a processed document to the index.
        
        Args:
            processed_doc: ProcessedDocument object to index
        """
        try:
            writer = self.index.writer()
            
            try:
                # Prepare document fields
                doc_fields = {
                    "path": str(processed_doc.file_info.path),
                    "filename": processed_doc.file_info.name,
                    "content": processed_doc.content,
                    "file_type": processed_doc.file_info.content_type,
                    "extension": processed_doc.file_info.extension,
                    "mime_type": processed_doc.file_info.content_type,
                    "size": processed_doc.file_info.size,
                    "created_date": processed_doc.file_info.created,
                    "modified_date": processed_doc.file_info.modified,
                    "processed_date": processed_doc.processed_date,
                    "keywords": ",".join(processed_doc.keywords),
                    "language": processed_doc.language,
                    "metadata": json_dumps(processed_doc.metadata),
                    "summary": processed_doc.summary
                }
                
                # Add or update document
                writer.update_document(**doc_fields)
                
                # Update vector cache
                self._update_vector_cache(
                    str(processed_doc.file_info.path),
                    processed_doc.content
                )
                
            finally:
                writer.commit()
                
        except Exception as e:
            raise SearchIndexException(f"Failed to add document to index: {e}")
            
    def remove_document(self, file_path: str | Path) -> None:
        """Remove a document from the index.
        
        Args:
            file_path: Path of document to remove
        """
        try:
            writer = self.index.writer()
            try:
                # Remove from Whoosh index
                writer.delete_by_term('path', str(file_path))
                
                # Remove from vector cache
                self._vector_cache.pop(str(file_path), None)
                
                # Update document lookup
                self._rebuild_document_lookup()
                
            finally:
                writer.commit()
                
        except Exception as e:
            raise SearchIndexException(
                f"Failed to remove document from index: {e}"
            )
            
    def _update_vector_cache(
        self,
        file_path: str,
        content: str
    ) -> None:
        """Update the vector cache for a document."""
        try:
            # Convert content to vector
            vector = self._vectorizer.fit_transform([content]).toarray()[0]
            
            # Add to cache
            self._vector_cache[file_path] = vector
            
            # Maintain cache size
            if len(self._vector_cache) > self.vector_cache_size:
                # Remove oldest entries
                remove_count = len(self._vector_cache) - self.vector_cache_size
                for key in sorted(self._vector_cache.keys())[:remove_count]:
                    self._vector_cache.pop(key)
                    
            # Rebuild BM25 index
            self._rebuild_bm25_index()
            
        except Exception as e:
            logger.error(f"Error updating vector cache: {e}")
            
    def search(
        self,
        query: str,
        search_type: str = 'keyword',
        fields: Optional[list[str]] = None,
        limit: int = 10,
        min_score: float = 0.1,
        **filters: Any
    ) -> list[dict[str, Any]]:
        """Search the index using specified search type.
        
        Args:
            query: Search query
            search_type: Type of search ('keyword', 'semantic', 'combined')
            fields: Fields to search in (default: content and summary)
            limit: Maximum number of results
            min_score: Minimum relevance score
            **filters: Field-specific filters
            
        Returns:
            List of search results with scores
        """
        if not query.strip():
            return []
            
        try:
            if search_type == 'semantic':
                return self._semantic_search(
                    query, limit, min_score, filters
                )
            elif search_type == 'combined':
                return self._combined_search(
                    query, fields, limit, min_score, filters
                )
            else:  # keyword search
                return self._keyword_search(
                    query, fields, limit, min_score, filters
                )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
            
    def _keyword_search(
        self,
        query: str,
        fields: Optional[list[str]],
        limit: int,
        min_score: float,
        filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Perform keyword-based search."""
        search_fields = fields or ['content', 'summary']
        
        try:
            with self.index.searcher() as searcher:
                # Create parser
                if len(search_fields) > 1:
                    parser = MultifieldParser(
                        search_fields,
                        self.schema
                    )
                else:
                    parser = QueryParser(
                        search_fields[0],
                        self.schema
                    )
                
                # Parse query
                query_obj = parser.parse(query)
                
                # Apply filters
                filter_queries = []
                for field, value in filters.items():
                    if isinstance(value, (list, tuple, set)):
                        # Multiple values for field
                        filter_queries.extend(
                            Term(field, str(v)) for v in value
                        )
                    else:
                        filter_queries.append(
                            Term(field, str(value))
                        )
                
                # Perform search
                results = searcher.search(
                    query_obj,
                    limit=limit,
                    filter=filter_queries
                )
                
                # Format results
                search_results = []
                for hit in results:
                    if hit.score >= min_score:
                        result = dict(hit)
                        result['score'] = hit.score
                        result['highlights'] = hit.highlights('content')
                        search_results.append(result)
                        
                return search_results
                
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
            
    def _semantic_search(
        self,
        query: str,
        limit: int,
        min_score: float,
        filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Perform semantic similarity search."""
        try:
            if not self._bm25:
                return []
                
            # Get query vector
            query_vector = self._vectorizer.transform([query]).toarray()[0]
            
            # Calculate similarities
            similarities = []
            for path, doc_vector in self._vector_cache.items():
                similarity = self._cosine_similarity(
                    query_vector,
                    doc_vector
                )
                similarities.append((path, similarity))
                
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply filters and get results
            results = []
            with self.index.searcher() as searcher:
                for path, score in similarities[:limit]:
                    if score < min_score:
                        continue
                        
                    # Check filters
                    doc = searcher.document(path=path)
                    if doc and self._matches_filters(doc, filters):
                        doc['score'] = float(score)
                        results.append(doc)
                        
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
            
    def _combined_search(
        self,
        query: str,
        fields: Optional[list[str]],
        limit: int,
        min_score: float,
        filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Perform combined keyword and semantic search."""
        try:
            # Get both result sets
            keyword_results = self._keyword_search(
                query, fields, limit * 2, min_score / 2, filters
            )
            semantic_results = self._semantic_search(
                query, limit * 2, min_score / 2, filters
            )
            
            # Combine and deduplicate results
            seen_paths = set()
            combined_results = []
            
            for results in (keyword_results, semantic_results):
                for result in results:
                    path = result['path']
                    if path not in seen_paths:
                        seen_paths.add(path)
                        combined_results.append(result)
                        
            # Sort by combined score
            combined_results.sort(
                key=lambda x: x['score'],
                reverse=True
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Combined search error: {e}")
            return []
            
    def _matches_filters(
        self,
        doc: dict[str, Any],
        filters: dict[str, Any]
    ) -> bool:
        """Check if document matches all filters."""
        for field, value in filters.items():
            if field not in doc:
                return False
                
            if isinstance(value, (list, tuple, set)):
                if doc[field] not in value:
                    return False
            elif doc[field] != value:
                return False
        return True
    
    def get_document(self, file_path: str | Path) -> Optional[dict[str, Any]]:
        """Retrieve a document from the index by path."""
        try:
            with self.index.searcher() as searcher:
                return searcher.document(path=str(file_path))
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
            
    def _cosine_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def optimize(self) -> None:
        """Optimize the index."""
        try:
            writer = self.index.writer()
            try:
                writer.commit(optimize=True)
            except Exception as e:
                writer.cancel()
                raise SearchIndexException(f"Optimization failed: {e}")
                
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            
    def get_statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        try:
            with self.index.searcher() as searcher:
                stats = {
                    'document_count': searcher.doc_count(),
                    'last_modified': datetime.fromtimestamp(
                        self.index_path.stat().st_mtime
                    ),
                    'index_size': sum(
                        f.stat().st_size
                        for f in self.index_path.rglob('*')
                        if f.is_file()
                    ),
                    'fields': list(self.schema.names()),
                    'vector_cache_size': len(self._vector_cache),
                    'has_bm25_index': self._bm25 is not None
                }
                return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
            
    def clear(self) -> None:
        """Clear the entire index."""
        try:
            writer = self.index.writer()
            try:
                # Remove all documents
                writer.commit(mergetype=writing.CLEAR)
                
                # Clear caches
                self._vector_cache.clear()
                self._document_lookup.clear()
                self._bm25 = None
                
            except Exception as e:
                writer.cancel()
                raise SearchIndexException(f"Failed to clear index: {e}")
                
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            
    def save_state(self, path: str | Path) -> None:
        """Save vector cache and other state information."""
        try:
            state = {
                'vector_cache': self._vector_cache,
                'document_lookup': self._document_lookup,
                'vectorizer': self._vectorizer
            }
            
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            
    def load_state(self, path: str | Path) -> None:
        """Load saved state information."""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            self._vector_cache = state['vector_cache']
            self._document_lookup = state['document_lookup']
            self._vectorizer = state['vectorizer']
            
            # Rebuild BM25 index
            self._rebuild_bm25_index()
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            
    def iter_documents(
        self,
        batch_size: int = 100
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Iterate through all documents in the index."""
        try:
            with self.index.searcher() as searcher:
                total_docs = searcher.doc_count()
                current_batch = []
                
                for i, doc in enumerate(searcher.documents()):
                    current_batch.append(doc)
                    
                    if len(current_batch) >= batch_size or i == total_docs - 1:
                        yield current_batch
                        current_batch = []
                        
        except Exception as e:
            logger.error(f"Error iterating documents: {e}")
            
    def __enter__(self) -> SearchIndex:
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.optimize()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index."""
        try:
            # Get all documents
            with self.index.searcher() as searcher:
                documents = [
                    {
                        'path': doc['path'],
                        'content': doc['content']
                    }
                    for doc in searcher.documents()
                ]
            
            if not documents:
                self._bm25 = None
                self._document_lookup = {}
                return
                
            # Prepare corpus
            corpus = [doc['content'].split() for doc in documents]
            self._document_lookup = {
                i: doc['path']
                for i, doc in enumerate(documents)
            }
            
            # Create BM25 index
            self._bm25 = BM25Okapi(corpus)
            
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")
            self._bm25 = None
            self._document_lookup = {}
    
    def _rebuild_document_lookup(self) -> None:
        """Rebuild the document lookup table."""
        try:
            with self.index.searcher() as searcher:
                self._document_lookup = {
                    i: doc['path']
                    for i, doc in enumerate(searcher.documents())
                }
        except Exception as e:
            logger.error(f"Error rebuilding document lookup: {e}")
            self._document_lookup = {}