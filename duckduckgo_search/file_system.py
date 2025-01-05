from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator, Literal, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .exceptions import FileSystemException

logger = logging.getLogger(__name__)

@dataclass
class FileInfo:
    """File information dataclass."""
    path: Path
    name: str
    extension: str
    size: int
    created: datetime
    modified: datetime
    is_binary: bool
    content_type: str

class FileSystemEventProcessor(FileSystemEventHandler):
    """Process file system events for indexing."""
    
    def __init__(self, file_tracker: FileSystemTracker) -> None:
        self.file_tracker = file_tracker
        self._ignored_directories: Set[str] = {
            '.git', '__pycache__', 'node_modules', 'venv',
            '.idea', '.vscode', '.pytest_cache'
        }

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore(event.src_path):
            return
        try:
            self.file_tracker.add_file(Path(event.src_path))
        except Exception as e:
            logger.error(f"Error processing new file {event.src_path}: {e}")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore(event.src_path):
            return
        try:
            self.file_tracker.update_file(Path(event.src_path))
        except Exception as e:
            logger.error(f"Error processing modified file {event.src_path}: {e}")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or self._should_ignore(event.src_path):
            return
        try:
            self.file_tracker.remove_file(Path(event.src_path))
        except Exception as e:
            logger.error(f"Error processing deleted file {event.src_path}: {e}")

    def _should_ignore(self, path: str) -> bool:
        """Check if the path should be ignored."""
        path_parts = Path(path).parts
        return any(
            ignored_dir in path_parts 
            for ignored_dir in self._ignored_directories
        )

class FileSystemTracker:
    """Track and monitor file system changes."""

    def __init__(
        self, 
        root_paths: list[str | Path],
        allowed_extensions: Optional[set[str]] = None,
        max_file_size: int = 100 * 1024 * 1024  # 100MB
    ) -> None:
        """Initialize FileSystemTracker.
        
        Args:
            root_paths: List of paths to monitor
            allowed_extensions: Set of allowed file extensions (e.g. {'.txt', '.pdf'})
            max_file_size: Maximum file size in bytes to process
        """
        self.root_paths = [Path(p).resolve() for p in root_paths]
        self.allowed_extensions = allowed_extensions
        self.max_file_size = max_file_size
        self._observer = Observer()
        self._event_handler = FileSystemEventProcessor(self)
        self._tracked_files: dict[Path, FileInfo] = {}

    def start_monitoring(self) -> None:
        """Start monitoring file system changes."""
        try:
            for root_path in self.root_paths:
                if not root_path.exists():
                    raise FileSystemException(f"Path does not exist: {root_path}")
                self._observer.schedule(
                    self._event_handler,
                    str(root_path),
                    recursive=True
                )
            self._observer.start()
            logger.info(f"Started monitoring paths: {self.root_paths}")
        except Exception as e:
            raise FileSystemException(f"Failed to start file system monitoring: {e}")

    def stop_monitoring(self) -> None:
        """Stop monitoring file system changes."""
        try:
            self._observer.stop()
            self._observer.join()
            logger.info("Stopped file system monitoring")
        except Exception as e:
            raise FileSystemException(f"Failed to stop file system monitoring: {e}")

    def scan_existing_files(self) -> None:
        """Scan and index existing files in monitored directories."""
        for root_path in self.root_paths:
            try:
                for file_path in self._walk_directory(root_path):
                    self.add_file(file_path)
            except Exception as e:
                logger.error(f"Error scanning directory {root_path}: {e}")

    def add_file(self, file_path: Path) -> None:
        """Add or update a file in the tracking system."""
        try:
            if not self._should_process_file(file_path):
                return
            
            file_info = self._get_file_info(file_path)
            self._tracked_files[file_path] = file_info
            logger.debug(f"Added file to tracking: {file_path}")
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")

    def update_file(self, file_path: Path) -> None:
        """Update tracked file information."""
        self.add_file(file_path)

    def remove_file(self, file_path: Path) -> None:
        """Remove a file from tracking."""
        try:
            self._tracked_files.pop(file_path, None)
            logger.debug(f"Removed file from tracking: {file_path}")
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")

    def get_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """Get information about a tracked file."""
        return self._tracked_files.get(file_path)

    def _get_file_info(self, file_path: Path) -> FileInfo:
        """Get detailed file information."""
        stat = file_path.stat()
        is_binary = self._is_binary_file(file_path)
        content_type = self._get_content_type(file_path)
        
        return FileInfo(
            path=file_path,
            name=file_path.name,
            extension=file_path.suffix.lower(),
            size=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            is_binary=is_binary,
            content_type=content_type
        )

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed based on rules."""
        try:
            if not file_path.is_file():
                return False
                
            if self.allowed_extensions and file_path.suffix.lower() not in self.allowed_extensions:
                return False
                
            if file_path.stat().st_size > self.max_file_size:
                logger.warning(f"File exceeds size limit: {file_path}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False

    def _walk_directory(self, path: Path) -> Generator[Path, None, None]:
        """Walk directory and yield file paths."""
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    yield entry
        except Exception as e:
            logger.error(f"Error walking directory {path}: {e}")

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except Exception:
            return True

    def _get_content_type(self, file_path: Path) -> str:
        """Determine file content type."""
        import magic
        try:
            return magic.from_file(str(file_path), mime=True)
        except Exception:
            return 'application/octet-stream'

    def __enter__(self) -> FileSystemTracker:
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_monitoring()