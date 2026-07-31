import re
from typing import List, Dict, Any, Optional

class TextChunker:
    """Configurable Document Chunker with paragraph/section path tracking and token/character overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document_id: str, content: str) -> List[Dict[str, Any]]:
        """Splits document content into chunks while extracting section header paths."""
        lines = content.split("\n")
        chunks = []
        current_section = "General"
        current_buffer = []
        current_char_count = 0
        chunk_idx = 0

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Detect Markdown / Text Header (e.g. "# Section", "1. Overview")
            if re.match(r'^(#+|\d+\.\s+|[A-Z\s]{4,}:)', trimmed):
                current_section = trimmed.lstrip("#").strip()

            line_length = len(trimmed)

            if current_char_count + line_length > self.chunk_size and current_buffer:
                chunk_text = "\n".join(current_buffer)
                chunks.append({
                    "chunk_id": f"{document_id}_chunk_{chunk_idx}",
                    "document_id": document_id,
                    "section_path": current_section,
                    "content": chunk_text
                })
                chunk_idx += 1

                # Retain overlap lines
                overlap_chars = 0
                overlap_buffer = []
                for prev_line in reversed(current_buffer):
                    if overlap_chars + len(prev_line) <= self.chunk_overlap:
                        overlap_buffer.insert(0, prev_line)
                        overlap_chars += len(prev_line)
                    else:
                        break

                current_buffer = overlap_buffer
                current_char_count = overlap_chars

            current_buffer.append(trimmed)
            current_char_count += line_length

        if current_buffer:
            chunk_text = "\n".join(current_buffer)
            chunks.append({
                "chunk_id": f"{document_id}_chunk_{chunk_idx}",
                "document_id": document_id,
                "section_path": current_section,
                "content": chunk_text
            })

        return chunks
