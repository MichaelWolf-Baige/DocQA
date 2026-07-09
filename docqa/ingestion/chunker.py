"""
句子边界分块器
"""
import re
from typing import List
from .base import Page, Chunk, Chunker


class SentenceChunker(Chunker):
    """按句子边界分块，避免截断语义"""

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[。！？；\.\!\?\;\n])', text) if s.strip()]

    def chunk(self, pages: List[Page]) -> List[Chunk]:
        chunks = []
        chunk_id = 0

        for page in pages:
            source_file = page.metadata.get('source_file', '')
            sentences = self._split_sentences(page.text)
            current = ''

            for sent in sentences:
                if len(current) + len(sent) > self.chunk_size and current:
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=current.strip(),
                        source_page=page.page_number,
                        source_file=source_file,
                    ))
                    chunk_id += 1

                    # overlap: 保留末尾几个句子
                    tail = ''
                    for s in reversed(sentences[:sentences.index(sent)]):
                        if len(tail) + len(s) <= self.overlap:
                            tail = s + tail
                        else:
                            break
                    current = tail

                current += sent

            if current.strip():
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=current.strip(),
                    source_page=page.page_number,
                    source_file=source_file,
                ))
                chunk_id += 1

        return chunks
