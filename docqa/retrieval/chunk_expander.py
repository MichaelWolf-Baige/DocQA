"""
Chunk 上下文扩展器 (Small-to-Big)
=================================
检索到小块后，取回同文档的相邻 chunk 拼成大块上下文给 LLM。

P0 修复: 多文档场景下 chunk_id 不分区文件——用 (source_file, chunk_id) 复合键
        确保"合同第5页"不会拼接"旅游指南第5页"。
"""
from typing import List, Dict
from docqa.ingestion.base import Chunk


class ChunkExpander:
    """检索结果上下文扩展器。多文档安全。"""

    def __init__(self, all_chunks: List[Chunk]):
        # 复合键: (source_file, chunk_id) → Chunk
        self._by_key: Dict[str, Chunk] = {}
        for c in all_chunks:
            file = getattr(c, 'source_file', '') or ''
            self._by_key[f'{file}::{c.chunk_id}'] = c

    def expand(self, retrieved: List[Chunk], window: int = 1) -> List[Chunk]:
        """
        扩展检索结果：每个 chunk 的 text 拼接同文档前后相邻 chunk。
        """
        expanded = []
        seen_keys = set()

        for c in retrieved:
            file = getattr(c, 'source_file', '') or ''
            base_key = f'{file}::{c.chunk_id}'
            if base_key in seen_keys:
                continue
            seen_keys.add(base_key)

            parts = []
            for offset in range(-window, window + 1):
                key = f'{file}::{c.chunk_id + offset}'
                if key in self._by_key:
                    parts.append(self._by_key[key].text)

            expanded_chunk = Chunk(
                chunk_id=c.chunk_id,
                text='\n'.join(parts),
                source_page=c.source_page,
                source_file=file,
                metadata=dict(c.metadata),
            )
            expanded_chunk.metadata['expanded_from_window'] = window
            expanded_chunk.metadata['original_text'] = c.text
            expanded.append(expanded_chunk)

        return expanded
