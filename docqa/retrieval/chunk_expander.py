"""
Chunk 上下文扩展器
==================
检索到小块后，自动取回相邻 chunk 拼成大块上下文给 LLM。

原理（Small-to-Big）：
- 小块（256 chars）：检索更精准，语义聚焦
- 大块（512-768 chars）：生成时上下文更完整
- 检索用小块，生成时扩展为相邻 chunk 的拼接

用法：
  expander = ChunkExpander(all_chunks)
  expanded = expander.expand(retrieved_chunks, window=1)
  # 每个检索到的 chunk 会附带前后各 1 个相邻 chunk
"""

from typing import List, Dict, Optional
from docqa.ingestion.base import Chunk


class ChunkExpander:
    """
    检索结果上下文扩展器。

    对每个检索到的 chunk，取回它前后 window 个相邻 chunk 的内容，
    拼接到 text 字段中，生成更完整的上下文。

    相邻 chunk 的判定基于 chunk_id 的连续性（假设同文档内 chunk_id 递增）。
    """

    def __init__(self, all_chunks: List[Chunk]):
        """
        参数
        ----
        all_chunks : List[Chunk]
            摄入时产生的全部 chunk 列表（按 chunk_id 索引）
        """
        self._by_id: Dict[int, Chunk] = {c.chunk_id: c for c in all_chunks}
        self._max_id = max(self._by_id.keys()) if self._by_id else 0

    def expand(
        self,
        retrieved: List[Chunk],
        window: int = 1,
    ) -> List[Chunk]:
        """
        扩展检索结果：每个 chunk 的 text 拼接前后相邻 chunk 的内容。

        参数
        ----
        retrieved : List[Chunk]
            检索返回的 chunk 列表
        window : int
            前后各取 window 个相邻 chunk（默认 1，即前后各 1 个）

        返回
        ----
        List[Chunk] : 扩展后的 chunk 列表（新增 expanded_text 字段）
        """
        expanded = []
        seen_ids = set()

        for c in retrieved:
            if c.chunk_id in seen_ids:
                continue
            seen_ids.add(c.chunk_id)

            # 取前后相邻 chunk
            parts = []
            for offset in range(-window, window + 1):
                neighbor_id = c.chunk_id + offset
                if neighbor_id in self._by_id:
                    parts.append(self._by_id[neighbor_id].text)

            # 创建扩展后的 chunk（保留原始 metadata）
            expanded_chunk = Chunk(
                chunk_id=c.chunk_id,
                text='\n'.join(parts),
                source_page=c.source_page,
                metadata=dict(c.metadata),
            )
            expanded_chunk.metadata['expanded_from_window'] = window
            expanded_chunk.metadata['original_text'] = c.text
            expanded.append(expanded_chunk)

        return expanded
