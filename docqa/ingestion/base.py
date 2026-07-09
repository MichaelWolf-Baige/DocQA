"""
摄入管线：抽象接口和数据结构
============================
定义 Page, Chunk 数据类和 Parser/Chunker/Embedder/VectorStore 接口。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ====== 数据结构 ======

@dataclass
class Page:
    """文档的一页"""
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """文本块——RAG 的基本检索单元"""
    chunk_id: int
    text: str
    source_page: int
    source_file: str = ""            # 来源文件名，多文档场景下必备
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ====== 抽象接口 ======

class DocumentParser(ABC):
    """文档解析器：PDF → List[Page]"""
    @abstractmethod
    def parse(self, path: str) -> List[Page]:
        ...


class Chunker(ABC):
    """分块器：List[Page] → List[Chunk]"""
    @abstractmethod
    def chunk(self, pages: List[Page]) -> List[Chunk]:
        ...


class Embedder(ABC):
    """嵌入器：文本 → 向量"""
    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class VectorStore(ABC):
    """向量存储：写入 + 向量检索"""
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """追加 chunks 到向量库（保留已有数据，按 source_file+chunk_id 去重）"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空全部数据"""
        ...

    def index(self, chunks: List[Chunk]) -> None:
        """兼容旧 API：先清空再写入"""
        self.clear()
        self.add(chunks)

    @abstractmethod
    def search(self, query_vec: List[float], top_k: int) -> List[Chunk]:
        """向量检索，返回带 text/source_page/score 的 Chunk 列表"""
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        """获取所有已索引的 chunk"""
        ...
