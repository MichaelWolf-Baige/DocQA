"""
ChromaDB 向量存储
"""
import chromadb
from typing import List
from .base import VectorStore, Chunk


class ChromaStore(VectorStore):
    """ChromaDB 向量存储（cosine 距离，支持多文档追加）"""

    def __init__(self, persist_dir: str = './chroma_db', collection_name: str = 'docqa'):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'}
        )

    def _make_id(self, chunk: Chunk) -> str:
        """生成唯一 ID：{source_file}_{chunk_id}，避免多文档冲突"""
        if chunk.source_file:
            return f"{chunk.source_file}_{chunk.chunk_id}"
        return str(chunk.chunk_id)

    def add(self, chunks: List[Chunk]) -> None:
        """追加 chunks（保留已有数据，用 ID 去重）"""
        if not chunks:
            return

        docs, metas, ids, embs = [], [], [], []
        for c in chunks:
            cid = self._make_id(c)
            docs.append(c.text)
            metas.append({
                'chunk_id': c.chunk_id,
                'source_page': c.source_page,
                'source_file': c.source_file,
            })
            ids.append(cid)
            embs.append(c.embedding)

        # 跳过已存在的 ID
        existing = set(self.collection.get()['ids']) if self.collection.count() > 0 else set()
        new_docs, new_metas, new_ids, new_embs = [], [], [], []
        for d, m, i, e in zip(docs, metas, ids, embs):
            if i not in existing:
                new_docs.append(d)
                new_metas.append(m)
                new_ids.append(i)
                new_embs.append(e)

        if new_ids:
            self.collection.add(
                documents=new_docs,
                metadatas=new_metas,
                ids=new_ids,
                embeddings=new_embs,
            )

    def clear(self) -> None:
        """清空全部数据"""
        if self.collection.count() > 0:
            existing_ids = self.collection.get()['ids']
            self.collection.delete(existing_ids)

    # index() 继承自父类 VectorStore: clear() + add()

    def search(self, query_vec: List[float], top_k: int) -> List[Chunk]:
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        chunks = []
        for i in range(len(result['ids'][0])):
            distance = result['distances'][0][i]
            score = 1.0 - distance  # cosine distance → similarity
            meta = result['metadatas'][0][i]
            chunks.append(Chunk(
                chunk_id=meta.get('chunk_id', -1),
                text=result['documents'][0][i],
                source_page=meta.get('source_page', -1),
                source_file=meta.get('source_file', ''),
                metadata={'score': round(score, 4)}
            ))
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def get_all_chunks(self) -> List[Chunk]:
        if self.collection.count() == 0:
            return []
        data = self.collection.get()
        chunks = []
        for i in range(len(data['ids'])):
            meta = data['metadatas'][i]
            chunks.append(Chunk(
                chunk_id=meta.get('chunk_id', -1),
                text=data['documents'][i],
                source_page=meta.get('source_page', -1),
                source_file=meta.get('source_file', ''),
            ))
        return chunks
