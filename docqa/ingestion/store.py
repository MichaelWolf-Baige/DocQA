"""
ChromaDB 向量存储
"""
import chromadb
from typing import List
from .base import VectorStore, Chunk


class ChromaStore(VectorStore):
    """ChromaDB 向量存储（cosine 距离）"""

    def __init__(self, persist_dir: str = './chroma_db', collection_name: str = 'docqa'):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'}
        )

    def index(self, chunks: List[Chunk]) -> None:
        # 清除旧数据
        if self.collection.count() > 0:
            existing_ids = self.collection.get()['ids']
            self.collection.delete(existing_ids)

        docs, metas, ids, embs = [], [], [], []
        for c in chunks:
            docs.append(c.text)
            metas.append({'chunk_id': c.chunk_id, 'source_page': c.source_page})
            ids.append(str(c.chunk_id))
            embs.append(c.embedding)

        self.collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)

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
            chunks.append(Chunk(
                chunk_id=int(result['ids'][0][i]),
                text=result['documents'][0][i],
                source_page=result['metadatas'][0][i].get('source_page', -1),
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
            chunks.append(Chunk(
                chunk_id=int(data['ids'][i]),
                text=data['documents'][i],
                source_page=data['metadatas'][i].get('source_page', -1),
            ))
        return chunks
