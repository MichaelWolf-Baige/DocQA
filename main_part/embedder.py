from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self,model_path=None):
        if model_path is None:
            model_path = 'models/qwen3-embedding-0.6b'
        self.model=SentenceTransformer(model_path,device='cpu',local_files_only=True)

    def embed_chunks(self, chunks):
        '''把每个chunk的text转成embedding向量，加到chunk里'''
        texts= [c['text'] for c in chunks]
        vectors = self.model.encode(texts)

        for i,c in enumerate(chunks):
            c['embedding'] = vectors[i].tolist()
        
        return chunks
    
    def embed_query(self, query):
        '''把用户问题转成向量'''
        return self.model.encode(query).tolist()
    
    @property
    def dim(self):
        return self.model.get_embedding_dimension()

if __name__ == '__main__':
    from pdf_parser import extract_text
    from chunker import chunk_by_size

    pages = extract_text(r'D:\桌面\I.pdf')
    chunks = chunk_by_size(pages)

    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)

    print(f'共{len(chunks)}个chunk')
    print(f'向量维度：{embedder.dim}')
    print(f'第一个向量前5维：{chunks[0]["embedding"][:5]}')

    #验证语义相似度
    from sklearn.metrics.pairwise import cosine_similarity

    #相邻chunk应该有一定相似度
    s1 = np.array(chunks[0]['embedding']).reshape(1,-1)
    s2 = np.array(chunks[1]['embedding']).reshape(1,-1)
    sim = cosine_similarity(s1,s2)[0][0]
    print(f'chunk0 vs chunk1 余弦相似度：{sim:.4f}')

