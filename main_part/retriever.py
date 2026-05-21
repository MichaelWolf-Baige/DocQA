import chromadb

class Retriever:
    def __init__(self,persist_dir = './chroma_db',collection_name='docqa'):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata = {'hnsw:space':'cosine'}
        )

    def index_chunks(self,chunks):
        '''把chunk列表写入Chroma，会覆盖同collection的旧数据'''
        existing = self.collection.count()
        if existing > 0 :
            ids = self.collection.get()['ids']
            self.collection.delete(ids)
        
        docs = []
        metas = []
        ids = []
        embs = []

        for c in chunks:
            docs.append(c['text'])
            metas.append({
                'chunk_id':c['chunk_id'],
                'source_page':c['source_page']
            })
            ids.append(str(c['chunk_id']))
            embs.append(c['embedding'])

        self.collection.add(
            documents=docs,
            metadatas=metas,
            ids=ids,
            embeddings=embs
        )
        print(f'索引完成，共{self.collection.count()}条')

    def search(self,query,embedder,top_k=5):
        '''用问题检索相关的top_k个chunk'''
        q_vec = embedder.embed_query(query)

        result = self.collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=['documents','metadatas','distances']
        )
        chunks = []
        for i in range(len(result['ids'][0])):
            distance = result['distances'][0][i]
            score = 1 - distance
            chunks.append({
                'chunk_id':int(result['ids'][0][i]),
                'text':result['documents'][0][i],
                'source_page':result['metadatas'][0][i]['source_page'],
                'score':round(score,4)
            })
        return chunks

if __name__ == '__main__':
    from pdf_parser import extract_text
    from chunker import chunk_by_size
    from embedder import Embedder

    print('加载模型...')
    embedder = Embedder()

    print('解析pdf并分块...')
    pages = extract_text(r'D:\桌面\I.pdf')
    chunks = chunk_by_size(pages)
    chunks = embedder.embed_chunks(chunks)

    print('建索引...')
    retriever = Retriever()
    retriever.index_chunks(chunks)

    print('\n---测试检索---')
    query = '怎么退款'
    result = retriever.search(query, embedder, top_k=3)

    for r in result:
        print(f"\n[chunk{r['chunk_id']}]第{r['source_page']}页score = {r['score']}")
        print(r['text'][:200])














