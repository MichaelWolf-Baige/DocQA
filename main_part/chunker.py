def chunk_by_size(pages,chunk_size=512,overlap=128):
    '''
    把多页文本按固定大小切成chunk，相邻chunk之间有重叠
    '''
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page['text']
        page_num = page['page']
        start = 0

        while start < len(text):
            end = start +chunk_size
            chunk_text = text[start:end]
            chunks.append({
                'chunk_id':chunk_id,
                'text':chunk_text,
                'source_page':page_num
            })
            chunk_id += 1
            start = end - overlap

    return chunks

if __name__ == "__main__":
    from pdf_parser import extract_text
    pages = extract_text('D:\桌面\I.pdf')
    chunks = chunk_by_size(pages)
    print(f'共{len(chunks)}个chunk\n')
    for c in chunks[:5]:
        print(f"[chunk{c['chunk_id']}]第{c['source_page']}页")
        print(c['text'][:384])
        print('---')







