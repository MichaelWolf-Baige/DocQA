import re

def _split_sentences(text):
    """按中英文标点分句，保留分隔符"""
    return [s.strip() for s in re.split(r'(?<=[。！？；\.\!\?\;\n])', text) if s.strip()]

def chunk_by_size(pages, chunk_size=512, overlap=128):
    '''
    按句子边界分块，避免截断语义。相邻chunk之间有overlap个字符的句子重叠。
    '''
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page['text']
        page_num = page['page']
        sentences = _split_sentences(text)

        current = ''
        for sent_idx, sent in enumerate(sentences):
            if len(current) + len(sent) > chunk_size and current:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current.strip(),
                    'source_page': page_num,
                })
                chunk_id += 1
                # 保留最后几个句子做overlap（用显式索引，避免重复句子导致 .index() 误匹配）
                tail = ''
                for s in reversed(sentences[:sent_idx]):
                    if len(tail) + len(s) <= overlap:
                        tail = s + tail
                    else:
                        break
                current = tail

            current += sent

        if current.strip():
            chunks.append({
                'chunk_id': chunk_id,
                'text': current.strip(),
                'source_page': page_num,
            })
            chunk_id += 1

    return chunks

if __name__ == "__main__":
    from pdf_parser import extract_text
    pages = extract_text('sample.pdf')
    chunks = chunk_by_size(pages)
    print(f'共{len(chunks)}个chunk\n')
    for c in chunks[:5]:
        print(f"[chunk{c['chunk_id']}]第{c['source_page']}页")
        print(c['text'][:384])
        print('---')







