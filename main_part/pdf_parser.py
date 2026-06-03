import fitz

def extract_text(pdf_path):
    '''读取pdf文件，返回每一页的文本
       输入：pdf文件路径（字符串）
       输出：列表，每个元素是{“page”：页码，“text”：该页文本}
    '''
    doc = fitz.open(pdf_path)
    pages = []
    for i,page in enumerate(doc):
        text = page.get_text()
        pages.append({
            'page':i+1,
            'text':text
        })
    doc.close()
    return pages
if __name__ == '__main__':
    import sys
    path=sys.argv[1] if len(sys.argv) > 1 else 'sample.pdf'
    results = extract_text(path)
    print(f'共提取{len(results)}页\n')
    for r in results[:3]:  #先看前三页
        print(f"===第{r['page']}页===")
        print(r['text'][:200])
        print()