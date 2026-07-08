"""
PyMuPDF 文档解析器
"""
import fitz
from typing import List
from .base import Page, DocumentParser


class PyMuPDFParser(DocumentParser):
    """使用 PyMuPDF (fitz) 解析 PDF"""

    def parse(self, path: str) -> List[Page]:
        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append(Page(page_number=i + 1, text=text))
        doc.close()
        return pages
