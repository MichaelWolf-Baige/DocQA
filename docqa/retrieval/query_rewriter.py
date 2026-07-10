"""
查询改写器
==========
用 LLM 将用户查询改写成 3-5 个语义变体。
"""
from typing import List


REWRITE_PROMPT = (
    "你是一个查询改写助手。将用户的问题改写成 3 个不同的表述方式。\n\n"
    "规则：\n"
    "1. 每个改写用不同的词汇和句式\n"
    "2. 保持原意不变\n"
    "3. 每个改写一行，不要编号\n\n"
    "问题：%s"
)


class QueryRewriter:
    """用 LLM 改写查询"""

    def __init__(self, generate_fn, prompt_template: str = None):
        self.generate = generate_fn
        self._template = prompt_template or REWRITE_PROMPT

    def rewrite(self, query: str, n_variants: int = 3) -> List[str]:
        """生成 n 个查询变体 + 原始查询。"""
        # 用 %s 替代 .format() 避免用户输入中的 {} 导致 KeyError
        prompt = self._template % query
        raw = self.generate(prompt)

        if raw.startswith('[ERROR'):
            return [query]

        variants = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 1:
                variants.append(line)

        variants = variants[:n_variants]
        if query not in variants:
            variants.insert(0, query)

        return variants
