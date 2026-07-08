"""
查询改写器
==========
用 LLM 将用户查询改写成 3-5 个语义变体。
解决"查询词汇"和"文档词汇"之间的语义鸿沟。

原理：
- 用户问"旅行社全称"，文档写的是"深圳市报春国际旅游集团有限公司"
- 改写为多个变体增加命中概率：补充同义词、改变句式、提取关键实体
- 保留原始查询（LiveRAG 基准：单次改写反而不如保留原始）
"""

from typing import List


REWRITE_PROMPT = """你是一个查询改写助手。将用户的问题改写成 3 个不同的表述方式，帮助搜索引擎找到答案。

规则：
1. 每个改写用不同的词汇和句式
2. 如果问题涉及专有名词（公司名、编号等），至少一个改写直接包含可能的答案形式
3. 保持原意不变
4. 每个改写一行，不要编号，不要多余解释

示例：
问题：旅行社的全称是什么？
深圳市报春国际旅游集团有限公司
合同甲方 旅行社 公司名称
旅行社叫什么名字 旅游公司全称

问题：{query}
"""


class QueryRewriter:
    """用 LLM 改写查询"""

    def __init__(self, generate_fn, prompt_template: str = None):
        """
        参数
        ----
        generate_fn : Callable[[str], str]
            LLM 生成函数（和 DocQAPipeline 的 llm.generate 签名一致）
        """
        self.generate = generate_fn
        self.prompt = prompt_template or REWRITE_PROMPT

    def rewrite(self, query: str, n_variants: int = 3) -> List[str]:
        """
        生成 n 个查询变体 + 原始查询。

        返回
        ----
        List[str] : [原始查询, 变体1, 变体2, ...]
        """
        prompt = self.prompt.format(query=query)
        raw = self.generate(prompt)

        if raw.startswith('[ERROR'):
            return [query]  # LLM 不可用时退化为原始查询

        # 解析：每行一个变体
        variants = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 1:
                variants.append(line)

        # 限制数量，始终包含原始查询
        variants = variants[:n_variants]
        if query not in variants:
            variants.insert(0, query)

        return variants
