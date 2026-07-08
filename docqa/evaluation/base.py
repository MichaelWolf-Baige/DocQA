"""
评估模块：抽象接口
==================
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class EvaluationSuite(ABC):
    """评估套件接口"""
    @abstractmethod
    def run(self, pipeline, questions: List[Dict]) -> Dict[str, Any]:
        ...
