"""
配置加载器 —— 从 YAML 加载全部配置，提供类型安全的访问接口。
"""
import os
import yaml
from typing import Dict, Any


def load_config(path: str = None) -> Dict[str, Any]:
    """
    加载 config.yaml。

    如果 path 为 None，自动查找 docqa/config.yaml。
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析相对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    # 展开 embedder model_path
    model_rel = config.get('ingestion', {}).get('embedder', {}).get('model_path', '')
    if model_rel and not os.path.isabs(model_rel):
        config['ingestion']['embedder']['model_path'] = os.path.normpath(os.path.join(project_root, model_rel))

    # 展开 vector_store persist_dir
    persist_rel = config.get('ingestion', {}).get('vector_store', {}).get('persist_dir', '')
    if persist_rel and not os.path.isabs(persist_rel):
        config['ingestion']['vector_store']['persist_dir'] = os.path.normpath(os.path.join(project_root, persist_rel))

    return config


def get_ingestion_config(cfg: Dict) -> Dict:
    return cfg.get('ingestion', {})


def get_retrieval_config(cfg: Dict) -> Dict:
    return cfg.get('retrieval', {})


def get_generation_config(cfg: Dict) -> Dict:
    return cfg.get('generation', {})


def get_evaluation_config(cfg: Dict) -> Dict:
    return cfg.get('evaluation', {})
