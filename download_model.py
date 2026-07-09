"""下载 embedding 模型到本地 models/ 目录。

只需运行一次：
    python download_model.py

模型：BAAI/bge-small-zh-v1.5（中文，512维，约95MB）
"""
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
LOCAL_DIR = 'models/bge-small-zh-v1.5'

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    if os.path.exists(LOCAL_DIR) and os.listdir(LOCAL_DIR):
        print(f'模型已存在: {LOCAL_DIR}')
    else:
        print(f'正在下载模型 {MODEL_NAME} ...')
        model = SentenceTransformer(MODEL_NAME)
        model.save(LOCAL_DIR)
        print(f'模型已保存到: {LOCAL_DIR}')

    # 验证
    model = SentenceTransformer(LOCAL_DIR, device='cpu')
    dim = model.get_sentence_embedding_dimension()
    print(f'[OK] model loaded, dim={dim}')
