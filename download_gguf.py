"""中国最快方案: ModelScope(Q4_K_M GGUF) → Ollama"""
import os, sys, shutil, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "models", "gguf")
os.makedirs(SAVE_DIR, exist_ok=True)

FILENAME = "qwen2.5-7b-it-Q4_K_M-LOT.gguf"
TARGET = os.path.join(SAVE_DIR, FILENAME)

if not os.path.exists(TARGET):
    print(f"ModelScope 下载 {FILENAME} (5.0GB)...")
    from modelscope.hub.file_download import model_file_download
    path = model_file_download(
        model_id="okwinds/Qwen2.5-7B-Instruct-GGUF-V3-LOT",
        file_path=FILENAME,
        cache_dir=SAVE_DIR,
    )
    # model_file_download 可能把文件下到 cache 子目录，复制出来
    if path != TARGET:
        print(f"移动到 {TARGET} ...")
        shutil.move(path, TARGET)
else:
    print(f"已存在: {TARGET}")

print(f"文件大小: {os.path.getsize(TARGET) / (1024**3):.1f} GB")

# Modelfile
mf = os.path.join(SAVE_DIR, "Modelfile")
with open(mf, "w") as f:
    f.write(f"FROM {TARGET}\n")

print("导入 Ollama ...")
subprocess.run(["ollama", "create", "qwen2.5:7b", "-f", mf], check=True)
print("\n✅ 完成! ollama run qwen2.5:7b '你好'")
