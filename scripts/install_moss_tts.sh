#!/usr/bin/env bash
# 安装 MOSS-TTS-Nano 及其依赖（CPU ONNX 推理）
# 在 GitHub Actions ubuntu-latest 上运行
set -e

INSTALL_DIR="${1:-MOSS-TTS-Nano}"

echo "=== [1/5] git clone MOSS-TTS-Nano ==="
if [ ! -d "$INSTALL_DIR" ]; then
  git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$INSTALL_DIR"
else
  echo "目录已存在，跳过 clone: $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

echo "=== [2/5] 安装 pynini (conda-forge) ==="
# pynini 在 pip 上 wheel 兼容性差，用 conda-forge 最稳定
conda install -y -c conda-forge pynini==2.1.6

echo "=== [3/5] 安装 WeTextProcessing (源码) ==="
pip install git+https://github.com/WhizZest/WeTextProcessing.git || \
  echo "警告: WeTextProcessing 安装失败，moss-tts 将使用内置文本规范化"

echo "=== [4/5] 安装 requirements.txt ==="
# 移除 WeTextProcessing 行避免重复/冲突
sed -i '/WeTextProcessing/d' requirements.txt || true
pip install -r requirements.txt

echo "=== [5/5] editable 安装 moss-tts-nano ==="
pip install -e .

echo "=== 对齐 torch CPU 版本 ==="
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "=== 清理可能损坏的 transformers 动态模块缓存 ==="
rm -rf ~/.cache/huggingface/modules/transformers_modules/OpenMOSS* || true

echo "=== MOSS-TTS-Nano 安装完成 ==="
