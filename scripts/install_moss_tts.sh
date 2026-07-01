#!/usr/bin/env bash
# 安装 MOSS-TTS-Nano 及其依赖（CPU ONNX 推理）
# 参考 moss_tts_demo.py 的安装顺序，适配 GitHub Actions
set -e

# 在 GitHub Actions 的非 login shell 里显式加载 conda shell 函数，
# 避免直接调用 /home/runner/miniconda3/condabin/conda 时被系统 Python 执行而报 No module named 'conda'。
if ! command -v conda &>/dev/null; then
  echo "错误：当前 shell 未找到 conda 命令，请检查 setup-miniconda 是否正确安装"
  exit 1
fi
# 兼容 conda>=4.6 的 shell 初始化方式（Miniforge 自带）
if command -v conda &>/dev/null && [ -z "${_CONDA_EXE+x}" ]; then
  eval "$(conda shell.bash hook)" || {
    echo "错误：conda shell.bash hook 初始化失败"
    exit 1
  }
fi

INSTALL_DIR="${1:-MOSS-TTS-Nano}"

echo "=== [1/7] git clone MOSS-TTS-Nano ==="
if [ ! -d "$INSTALL_DIR" ]; then
  git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$INSTALL_DIR"
else
  echo "目录已存在，跳过 clone: $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

echo "=== [2/7] 确保 Python 3.10 (pynini 2.1.6 不支持 3.13) ==="
conda install -y -c conda-forge "python=3.10"

echo "=== [3/7] 安装 pynini (conda-forge) ==="
conda install -y -c conda-forge pynini=2.1.6

# 后续 pip 操作强制使用 conda 的 pip，避免系统 pip(python3.12)污染
CONDA_PIP="python -m pip"

echo "=== [4/7] 预装 torch CPU 版本 (对齐 demo 顺序) ==="
$CONDA_PIP install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "=== [5/7] 安装 WeTextProcessing (源码) ==="
$CONDA_PIP install git+https://github.com/WhizZest/WeTextProcessing.git || \
  echo "警告: WeTextProcessing 安装失败，moss-tts 将使用内置文本规范化"

echo "=== [6/7] 安装 requirements.txt (moss-tts 自身依赖) ==="
sed -i '/WeTextProcessing/d' requirements.txt || true
$CONDA_PIP install -r requirements.txt

echo "=== [7/7] editable 安装 moss-tts-nano ==="
$CONDA_PIP install -e .

echo "=== 强制卸载 torchcodec (纯 CPU 环境不可用，会导致 torchaudio.load 崩溃) ==="
$CONDA_PIP uninstall -y torchcodec || true

echo "=== 清理可能损坏的 transformers 动态模块缓存 ==="
rm -rf ~/.cache/huggingface/modules/transformers_modules/OpenMOSS* || true

echo "=== MOSS-TTS-Nano 安装完成 ==="
python --version
python -c "import torch, torchaudio; print(f'torch={torch.__version__} torchaudio={torchaudio.__version__}')"
