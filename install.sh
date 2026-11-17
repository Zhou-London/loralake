#!/usr/bin/env bash
# loralake 一键安装：子模块、conda 环境（含 CUDA 13.0 版 PyTorch 和 diffusion-pipe 依赖）、config。
# 可重复运行，已完成的步骤会跳过或增量更新。
#
#   ./install.sh               按 env/constraints.txt 锁定的版本安装（推荐）
#   ./install.sh --latest      不锁版本，安装 diffusion-pipe 依赖的最新版
#   LORALAKE_ENV=foo ./install.sh   使用其他环境名（默认 loralake）
set -euo pipefail

ENV_NAME="${LORALAKE_ENV:-loralake}"
PYTHON_VERSION="3.12"
CUDA_VERSION="13.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_PACKAGES=(torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0)
MIN_DRIVER=580
MINIFORGE_DIR="${HOME}/miniforge3"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="${ROOT}/env/constraints.txt"
USE_CONSTRAINTS=1

for arg in "$@"; do
  case "$arg" in
    --latest) USE_CONSTRAINTS=0 ;;
    -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m警告: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m错误: %s\033[0m\n' "$*" >&2; exit 1; }

step "检查系统"
[[ "$(uname -s)" == Linux ]] || die "只支持 Linux（包括 WSL2）"
command -v git >/dev/null || die "需要 git: sudo apt install -y git"
command -v curl >/dev/null || die "需要 curl: sudo apt install -y curl"
[[ "$ROOT" == /mnt/* ]] && warn "仓库在 ${ROOT}（Windows 文件系统），缓存和训练读写会很慢，建议放到 ~/ 下"

if command -v nvidia-smi >/dev/null; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "GPU: ${gpu}，驱动 ${driver}"
  # WSL 下 nvidia-smi 显示的是 Windows 驱动版本号（如 581.xx / 621.xx），同样适用这个下限
  [[ "${driver%%.*}" =~ ^[0-9]+$ ]] && (( ${driver%%.*} >= MIN_DRIVER )) || warn "CUDA ${CUDA_VERSION} 需要 ${MIN_DRIVER} 及以上的驱动，当前 ${driver}"
else
  warn "找不到 nvidia-smi，训练需要 NVIDIA 驱动（Ubuntu: sudo ubuntu-drivers install；WSL: 在 Windows 上装驱动）"
fi

step "初始化 diffusion-pipe 子模块"
git -C "$ROOT" submodule update --init --recursive

step "准备 conda"
if ! command -v conda >/dev/null; then
  if [[ -x "${MINIFORGE_DIR}/bin/conda" ]]; then
    :
  else
    echo "没有找到 conda，安装 Miniforge 到 ${MINIFORGE_DIR}"
    installer="$(mktemp --suffix=.sh)"
    curl -fsSL -o "$installer" \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
    bash "$installer" -b -p "$MINIFORGE_DIR"
    rm -f "$installer"
    "${MINIFORGE_DIR}/bin/conda" init bash >/dev/null
  fi
  CONDA_BASE="$MINIFORGE_DIR"
else
  CONDA_BASE="$(conda info --base)"
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "环境 ${ENV_NAME} 已存在，增量更新"
else
  conda create -y -n "$ENV_NAME" -c conda-forge "python=${PYTHON_VERSION}"
fi
conda activate "$ENV_NAME"
conda install -y -c conda-forge "cuda-nvcc=${CUDA_VERSION}" "cuda-version=${CUDA_VERSION}"

step "安装 PyTorch（CUDA ${CUDA_VERSION}）"
python -m pip install --upgrade pip
python -m pip install "${TORCH_PACKAGES[@]}" --index-url "$TORCH_INDEX"

step "安装 diffusion-pipe 依赖和 loralake"
pip_args=(-r "${ROOT}/third_party/diffusion-pipe/requirements.txt" -e "$ROOT")
if (( USE_CONSTRAINTS )) && [[ -f "$CONSTRAINTS" ]]; then
  echo "按 env/constraints.txt 锁定版本"
  pip_args+=(-c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX")
fi
python -m pip install "${pip_args[@]}"

step "配置文件"
if [[ -f "${ROOT}/config" ]]; then
  echo "config 已存在，保持不变"
else
  cp "${ROOT}/config.example" "${ROOT}/config"
  echo "已从 config.example 生成 config，请填写其中的 API key 和模型路径"
fi

step "自检"
python - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    print("\033[33m警告: torch.cuda.is_available() 为 False，训练将无法运行\033[0m")
EOF
llk --help >/dev/null && echo "llk 可用"

cat <<EOF

安装完成。使用前先激活环境（新开的终端都需要）：

  conda activate ${ENV_NAME}
  llk --help
EOF
