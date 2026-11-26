#!/usr/bin/env bash
# 从 Hugging Face（circlestone-labs/Anima）下载训练用的 Anima 底模、Qwen3 文本编码器和 VAE，
# 并把 config 里的 ANIMA_DIT / ANIMA_QWEN3 / ANIMA_VAE 改成下载到的路径。约 5.6GB。
# 可重复运行：已下载的文件会跳过，中断的下载会续传。
#
#   ./install_models.sh                 下载到仓库下的 models/（ComfyUI 的目录结构）
#   ./install_models.sh --dir PATH      下载到别的目录，比如已有的 ComfyUI models 目录
set -euo pipefail

HF_REPO="circlestone-labs/Anima"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main/split_files"
# config 键 / 子目录 / 文件名
MODELS=(
  "ANIMA_DIT    diffusion_models anima-base-v1.0.safetensors"
  "ANIMA_QWEN3  text_encoders    qwen_3_06b_base.safetensors"
  "ANIMA_VAE    vae              qwen_image_vae.safetensors"
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT}/config"
MODELS_DIR="${ROOT}/models"

while (( $# )); do
  case "$1" in
    --dir) [[ $# -ge 2 ]] || { echo "--dir 需要一个路径" >&2; exit 1; }; MODELS_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31m错误: %s\033[0m\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "需要 curl: sudo apt install -y curl"
mkdir -p "$MODELS_DIR"
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"

if [[ ! -f "$CONFIG" ]]; then
  cp "${ROOT}/config.example" "$CONFIG"
  echo "已从 config.example 生成 config"
fi

# 读取 config 里的值（KEY=VALUE，去掉引号）
config_value() {
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$CONFIG" | tail -1 | sed "s/^[\"']//; s/[\"'][[:space:]]*\$//"
}

# 把 config 里的 KEY 设为 VALUE；注释掉的 "# KEY=" 也替换掉，没有就追加
set_config() {
  local key="$1" value="$2" escaped
  escaped="$(printf '%s' "$value" | sed 's/[&|\\]/\\&/g')"
  if grep -qE "^[#[:space:]]*${key}[[:space:]]*=" "$CONFIG"; then
    sed -i -E "s|^[#[:space:]]*${key}[[:space:]]*=.*|${key}=${escaped}|" "$CONFIG"
  else
    printf '%s=%s\n' "$key" "$value" >> "$CONFIG"
  fi
}

proxy="$(config_value PROXY)"
if [[ -n "$proxy" ]]; then
  export https_proxy="$proxy" HTTPS_PROXY="$proxy"
  echo "使用代理 ${proxy}"
fi

download() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "已存在，跳过: ${dest}"
    return
  fi
  echo "下载 ${url##*/} → ${dest}"
  mkdir -p "$(dirname "$dest")"
  # 下载到 .part，完成后再改名；中断后重新运行会从 .part 续传
  curl -fL -# --retry 5 --retry-delay 5 --retry-all-errors -C - -o "${dest}.part" "$url" \
    || die "下载失败: ${url}。重新运行本脚本可以续传"
  mv "${dest}.part" "$dest"
}

step "从 ${HF_REPO} 下载模型到 ${MODELS_DIR}"
for entry in "${MODELS[@]}"; do
  read -r key subdir file <<< "$entry"
  download "${HF_BASE}/${subdir}/${file}" "${MODELS_DIR}/${subdir}/${file}"
done

step "更新 config 里的模型路径"
for entry in "${MODELS[@]}"; do
  read -r key subdir file <<< "$entry"
  set_config "$key" "${MODELS_DIR}/${subdir}/${file}"
  echo "${key}=${MODELS_DIR}/${subdir}/${file}"
done

echo
echo "完成。模型协议是 CircleStone Labs 非商业许可，详见 https://huggingface.co/${HF_REPO}"
