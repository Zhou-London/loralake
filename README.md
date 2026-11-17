# loralake

从 Danbooru 下载图片、整理 caption、用 [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) 训练 Anima LoRA 的命令行工具。运行在 Ubuntu（原生或 WSL2）+ NVIDIA GPU 上。

典型流程：

```bash
llk -s 初音未来                  # 搜 tag，看帖子数和预估大小
llk -d hatsune_miku              # 下载到 datasets/hatsune_miku
llk -p hatsune_miku              # 预览 caption
llk -t hatsune_miku              # 训练，结果在 output/
```

## 安装

前提：

- Ubuntu 22.04 / 24.04（原生或 WSL2），`git` 和 `curl`（`sudo apt install -y git curl`）
- NVIDIA 驱动 580 及以上（CUDA 13.0 的要求）。原生 Ubuntu 用 `sudo ubuntu-drivers install` 安装；WSL2 只需在 Windows 上装驱动，不要在 WSL 里装
- Anima 底模、Qwen3 0.6B 文本编码器、Qwen-Image VAE 三个模型文件（通常已在 ComfyUI 的 `models/` 下）

```bash
git clone --recurse-submodules https://github.com/Zhou-London/loralake.git ~/loralake
cd ~/loralake
./install.sh
```

`install.sh` 会依次：初始化 diffusion-pipe 子模块 → 没有 conda 时安装 [Miniforge](https://github.com/conda-forge/miniforge) 到 `~/miniforge3` → 创建 `loralake` 环境（Python 3.12、`cuda-nvcc` 13.0、PyTorch 2.9.0+cu130）→ 安装 diffusion-pipe 依赖和 `llk` → 从 `config.example` 生成 `config` → 检查 GPU 是否可用。脚本可重复运行，更新代码后再跑一次即可同步依赖。

- 依赖版本默认按 [env/constraints.txt](env/constraints.txt) 锁定（一套验证过能训练的组合）。想用 diffusion-pipe 依赖的最新版：`./install.sh --latest`。
- 环境名可用 `LORALAKE_ENV=xxx ./install.sh` 修改。

之后每个新终端先激活环境再用：

```bash
conda activate loralake
llk --help
```

WSL 下仓库要放在 Linux 文件系统里（如 `~/loralake`），不要放在 `/mnt/c` 下，否则缓存和训练的读写会很慢。

## 配置

编辑仓库根目录的 `config`（`KEY=VALUE`，`#` 开头为注释；模板见 [config.example](config.example)）。至少要填 `COMFY_MODELS_PATH` 或三个 `ANIMA_*`，下载大数据集建议填 `HF_API_KEY`。

| 配置项 | 说明 | 默认 |
|---|---|---|
| `DANBOORU_LOGIN` / `DANBOORU_API_KEY` | Danbooru 用户名和 API key，需同时填写 | 匿名 |
| `HF_API_KEY` | Hugging Face token，用于 HF 镜像下载；需先在数据集页面同意访问条款 | 读取环境变量 `HF_TOKEN` |
| `PROXY` | HTTP 代理地址 | 无 |
| `DOWNLOAD_WORKERS` | Danbooru CDN 并发数 | 8 |
| `MIRROR_WORKERS` | HF 镜像并发数 | 32 |
| `DATASET_DIR` / `DERIVED_DIR` / `OUTPUT_DIR` | 数据集、训练视图、训练输出目录（相对路径以仓库根目录为准） | `datasets` / `derived` / `output` |
| `COMFY_MODELS_PATH` | ComfyUI 的 `models` 目录。原生 Linux 如 `~/ComfyUI/models`（写绝对路径），WSL 访问 Windows 上的 ComfyUI 如 `/mnt/c/Users/<用户名>/Documents/ComfyUI/models` | 无 |
| `ANIMA_DIT` / `ANIMA_QWEN3` / `ANIMA_VAE` | 直接指定底模、文本编码器、VAE 文件，不填则在 models 目录中自动查找 | 自动 |

## 迁移到另一台机器

`config`、`datasets/`、`derived/`、`output/`、`.cache/` 都不在 git 里。在新机器上 `git clone` + `./install.sh` 之后：

- 复制旧机器的 `config`，按新机器改 `COMFY_MODELS_PATH` 等路径；或者重新从 `config.example` 填写
- 需要继续用的数据集复制 `datasets/<tag>/`（下载中断的数据集复制过去后重新运行 `llk -d` 可以续传）
- 自己改过的 `presets/*.yaml` 记得提交或一并复制
- `derived/` 是训练视图和 latent 缓存，可以不复制，训练时会重新生成；`.cache/` 是 HF 镜像索引，会自动重新下载
- 模型文件放到新机器的 ComfyUI `models/` 下，或者用 `ANIMA_*` 指定

## 搜索 tag：`-s` / `--search`

```bash
llk -s 初音未来
llk -s miku --limit 20
```

支持英文、日文、中文等别名，列出匹配的 tag、类别、帖子数、可下载图片数和预估大小。`--limit` 控制显示条数（默认 10）。

## 下载：`-d` / `--download`

```bash
llk -d hatsune_miku
llk -d hatsune_miku kimono            # 多个 tag 直接跟在后面，等同于搜索 "hatsune_miku kimono"
llk -d hatsune_miku --limit 500 -y
```

| 选项 | 说明 |
|---|---|
| `--include TAGS` | 只下载同时带有这些 tag 的帖子，逗号分隔 |
| `--exclude TAGS` | 跳过带有这些 tag 的帖子，逗号分隔（写不写 `-` 前缀都可以） |
| `--limit N` | 本次最多下载 N 张 |
| `--output DIR` | 保存目录，默认 `datasets/<tag>` |
| `--workers N` | 覆盖 `DOWNLOAD_WORKERS` |
| `--mirror` / `--no-mirror` | 强制启用 / 禁用 HF 镜像 |
| `-y` / `--yes` | 跳过确认 |

### 下载来源

1. **HF 镜像**：[nyanko-devs/danbooru2026](https://huggingface.co/datasets/nyanko-devs/danbooru2026)，配合 [petersunde/danbooru2026-index](https://huggingface.co/datasets/petersunde/danbooru2026-index) 的偏移索引按帖子 id 取图，覆盖到 id 11,786,128。镜像图片是 WebP（质量 95），超过 4MP 的图被缩到 4MP。
   - 需要填写 `HF_API_KEY`。首次使用会下载约 80MB 索引，缓存在 `.cache/`。
   - 默认只在计划下载 200 张以上时启用，`--mirror` 可强制启用。
2. **Danbooru CDN**：镜像没有的图（更新的帖子、镜像缺失的图）从 CDN 补。
   - 需要从 CDN 下载的少于 150 张时全速下载；150 张及以上时按每秒 2.5 个请求限速，避免频繁撞到限流。
   - 被限流时会自动暂停 60 秒后继续。

所有图片最终都保存为 WebP：CDN 下载的原图校验 md5 后转码（质量 95，保留透明通道）。

### include / exclude 的处理方式

Danbooru 每次搜索能带的 tag 数有限（Member 账号 2 个，`id:`、`rating:` 不占名额）。llk 会先用满服务端名额：优先放最少见的 include，其次是覆盖面最大的 exclude。剩下的在本地按每个帖子自带的 tag 过滤，不会多发请求，被滤掉的图也不会下载。开始前会显示实际的服务端搜索和本地过滤条件。

下载阶段的过滤适合宏观筛选（比如只要 solo、不要 comic）；画质、AI 图等细粒度排除放在预处理阶段的 `exclude_images` 做。

### 保存内容

```
datasets/<tag>/
├── danbooru_<id>.webp     图片
├── danbooru_<id>.txt      原始 tag 列表
└── metadata.jsonl         每张图的元数据（tag、评级、尺寸、来源 mirror/cdn 等）
```

中断或有失败时重新运行同一命令即可继续，已下载的会跳过。有失败项时退出码为 2。

## 预处理：`-p` / `--preprocess`

```bash
llk -p hatsune_miku                                   # 随机预览 5 条 caption
llk -p hatsune_miku --limit 10                        # 预览 10 条
llk -p hatsune_miku --output derived/hatsune_miku/anima   # 生成训练视图
```

按预设排除图片、生成 Anima 格式的 caption。不带 `--output` 时只预览；带 `--output` 时把选中的图片以硬链接（不支持时复制）放到视图目录，并写入 caption。

| 选项 | 说明 |
|---|---|
| `-c` / `--config PRESET` | 使用 `presets/` 下的预设或指定 YAML 文件，默认 `presets/default.yaml` |
| `--trigger TEXT` | 触发词，放在 caption 最前面 |
| `--remove-tags TAGS` | 从 caption 中去掉的 tag，逗号分隔 |
| `--exclude-images TAGS` | 带有这些 tag 的图片不参与训练，逗号分隔（会覆盖预设里的列表） |

## 训练：`-t` / `--train`

```bash
llk -t hatsune_miku
llk -t hatsune_miku --trigger miku --epochs 15 --name miku_v1
```

自动生成训练视图和 diffusion-pipe 配置，训练完成后把各 epoch 的 LoRA（本身就是 ComfyUI 格式）移到 `output/<name>/`：最后一个 epoch 是 `<name>.safetensors`，中间版本是 `<name>-<epoch>.safetensors`。diffusion-pipe 的配置副本和 TensorBoard 日志在 `output/<name>/checkpoints/`（训练成功后会删掉 DeepSpeed 的训练状态，不支持续训），latent 和文本缓存在训练视图目录的 `cache/` 下。LLM adapter 不参与训练（`llm_adapter_lr = 0`，与官方示例一致）。

diffusion-pipe 按宽高比分桶，每个桶只保留 batch 大小整数倍的图片，凑不满的直接丢弃。llk 会读取每张图的实际尺寸，按 diffusion-pipe 的分桶规则（进入 log 空间中最近的宽高比）模拟，选出最多 9 个宽高比，使丢弃的图片数加上被裁掉的面积最小，写入 `dataset.toml` 的 `ar_buckets`。开始前会显示分桶数、每个 epoch 丢弃的张数和平均裁切比例；丢弃数通常等于图片总数除以 batch 的余数。底模、Qwen3 文本编码器和 VAE 默认在 ComfyUI 的 `models/diffusion_models`、`models/text_encoders`、`models/vae` 下查找。

| 选项 | 说明 | 内置默认 |
|---|---|---|
| `--name NAME` | 输出目录名 | `<数据集>_<时间>` |
| `--rank N` | LoRA rank | 32 |
| `--lr X` | 学习率 | 2e-5 |
| `--epochs N` | 训练轮数 | 10 |
| `--batch-size N` | batch 大小 | 4 |
| `--resolution N` | 训练分辨率 | 1024 |
| `--save-every N` | 每 N 个 epoch 保存一次 | 2 |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | 梯度检查点 | 开 |
| `-y` / `--yes` | 跳过确认 | |

优先级：命令行参数 > 预设（默认 `presets/default.yaml`）> 内置默认。预处理的选项（`-c`、`--trigger`、`--remove-tags`、`--exclude-images`）同样适用于训练。输出目录里还有 `run.json`（本次用到的全部设置）和 `train.log`。

## 预设

预设是 `presets/` 下的 YAML 文件，分 `preprocess` 和 `train` 两节，只需写要改的键，没写的用内置默认。命令行参数优先于预设。下面列出全部键及其内置默认值：

```yaml
preprocess:
  trigger: ""
  exclude_images: []     # 带这些 tag 的图片不参与训练，如 [ai-generated, lowres, sketch]
  remove_tags: []
  prefix_tags: []
  include_rating: true    # caption 中加入 safe / sensitive / nsfw / explicit
  include_year: true      # 加入 "year 2025"
  include_artist: true    # 加入 "@画师名"

train:
  rank: 32
  learning_rate: 2.0e-5
  epochs: 10
  batch_size: 4
  resolution: 1024
  save_every_epochs: 2
  optimizer: adamw_optimi # diffusion-pipe 的优化器名，也可用 AdamW8bit、AdamW8bitKahan、Prodigy 等
  caption_dropout: 0      # 以空 caption 训练的比例（diffusion-pipe 的 uncond_fraction）
  gradient_checkpointing: true
```

用 `-c 预设名` 选择，例如 `presets/style.yaml` 对应 `-c style`。
