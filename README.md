# loralake

从 Danbooru 下载图片、整理 caption、用 [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) 训练 Anima LoRA 的命令行工具。运行在 Ubuntu（原生或 WSL2）+ NVIDIA GPU 上。

典型流程：

```bash
llk -s 初音未来                  # 搜 tag，看帖子数和预估大小
llk -d hatsune_miku              # 下载到 datasets/hatsune_miku
llk --review hatsune_miku        # 在浏览器里整理数据集：去重、丢图、改 tag
llk -p hatsune_miku              # 预览 caption
llk -t hatsune_miku              # 训练，结果在 output/
```

## 安装

前提：

- Ubuntu 22.04 / 24.04（原生或 WSL2），`git` 和 `curl`（`sudo apt install -y git curl`）
- NVIDIA 驱动 580 及以上（CUDA 13.0 的要求）。原生 Ubuntu 用 `sudo ubuntu-drivers install` 安装；WSL2 只需在 Windows 上装驱动，不要在 WSL 里装
- Anima 底模、Qwen3 0.6B 文本编码器、Qwen-Image VAE 三个模型文件（通常已在 ComfyUI 的 `models/` 下；没有的话用下面的 `./install_models.sh` 下载）

```bash
git clone --recurse-submodules https://github.com/Zhou-London/loralake.git ~/loralake
cd ~/loralake
./install.sh
```

`install.sh` 会依次：初始化 diffusion-pipe 子模块 → 没有 conda 时安装 [Miniforge](https://github.com/conda-forge/miniforge) 到 `~/miniforge3` → 创建 `loralake` 环境（Python 3.12、`cuda-nvcc` 13.0、PyTorch 2.9.0+cu130；系统里没有 `nvim` 时一并装上 conda-forge 的 neovim）→ 安装 diffusion-pipe 依赖和 `llk` → 从 `config.example` 生成 `config` → 检查 GPU 是否可用。脚本可重复运行，更新代码后再跑一次即可同步依赖。

- 依赖版本默认按 [env/constraints.txt](env/constraints.txt) 锁定（一套验证过能训练的组合）。想用 diffusion-pipe 依赖的最新版：`./install.sh --latest`。
- 环境名可用 `LORALAKE_ENV=xxx ./install.sh` 修改。
- 不想用 conda、已经有装好 CUDA 版 PyTorch 的 Python 时：`./install.sh --system`，直接装进当前 Python，沿用已有的 torch，不使用 `env/constraints.txt`。需要能找到 `nvcc`（deepspeed 导入时要用）。

之后每个新终端先激活环境再用：

```bash
conda activate loralake
llk --help
```

### 下载模型：`install_models.sh`

没有现成的 ComfyUI 模型时（比如新的服务器），用它从 Hugging Face 的 [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) 下载三个模型（共约 5.6GB），并把 `config` 里的 `ANIMA_DIT` / `ANIMA_QWEN3` / `ANIMA_VAE` 改成下载到的路径：

```bash
./install_models.sh                  # 下载到仓库下的 models/
./install_models.sh --dir ~/ComfyUI/models   # 或者下载到指定目录（同样按 ComfyUI 的子目录放）
```

不需要激活 conda 环境，也不需要 HF token；会使用 `config` 里的 `PROXY`。已下载的文件会跳过，中断后重新运行会续传。模型是 CircleStone Labs 非商业许可。

WSL 下仓库要放在 Linux 文件系统里（如 `~/loralake`），不要放在 `/mnt/c` 下，否则缓存和训练的读写会很慢。

### Google Colab

用 [colab.ipynb](colab.ipynb)：上传到 Colab（或在 Colab 里从 GitHub 打开），按顺序运行即可。

- 运行时类型选 L4 或更好的 GPU。T4（免费版）不支持 bf16，不适合训练。
- `install.sh` 检测到 Colab 会自动使用 `--system` 模式，沿用 Colab 自带的 PyTorch。编写时（2026-11）Colab 的环境为 Ubuntu 24.04、Python 3.13、torch 2.14.1+cu130、驱动 580.82.07。
- `config` 和 `presets/default.yaml` 的内容各在一个单元格里，只有要改的文本；改完运行后面的「应用」单元格写入仓库（预设同时保存到 Drive）。
- Google Drive 的 `MyDrive/loralake/` 用来长期保存预设、模型、数据集和训练输出。启动时把 Drive 上的内容复制到本地 `/content/work`，之后全部在本地读写；下载或训练后运行「保存到 Drive」单元格，把新增的数据集同步回 Drive、把训练输出移动到 Drive 的 `output/`。会话断开前没保存的内容会丢失。
- Drive 上 `presets/` 里的其他预设会覆盖仓库里的同名预设，用 `-c 名称` 选择。

## 配置

编辑仓库根目录的 `config`（`KEY=VALUE`，`#` 开头为注释；模板见 [config.example](config.example)）。至少要填 `COMFY_MODELS_PATH` 或三个 `ANIMA_*`（`./install_models.sh` 会自动填 `ANIMA_*`），下载大数据集建议填 `HF_API_KEY`。

| 配置项 | 说明 | 默认 |
|---|---|---|
| `DANBOORU_LOGIN` / `DANBOORU_API_KEY` | Danbooru 用户名和 API key，需同时填写 | 匿名 |
| `HF_API_KEY` | Hugging Face token，用于 HF 镜像下载（需先在数据集页面同意访问条款）、`llk --upload`（需要 write 权限）和 `llk --fetch` | 读取环境变量 `HF_TOKEN` |
| `HF_REPO` | `llk --upload` / `--fetch` 使用的私有仓库（旧名 `HF_UPLOAD_REPO` 仍可用） | `<用户名>/loralake` |
| `PROXY` | HTTP 代理地址 | 无 |
| `DOWNLOAD_WORKERS` | Danbooru CDN 并发数 | 8 |
| `MIRROR_WORKERS` | HF 镜像并发数 | 32 |
| `DATASET_DIR` / `DERIVED_DIR` / `OUTPUT_DIR` | 数据集、训练视图、训练输出目录（相对路径以仓库根目录为准） | `datasets` / `derived` / `output` |
| `COMFY_MODELS_PATH` | ComfyUI 的 `models` 目录。原生 Linux 如 `~/ComfyUI/models`（写绝对路径），WSL 访问 Windows 上的 ComfyUI 如 `/mnt/c/Users/<用户名>/Documents/ComfyUI/models` | 无 |
| `ANIMA_DIT` / `ANIMA_QWEN3` / `ANIMA_VAE` | 直接指定底模、文本编码器、VAE 文件，不填则在 models 目录中自动查找 | 自动 |

## 迁移到另一台机器

`config`、`datasets/`、`derived/`、`output/`、`.cache/` 都不在 git 里。在新机器上 `git clone` + `./install.sh` 之后：

- 复制旧机器的 `config`，按新机器改 `COMFY_MODELS_PATH` 等路径；或者重新从 `config.example` 填写
- 需要继续用的数据集复制 `datasets/<tag>/`，或者在旧机器上 `llk --upload`、新机器上 `llk --fetch` 经 Hugging Face 中转（见下文）；下载中断的数据集复制过去后重新运行 `llk -d` 可以续传
- 自己改过的 `presets/*.yaml` 记得提交或一并复制
- `derived/` 是训练视图和 latent 缓存，可以不复制，训练时会重新生成；`.cache/` 是 HF 镜像索引，会自动重新下载
- 模型文件放到新机器的 ComfyUI `models/` 下，或者用 `ANIMA_*` 指定；也可以运行 `./install_models.sh` 从 Hugging Face 下载

### 打包数据集：`--pack` / `--unpack`

```bash
llk --pack hatsune_miku                         # 生成 datasets/hatsune_miku.tar.zst
llk --unpack hatsune_miku                       # 解压 datasets/hatsune_miku.tar.zst 到 datasets/hatsune_miku/
llk --unpack /path/to/hatsune_miku.tar.zst      # 也可以直接给压缩包路径
```

把整个数据集目录（图片、caption、`metadata.jsonl`、`review.json` 等，跳过下载到一半的 `.part` 文件）打成一个 tar + zstd 压缩包，放在 `datasets/` 下。图片本身是 WebP，基本压不小，打包主要是方便拷贝和上传；压缩包带校验和，传输损坏会在解压时报错。

解压时数据集名取自压缩包内容。目标已存在时会先询问，确认后整个替换（`-y` 跳过询问）；解压失败不会动原有的数据集。

`llk --upload` 会自动打包上传，`llk --fetch` 会自动下载解压，经 Hugging Face 中转时不需要手动打包。

### 经 Hugging Face 中转：`--upload` / `--fetch`

在机器之间搬训练输出和数据集，比如在计算中心训练完把结果取回本地。中转用自己的 Hugging Face 私有仓库：

```bash
llk --upload kincora_20261126-1645   # 训练输出 output/<名字>/ → 仓库的 output/<名字>/
llk --upload kincora                 # 数据集：打包后上传到仓库的 datasets/kincora.tar.zst

llk --fetch kincora_20261126-1645    # 仓库的 output/<名字>/ → 本地 output/<名字>/
llk --fetch kincora                  # 仓库的 datasets/kincora.tar.zst → datasets/，并自动解压
```

- 使用 `config` 里的 `HF_API_KEY`，不需要浏览器登录，token 不会过期。上传需要 write 权限，只取回的机器用 read 权限即可（在 <https://huggingface.co/settings/tokens> 生成 Write 类型，或 Fine-grained 类型勾选自己仓库的写权限）。
- 仓库默认是 `<用户名>/loralake`，私有的 dataset 仓库，不存在时自动创建；可用 `HF_REPO` 指定。
- 上传时名字先按本地 `output/` 下的训练输出查找，找不到再按数据集查找；取回时按仓库里的训练输出和数据集查找，找不到会列出仓库里有哪些。覆盖本地已有的内容前会确认，`-y` 跳过。
- 训练输出整个目录上传；训练中断时目录里还有训练状态，可以拿到别的机器上续训。
- 中断后重新运行同一命令即可：上传时没有变化的大文件不会重复传，取回时从断点继续（临时目录 `output/.hf-fetch`、`datasets/.hf-fetch`）。会使用 `config` 里的 `PROXY`。很多计算中心只有登录节点能上网，这时在登录节点上运行。
- 取回的训练输出替换本地同名目录，中断的训练取回后可以直接 `llk --resume <名字>`。

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

查重时还会在这里生成 `.dedup_cache.json`（哈希缓存，可以删除）；`llk --review` 的整理结果保存在 `review.json`（见下文）。

中断或有失败时重新运行同一命令即可继续，已下载的会跳过。有失败项时退出码为 2。

## 整理数据集：`--review`

```bash
llk --review hatsune_miku
```

在本机启动一个页面（默认 <http://localhost:8765/>，端口被占用时自动顺延，也可用 `--port` 指定），WSL 下直接用 Windows 的浏览器打开。这一步只针对数据集本身，与模型无关，在预处理之前做（预设里只用到 `dedup_threshold`，可用 `-c` 选择）；修改实时保存到 `datasets/<tag>/review.json`，之后的预处理和训练都会先应用它。完成后在终端按 Ctrl+C 退出。

- **图片**：浏览全部图片。搜索框按 tag 筛选（空格分隔多个，`-tag` 排除，支持 `*` 通配和帖子 id），顶部可切换全部 / 保留 / 丢弃 / 已改。
- **重复**：逐组比较疑似重复的图片，丢弃不要的后点「确认」（或按 Enter）。分辨率最高的一张加粗，下方列出 `textless version`、`watermark` 等有助于判断的 tag。
- **选中**：单击选中，Ctrl 单击多选，Shift 单击或 Shift+方向键连选，Ctrl+A 全选当前筛选结果。右侧面板显示选中图片的 tag：单击 tag 搜索同 tag 的图片，× 删除，输入框添加（可用 `artist:`、`character:` 等前缀指定类别）；多选时对所有选中图片生效，只有部分图片有的 tag 显示数量，点 + 补全到全部。
- **丢弃**：按 D（或 Delete）、点面板上的按钮或缩略图右上角的按钮，丢弃 / 恢复选中的图片。单张时丢弃后自动跳到下一张。
- **大图**：双击或按空格查看原图，方向键切换，Esc 关闭。

去重分两步。Danbooru 上同一张插画常常有好几份：不同尺寸的重传、去字版、加了 logo 或标题的封面、表情差分、去背景版等，重复的图会让 LoRA 过拟合到那几张构图上。

1. **自动去除**：感知哈希（pHash）相近、宽高比相同，并且缩小后逐块比较几乎没有差别的图片，视为同一张图（缩放、重新编码、轻微调色），只保留像素最多的一张。这一步只处理非常确定的情况，页面上不显示被去除的图。
2. **人工确认**：其余相近的图片（去字版、加 logo 或签名、表情差分、去背景、裁切等）按组列出，确认前全部保留。这一步宁可多列，会有一些只是构图相似的图，点「确认」跳过即可；嫌多或嫌少可以调预设的 `dedup_threshold`。预处理和训练时会提示待确认的组数。只剩一张没丢弃的组自动算作已确认；新下载的图片和已确认的图片相近时，所在的组会重新变成待确认。

改过的 tag 会替换 `metadata.jsonl` 里的 tag，参与预处理的 `exclude_images` 判断和 caption 生成；在面板上点「还原 tag」恢复原样。缩略图缓存在 `derived/<tag>/.thumbs/`。

Colab 上在同一个单元格里先运行 `from google.colab import output; output.serve_kernel_port_as_window(8765)`，再运行 `!llk --review <数据集>`，点击输出的链接打开页面；完成后中断该单元格。

## 预处理：`-p` / `--preprocess`

```bash
llk -p hatsune_miku                                   # 随机预览 5 条 caption
llk -p hatsune_miku --limit 10                        # 预览 10 条
llk -p hatsune_miku --output derived/hatsune_miku/anima   # 生成训练视图
```

应用 `--review` 的整理结果，再按预设排除图片、去重、生成 Anima 格式的 caption。不带 `--output` 时只预览；带 `--output` 时把选中的图片以硬链接（不支持时复制）放到视图目录，并写入 caption。

| 选项 | 说明 |
|---|---|
| `-c` / `--config PRESET` | 使用 `presets/` 下的预设或指定 YAML 文件，默认 `presets/default.yaml` |
| `--trigger TEXT` | 触发词，放在 caption 最前面 |
| `--remove-tags TAGS` | 从 caption 中去掉的 tag，逗号分隔 |
| `--exclude-images TAGS` | 带有这些 tag 的图片不参与训练，逗号分隔（会覆盖预设里的列表） |
| `--dedup` / `--no-dedup` | 开启 / 关闭去重（覆盖预设的 `dedup`，默认开启） |

## 训练：`-t` / `--train`

```bash
llk -t hatsune_miku
llk -t hatsune_miku --trigger miku --epochs 15 --name miku_v1
```

自动生成训练视图和 diffusion-pipe 配置，训练完成后把各 epoch 的 LoRA（本身就是 ComfyUI 格式）移到 `output/<name>/`：最后一个 epoch 是 `<name>.safetensors`，中间版本是 `<name>-<epoch>.safetensors`。diffusion-pipe 的配置副本和 TensorBoard 日志在 `output/<name>/checkpoints/`，latent 和文本缓存在训练视图目录的 `cache/` 下。LLM adapter 不参与训练（`llm_adapter_lr = 0`，与官方示例一致）。

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

### 中断和续训：`-r` / `--resume`

```bash
llk --resume miku_v1
```

训练中可以随时按 Ctrl+C 中断，已经保存的各 epoch LoRA 会移到 `output/<name>/`。训练状态（优化器、数据进度）每 30 分钟保存一次（预设键 `checkpoint_every_minutes`），只保留最新一份；中断后用 `--resume <name>` 从最近一次保存的状态继续，最多重练这 30 分钟内的步数。续训沿用 `run.json` 里的全部设置，不能再改 `--epochs`、`--lr` 等选项；数据集、筛选结果或底模和开始时不同则拒绝续训。训练成功结束后会删掉训练状态。

优先级：命令行参数 > 预设（默认 `presets/default.yaml`）> 内置默认。预处理的选项（`-c`、`--trigger`、`--remove-tags`、`--exclude-images`、`--dedup`/`--no-dedup`）同样适用于训练。输出目录里还有 `run.json`（本次用到的全部设置）和 `train.log`。

## 预设

预设是 `presets/` 下的 YAML 文件，分 `preprocess` 和 `train` 两节，只需写要改的键，没写的用内置默认。命令行参数优先于预设。下面列出全部键及其内置默认值：

```yaml
preprocess:
  trigger: ""
  exclude_images: []     # 带这些 tag 的图片不参与训练，如 [ai-generated, lowres, sketch]
  dedup: true             # 去重（自动去除像素相同的图片）
  dedup_threshold: 16     # 疑似重复的 pHash 距离上限（0～64），越大 --review 里列出的组越多；自动去除基本不受影响
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
  checkpoint_every_minutes: 30 # 每隔多少分钟保存一次训练状态，用于 --resume
  optimizer: adamw_optimi # diffusion-pipe 的优化器名，也可用 AdamW8bit、AdamW8bitKahan、Prodigy 等
  caption_dropout: 0      # 以空 caption 训练的比例（diffusion-pipe 的 uncond_fraction）
  gradient_checkpointing: true
```

用 `-c 预设名` 选择，例如 `presets/style.yaml` 对应 `-c style`。
