# loralake

从 Danbooru 下载图片、整理 caption、用 sd-scripts 训练 Anima LoRA 的命令行工具。

典型流程：

```bash
llk -s 初音未来                  # 搜 tag，看帖子数和预估大小
llk -d hatsune_miku              # 下载到 datasets/hatsune_miku
llk -p hatsune_miku              # 预览 caption
llk -t hatsune_miku              # 训练，结果在 output/
```

## 安装与配置

1. 初始化 sd-scripts 子模块：

   ```bash
   git submodule update --init
   ```

2. 在仓库根目录的 `config` 文件里填写配置（`KEY=VALUE`，`#` 开头为注释）。
3. Windows 下用仓库根目录的 `llk.cmd` 运行，它会用 `PYTHON_PATH` 指定的 Python（通常是 ComfyUI 的 venv）执行 `python -m loralake`。依赖见 `requirements.txt`。

| 配置项 | 说明 | 默认 |
|---|---|---|
| `PYTHON_PATH` | 运行 llk 的 Python，一般指向 ComfyUI 的 `.venv\Scripts\python.exe` | `python` |
| `DANBOORU_LOGIN` / `DANBOORU_API_KEY` | Danbooru 用户名和 API key，需同时填写 | 匿名 |
| `HF_API_KEY` | Hugging Face token，用于 HF 镜像下载；需先在数据集页面同意访问条款 | 读取环境变量 `HF_TOKEN` |
| `PROXY` | HTTP 代理地址 | 无 |
| `DOWNLOAD_WORKERS` | Danbooru CDN 并发数 | 8 |
| `MIRROR_WORKERS` | HF 镜像并发数 | 32 |
| `DATASET_DIR` / `DERIVED_DIR` / `OUTPUT_DIR` | 数据集、训练视图、训练输出目录（相对路径以仓库根目录为准） | `datasets` / `derived` / `output` |
| `COMFY_MODELS_PATH` | ComfyUI 的 `models` 目录；不填时按 `PYTHON_PATH` 推断 | 推断 |
| `ANIMA_DIT` / `ANIMA_QWEN3` / `ANIMA_VAE` | 直接指定底模、文本编码器、VAE 文件，不填则在 models 目录中自动查找 | 自动 |

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

自动生成训练视图和 sd-scripts 配置，训练完成后把各 epoch 的 LoRA 转成 ComfyUI 格式，保存在 `output/<name>/`。底模、Qwen3 文本编码器和 VAE 默认在 ComfyUI 的 `models/diffusion_models`、`models/text_encoders`、`models/vae` 下查找。

| 选项 | 说明 | 默认（`presets/default.yaml`） |
|---|---|---|
| `--name NAME` | 输出目录名 | `<数据集>_<时间>` |
| `--rank N` | LoRA rank | 32 |
| `--lr X` | 学习率 | 2e-5 |
| `--epochs N` | 训练轮数 | 10 |
| `--batch-size N` | batch 大小 | 4 |
| `--resolution N` | 训练分辨率 | 768 |
| `--save-every N` | 每 N 个 epoch 保存一次 | 5 |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | 梯度检查点 | 开 |
| `-y` / `--yes` | 跳过确认 | |

预处理的选项（`-c`、`--trigger`、`--remove-tags`、`--exclude-images`）同样适用于训练。输出目录里还有 `run.json`（本次用到的全部设置）和 `train.log`。

## 预设

预设是 `presets/` 下的 YAML 文件，分 `preprocess` 和 `train` 两节，命令行参数优先于预设：

```yaml
preprocess:
  trigger: ""
  exclude_images: [ai-generated, lowres, sketch]
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
  resolution: 768
  save_every_epochs: 5
  seed: 42
  optimizer: AdamW8bit
  caption_dropout: 0.05
  gradient_checkpointing: true
```

用 `-c 预设名` 选择，例如 `presets/style.yaml` 对应 `-c style`。
