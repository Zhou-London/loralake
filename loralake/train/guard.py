"""由 DeepSpeed 代替 diffusion-pipe 的 train.py 启动：装上检查后再原样运行 train.py（不改子模块）。

- 每一步在 train.log 里多打一行 [loralake]：loss、梯度范数、LoRA 权重最大值、这一步的时间步 t 和图片
- 优化器更新权重之前检查梯度，出现 NaN/inf 就不让这一步生效：写下诊断信息，重放这个 batch 找出最先
  出现非有限值的模块，然后以 NAN_EXIT_CODE 退出，由 llk 回退到上一个训练状态重试（见 commands._run）
- 每次启动用新的随机种子并记进日志；设置环境变量 LORALAKE_SEED 可以指定种子，用来复现

这个文件在 diffusion-pipe 目录下运行，只能依赖标准库和训练环境里的包，不能 import loralake。
"""

import json
import os
import random
import re
import runpy
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

LOG_PREFIX = "[loralake]"
NAN_EXIT_CODE = 86
EVENTS_DIR = "nan_events"
SEED_ENV = "LORALAKE_SEED"
MAX_LISTED = 20
REMEMBERED_BATCHES = 8
LAYER_INDEX = re.compile(r"^(\d+)\.")


def _arg_value(flag: str) -> str:
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    raise SystemExit(f"{LOG_PREFIX} 缺少参数 {flag}")


def _caption_index(config: dict[str, Any]) -> dict[str, str]:
    """caption 文本 → 图片文件名（不含扩展名），用来在日志里写出每一步用了哪些图。"""
    index: dict[str, str] = {}
    try:
        dataset = tomllib.loads(Path(config["dataset"]).read_text(encoding="utf-8"))
        for directory in dataset.get("directory", []):
            for caption_file in Path(directory["path"]).glob("*.txt"):
                index.setdefault(caption_file.read_text(encoding="utf-8").strip(), caption_file.stem)
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    return index


def _is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _say(message: str) -> None:
    if _is_main_process():
        print(f"{LOG_PREFIX} {message}", flush=True)


class Guard:
    def __init__(self, seed: int, events_dir: Path, captions: dict[str, str]) -> None:
        self.seed = seed
        self.events_dir = events_dir
        self.captions = captions
        self.batches: dict[int, dict[str, Any]] = {}  # noisy_latents 的 data_ptr → 这个 batch 的 caption 和 t
        self.current: Any = None  # 这一步送进模型的 (features, label)
        self.loss: Any = None
        self.epoch: int | None = None  # 这一步的 batch 属于哪个 epoch
        self.params: list[tuple[str, Any]] | None = None

    # ---- 记录每一步的输入 ----

    def remember(self, features: Any, batch: dict[str, Any]) -> None:
        import torch
        try:
            t = features[1]
            info = {
                "captions": list(batch.get("caption") or []),
                "t": t.detach().float().flatten().tolist() if torch.is_tensor(t) else [],
                "shape": list(features[0].shape),
            }
            self.batches[features[0].data_ptr()] = info
            while len(self.batches) > REMEMBERED_BATCHES:
                self.batches.pop(next(iter(self.batches)))
        except (IndexError, TypeError, AttributeError):
            pass

    def current_info(self) -> dict[str, Any]:
        try:
            return self.batches.get(self.current[0][0].data_ptr(), {})
        except (IndexError, TypeError, AttributeError):
            return {}

    def image_names(self, info: dict[str, Any]) -> list[str]:
        return [self.captions.get(caption.strip(), caption[:30]) for caption in info.get("captions", [])]

    def install(self) -> None:
        import utils.dataset as dataset_util
        import utils.pipeline as pipeline_util
        from deepspeed.runtime.engine import DeepSpeedEngine

        guard = self
        loader_init = dataset_util.PipelineDataLoader.__init__
        loader_next = dataset_util.PipelineDataLoader.__next__
        module_init = pipeline_util.ManualPipelineModule.__init__
        take_model_step = DeepSpeedEngine._take_model_step

        def init(loader: Any, dataset: Any, model_engine: Any, gradient_accumulation_steps: Any, model: Any,
                 *args: Any, **kwargs: Any) -> None:
            if not getattr(model, "_loralake_guarded", False):
                prepare_inputs = model.prepare_inputs

                def guarded_prepare_inputs(batch: Any, *a: Any, **k: Any) -> Any:
                    features, label = prepare_inputs(batch, *a, **k)
                    guard.remember(features, batch)
                    return features, label

                model.prepare_inputs = guarded_prepare_inputs
                model._loralake_guarded = True
            loader_init(loader, dataset, model_engine, gradient_accumulation_steps, model, *args, **kwargs)

        def next_batch(loader: Any) -> Any:
            epoch = loader.epoch  # 取出一个 epoch 的最后一个 batch 时 diffusion-pipe 就会加一，所以先记下
            item = loader_next(loader)
            guard.current = item
            guard.epoch = epoch
            return item

        def pipeline_init(module: Any, *args: Any, loss_fn: Any = None, **kwargs: Any) -> None:
            if loss_fn is not None:
                inner = loss_fn

                def loss_fn(output: Any, label: Any) -> Any:
                    loss = inner(output, label)
                    guard.loss = loss.detach()
                    return loss
            module_init(module, *args, loss_fn=loss_fn, **kwargs)

        def guarded_step(engine: Any, *args: Any, **kwargs: Any) -> Any:
            guard.check(engine)
            return take_model_step(engine, *args, **kwargs)

        dataset_util.PipelineDataLoader.__init__ = init
        dataset_util.PipelineDataLoader.__next__ = next_batch
        pipeline_util.ManualPipelineModule.__init__ = pipeline_init
        DeepSpeedEngine._take_model_step = guarded_step

    # ---- 每一步更新权重之前 ----

    def check(self, engine: Any) -> None:
        import torch
        if self.params is None:
            self.params = [(name, p) for name, p in engine.module.named_parameters() if p.requires_grad]
        with_grad = [(name, p) for name, p in self.params if p.grad is not None]
        if not with_grad:
            return
        grad_norms = torch.stack(torch._foreach_norm([p.grad for _, p in with_grad])).float()
        weight_max = torch.stack(torch._foreach_norm([p.detach() for _, p in with_grad], ord=float("inf"))).float().max()
        finite = torch.isfinite(grad_norms)
        loss = self.loss.float() if self.loss is not None else torch.tensor(float("nan"))
        summary = torch.stack([loss.to(grad_norms.device), grad_norms.norm(), weight_max, finite.all().float()]).tolist()
        loss_value, grad_norm, weight_max_value, all_finite = summary

        step = engine.global_steps + 1
        info = self.current_info()
        t = info.get("t") or [float("nan")]
        epoch = f"epoch {self.epoch} " if self.epoch is not None else ""
        _say(f"step {step} {epoch}loss {loss_value:.4g} grad_norm {grad_norm:.4g} lora_max {weight_max_value:.4g} "
             f"t {min(t):.3f}-{max(t):.3f} images {','.join(self.image_names(info))}")
        if not all_finite:
            bad = [with_grad[i][0] for i in torch.nonzero(~finite).flatten().tolist()]
            self.on_nonfinite(engine, step, loss_value, bad, len(with_grad), info)

    def on_nonfinite(self, engine: Any, step: int, loss: float, bad: list[str], total: int,
                     info: dict[str, Any]) -> None:
        import torch
        self.events_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step{step:06d}-{int(time.time())}"
        event_file = self.events_dir / f"{stem}.json"
        dump_file = self.events_dir / f"{stem}.pt"
        layers = sorted({int(m[1]) for name in bad if (m := LAYER_INDEX.match(name))})
        event: dict[str, Any] = {
            "step": step,
            "epoch": self.epoch,
            "seed": self.seed,
            "loss": loss,
            "t": info.get("t", []),
            "latent_shape": info.get("shape"),
            "images": self.image_names(info),
            "nonfinite_grad_params": len(bad),
            "trainable_params": total,
            "nonfinite_grad_layers": layers,
            "nonfinite_grad_examples": bad[:MAX_LISTED],
            "dump": dump_file.name,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        }
        _say(f"第 {step} 步出现非有限梯度（{len(bad)}/{total} 个 LoRA 参数），不更新权重，停止训练")
        self._write(event_file, event)
        try:
            features, label = self.current
            torch.save({"step": step, "seed": self.seed, "captions": info.get("captions", []),
                        "features": _to_cpu(features), "label": _to_cpu(label),
                        "lora": {name: p.detach().cpu() for name, p in self.params or []}}, dump_file)
        except Exception as e:  # 诊断失败不能影响退出
            event["dump_error"] = repr(e)
        event["replay"] = self.replay(engine)
        self._write(event_file, event)
        _say(f"诊断信息: {event_file}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(NAN_EXIT_CODE)

    def replay(self, engine: Any) -> dict[str, Any]:
        """用同一个 batch（同样的噪声和 t）再跑一遍前向和反向，记下最先出现非有限值的模块。"""
        import torch
        if self.current is None:
            return {"error": "没有记录到这一步的输入"}
        result: dict[str, Any] = {"first_nonfinite_forward": None, "nonfinite_backward_order": []}
        peaks: dict[str, float] = {}
        handles = []

        def forward_hook(name: str) -> Any:
            def hook(module: Any, inputs: Any, output: Any) -> None:
                tensors = [output] if torch.is_tensor(output) else [
                    x for x in (output if isinstance(output, (tuple, list)) else []) if torch.is_tensor(x)]
                for tensor in tensors:
                    if not tensor.is_floating_point():
                        continue
                    if result["first_nonfinite_forward"] is None and not torch.isfinite(tensor).all():
                        result["first_nonfinite_forward"] = name
                    if "." not in name:  # 流水线的顶层层（每个 transformer block 一层）
                        peaks[name] = max(peaks.get(name, 0.0), tensor.detach().abs().max().float().item())
                    if tensor.requires_grad:
                        tensor.register_hook(backward_hook(name))
            return hook

        def backward_hook(name: str) -> Any:
            def hook(grad: Any) -> None:
                order = result["nonfinite_backward_order"]
                if len(order) < MAX_LISTED and name not in order and not torch.isfinite(grad).all():
                    order.append(name)
            return hook

        try:
            module = engine.module
            for _, p in self.params or []:
                p.grad = None
            for name, submodule in module.named_modules():
                if name:
                    handles.append(submodule.register_forward_hook(forward_hook(name)))
            features, label = self.current
            device = engine.device
            features = tuple(x.to(device) if torch.is_tensor(x) else x for x in features)
            label = tuple(x.to(device) if torch.is_tensor(x) else x for x in label)
            with torch.enable_grad():
                loss = module.loss_fn(module(features), label)
                result["loss"] = loss.float().item()
                loss.backward()
            bad = [name for name, p in self.params or [] if p.grad is not None and not torch.isfinite(p.grad).all()]
            result["reproduced"] = bool(bad) or not torch.isfinite(loss).item()
            result["nonfinite_grad_params"] = len(bad)
            result["largest_layer_outputs"] = dict(sorted(peaks.items(), key=lambda kv: -kv[1])[:5])
        except Exception as e:
            result["error"] = repr(e)
        finally:
            for handle in handles:
                handle.remove()
        return result

    @staticmethod
    def _write(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _to_cpu(value: Any) -> Any:
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu(x) for x in value)
    return value


def main() -> None:
    pipe_dir = Path.cwd()
    # 脚本所在目录（loralake/train）里有 models.py、settings.py，会挡住 diffusion-pipe 的同名包
    sys.path[0] = str(pipe_dir)
    config = tomllib.loads(Path(_arg_value("--config")).read_text(encoding="utf-8"))
    seed = int(os.environ.get(SEED_ENV) or random.SystemRandom().randrange(2**31))

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _say(f"seed {seed}（设置环境变量 {SEED_ENV}={seed} 可以复现）")

    Guard(seed, Path(config["output_dir"]) / EVENTS_DIR, _caption_index(config)).install()
    train_script = pipe_dir / "train.py"
    sys.argv[0] = str(train_script)
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
