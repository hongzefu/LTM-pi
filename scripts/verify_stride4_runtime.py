#!/usr/bin/env python3
"""真实执行stride4缓存、20→21恢复及在线对拍；未执行的阶段绝不记为通过。"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
CONFIG = "pi05_robomme_16x100_dreamdojo_lmv_causal_fixedlag3_stride4_anchor4x4_h50_default"
GIB = 1024**3
GLOBAL_RELATIVE_L2_TOLERANCE = 0.01
NORMALIZED_TOKEN_RELATIVE_L2_TOLERANCE = 0.02


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_env(path: Path) -> dict[str, str]:
    """按现有shell入口解析指定环境，路径作为独立参数传递。"""
    require(path.is_file(), f"环境文件不存在：{path}")
    result = subprocess.run(
        ["bash", "-c", 'set -ae; source "$1" >&2; env -0', "runtime-env", str(path.resolve())],
        check=True,
        stdout=subprocess.PIPE,
    )
    return dict(item.decode().split("=", 1) for item in result.stdout.split(b"\0") if b"=" in item)


def overlay_env(source: Path, target: Path, overrides: dict[str, str]) -> Path:
    target.write_text(
        "source "
        + shlex.quote(str(source.resolve()))
        + "\n"
        + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in overrides.items())
    )
    return target


def directory_bytes(path: Path) -> int:
    """实际分配字节；容忍Orbax扫描期间原子换名。"""
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                file = Path(root) / name
                if not file.is_symlink():
                    total += file.stat().st_blocks * 512
            except FileNotFoundError:
                pass
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_command(
    argv: list[str], *, env: dict[str, str], log: Path, watch: dict[str, Path], timeout: float, interval: float = 1.0
) -> dict[str, Any]:
    """留完整输出并采样磁盘；异常时回收整个子进程组。"""
    peaks = {key: directory_bytes(path) for key, path in watch.items()}
    baseline = dict(peaks)
    rows = []
    stop = threading.Event()
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )

    def monitor():
        while not stop.wait(interval):
            for key, path in watch.items():
                peaks[key] = max(peaks[key], directory_bytes(path))
            if time.monotonic() - started > timeout:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    if not stop.wait(15):
                        os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with log.open("w") as stream:
            assert process.stdout is not None
            for raw in iter(process.stdout.readline, b""):
                line = raw.decode(errors="replace")
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
                for match in re.finditer(r"Step (\d+): ([^\r\n]+)", line):
                    rows.append(
                        {"step": int(match[1]), "metrics": match[2], "elapsed_seconds": time.monotonic() - started}
                    )
            code = process.wait()
        require(code == 0, f"子进程退出码{code}；日志：{log}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop.set()
        thread.join(timeout=20)
        if process.stdout is not None:
            process.stdout.close()
    for key, path in watch.items():
        peaks[key] = max(peaks[key], directory_bytes(path))
    return {
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
        "baseline_bytes": baseline,
        "peak_sampled_bytes": peaks,
    }


def verify_train_rows(rows: list[dict], first: int, stop: int) -> dict:
    require([row["step"] for row in rows] == list(range(first, stop)), f"训练日志缺步或重复：期望[{first},{stop})")
    for row in rows:
        metrics = dict(re.findall(r"([A-Za-z_][A-Za-z_0-9/]*)=([^,\s]+)", row["metrics"]))
        for key in ("loss", "grad_norm", "learning_rate", "completed_step"):
            require(key in metrics, f"Step {row['step']}缺少{key}")
            require(math.isfinite(float(metrics[key])), f"Step {row['step']} {key}非有限")
        require(float(metrics["completed_step"]) == row["step"] + 1, "完成更新数错误")
        expected = 2.5e-5 / 1001 + row["step"] / 1000 * (2.5e-5 - 2.5e-5 / 1001)
        require(
            math.isclose(float(metrics["learning_rate"]), expected, rel_tol=2e-5, abs_tol=1e-11),
            "学习率没有沿用正式时程",
        )
        require(float(metrics["grad_norm"]) >= 0, "梯度范数为负")
    intervals = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(rows, rows[1:]) if b["step"] >= 5]
    return {
        "updates": stop - first,
        "stable_step_seconds_median": statistics.median(intervals) if intervals else None,
        "stable_step_seconds_samples": intervals,
    }


def numeric_parity(
    actual,
    reference,
    mean,
    std,
    *,
    global_tolerance: float = GLOBAL_RELATIVE_L2_TOLERANCE,
    normalized_token_tolerance: float = NORMALIZED_TOKEN_RELATIVE_L2_TOLERANCE,
) -> dict:
    import numpy as np

    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    require(actual.shape == reference.shape and actual.ndim == 2, "LAM形状不同")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "LAM含非有限值")
    tolerances = {
        "global_relative_l2_tolerance": global_tolerance,
        "normalized_token_relative_l2_tolerance": normalized_token_tolerance,
    }
    if not len(reference):
        return {"tokens": 0, "relative_l2": 0.0, "normalized_token_relative_l2_max": 0.0, "max_abs": 0.0, **tolerances}
    mean, std = np.asarray(mean), np.asarray(std)
    require(mean.shape == (actual.shape[1],) and std.shape == mean.shape and (std > 0).all(), "归一化形状错误")
    difference = actual - reference
    relative = float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-6))
    maximum = float(
        (
            np.linalg.norm(difference / std, axis=1)
            / np.maximum(np.linalg.norm((reference - mean) / std, axis=1), 1e-6)
        ).max()
    )
    require(
        relative <= global_tolerance and maximum <= normalized_token_tolerance,
        f"LAM超限：全局={relative}，全局阈值={global_tolerance}；"
        f"归一化逐token最大={maximum}，逐token阈值={normalized_token_tolerance}",
    )
    return {
        "tokens": len(reference),
        "relative_l2": relative,
        "normalized_token_relative_l2_max": maximum,
        "max_abs": float(np.abs(difference).max()),
        **tolerances,
    }


def effective_config(env_file: Path):
    from openpi.training import config as configs

    result = subprocess.run(
        ["bash", "bin/train.sh", "print-args"],
        cwd=ROOT,
        env={**os.environ, "ROBOMME_ENV_FILE": str(env_file)},
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    argv = json.loads(result.stdout)
    require(isinstance(argv, list) and argv[0] == CONFIG, "print-args没有返回新配置实际参数")
    original = sys.argv
    try:
        sys.argv = ["train.py", *argv]
        config = configs.cli()
    finally:
        sys.argv = original
    require(config.batch_size == 128 and config.fsdp_devices == 8, "必须B128/8卡")
    require(config.model.memory_stride == 4 and config.model.memory_anchor_pool_grid == 4, "不是stride4/grid4")
    require(config.lr_schedule.decay_steps == 60000 and config.lr_schedule.warmup_steps == 1000, "必须正式60k时程")
    require(config.training_contract_path is not None, "缺少训练合同")
    return config


def memory_dataset(config):
    from openpi.training import data_loader
    from openpi.training.cdlam_memory_dataset import DUAL_ANCHOR_IMAGE_DEFINITION, CDLAMMemoryDataset

    data = config.data.create(config.assets_dirs, config.model)
    require(
        data.memory_random_drop_execution_tail_min == data.memory_random_drop_execution_tail_max == 3, "离线lag不是3"
    )
    require(
        Path(data_loader._resolve_dataset_root(data.repo_id)).resolve() == Path(data.repo_id).resolve(),
        "实际数据根被覆盖",
    )
    base = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    dataset = CDLAMMemoryDataset(
        base,
        data.memory_cache_dir,
        memory_horizon=config.model.memory_horizon,
        latent_dim=config.model.memory_latent_dim,
        memory_stride=4,
        expected_dataset_root=data.repo_id,
        expected_latent_mean=config.model.memory_latent_mean,
        expected_latent_std=config.model.memory_latent_std,
        random_drop_execution_tail_min=3,
        random_drop_execution_tail_max=3,
        phase_metadata_cache_dir=data.phase_metadata_cache_dir,
        demo_anchor_cache_dir=data.demo_anchor_cache_dir,
        expected_demo_anchor_definition=DUAL_ANCHOR_IMAGE_DEFINITION,
        execution_anchor=True,
    )
    return data, dataset


def cache_check(env_file: Path) -> dict:
    import numpy as np

    from openpi.shared import ltmpi_contract
    from openpi.training import data_loader

    config = effective_config(env_file)
    ltmpi_contract.prepare_training_contract(config)
    data, dataset = memory_dataset(config)
    manifest = json.loads((Path(data.memory_cache_dir) / "manifest.json").read_text())
    require(manifest.get("source_input_pin"), "缓存没有生成时输入身份")
    for key, name in (("global_latents", "latents.npy"), ("global_frame_index", "frame_index.npz")):
        path = Path(data.memory_cache_dir) / manifest.get(key, name)
        require(sha256(path) == manifest.get(key + "_sha256"), f"缓存hash错误：{path}")
    indices = dataset._frame_index
    require(np.array_equal(dataset.memory_lengths, np.maximum(indices // 4, 1)), "sampler长度错误")
    require(np.array_equal(dataset._valid, (indices > 0) & (indices % 4 == 0)), "有效端点错误")
    for start in range(0, len(dataset), 65536):
        values, valid = np.asarray(dataset._latents[start : start + 65536]), dataset._valid[start : start + 65536]
        require(np.isfinite(values).all() and not np.any(values[~valid]), "缓存非有限或无效行非零")
    boundaries = dataset._exec_start_by_episode[dataset._episode_index]
    execution = np.flatnonzero(indices >= boundaries)
    require(len(execution) >= 128, f"只有{len(execution)}个执行样本，不能构成B128")
    selected = execution[np.linspace(0, len(execution) - 1, 128).astype(int)]
    transformed = data_loader.transform_dataset(dataset, data)
    batch = data_loader._collate_fn(
        [transformed[int(i)] for i in selected],
        memory_pad_to_multiple=512,
        memory_max_length=1800,
        trim_prompt_padding=False,
    )
    require(batch["memory_latents"].shape == (128, 512, 64), "实际B128 padding不是512×64")
    valid_max = int(np.asarray(batch["memory_mask"]).sum(axis=1).max())
    require(valid_max <= 449, "有效LAM超过449")
    return {
        "status": "pass",
        "dataset_frames": len(dataset),
        "execution_samples": len(execution),
        "source_input_pin": manifest["source_input_pin"],
        "valid_lam_max": valid_max,
        "padded_lam_slots": 512,
        "memory_group_slots": 545,
        "prefix_slots": 1257,
    }


def checkpoint_counters(checkpoint: Path) -> dict:
    """只恢复step和优化器计数标量；不在CPU复制数十GB权重。"""
    import numpy as np
    import orbax.checkpoint as ocp

    from scripts.serve import _checkpoint_is_committed

    _checkpoint_is_committed(checkpoint)
    metadata = json.loads((checkpoint / "train_state" / "_METADATA").read_text())
    paths = []
    for value in metadata.get("tree_metadata", {}).values():
        path = tuple(str(item["key"]) for item in value.get("key_metadata", []))
        if path == ("step",) or ("opt_state" in path and path[-1:] == ("count",)):
            paths.append(path)
    require(("step",) in paths and any("opt_state" in p for p in paths), "找不到step或优化器count")
    item = {f"leaf_{i}": np.asarray(0, dtype=np.int32) for i in range(len(paths))}
    transforms = {f"leaf_{i}": ocp.Transform(original_key="/".join(path)) for i, path in enumerate(paths)}
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            checkpoint / "train_state",
            item=item,
            transforms=transforms,
            restore_args={key: ocp.RestoreArgs(restore_type=np.ndarray) for key in item},
        )
    values = {"/".join(path): int(np.asarray(restored[f"leaf_{i}"])) for i, path in enumerate(paths)}
    expected = int(checkpoint.name)
    require(all(value == expected for value in values.values()), f"checkpoint计数错误：{values}")
    parts = {name: directory_bytes(checkpoint / name) for name in ("params", "train_state", "assets")}
    require(all(size > 0 for size in parts.values()), "checkpoint缺少params/train_state/assets")
    return {
        "step": expected,
        "counts": values,
        "parts_bytes": parts,
        "complete_checkpoint_bytes": directory_bytes(checkpoint),
    }


class CapturePolicy:
    """捕获真实MemorySession输入；最后单独启用真正动作前向。"""

    def __init__(self, policy):
        self.policy, self.element, self.forward = policy, None, False

    def infer(self, element, *, noise):
        self.element = element
        return self.policy.infer(element, noise=noise) if self.forward else {}


def offline_visible(latents, valid, frame_index, boundary: int, last: int):
    import numpy as np

    from openpi.training.cdlam_memory_dataset import classify_memory_segments, drop_execution_lam_tail

    selected = valid & (frame_index <= last)
    endpoints = frame_index[selected]
    segments = classify_memory_segments(endpoints, boundary)
    visible_endpoints, _, visible_segments = drop_execution_lam_tail(
        endpoints[:, None], np.ones(len(endpoints), dtype=bool), segments, 3
    )
    values, _, _ = drop_execution_lam_tail(latents[selected], np.ones(len(endpoints), dtype=bool), segments, 3)
    return values, visible_segments, visible_endpoints[:, 0]


def compare_memory(element: dict, reference, segments, mean, std) -> dict:
    import numpy as np

    mask = np.asarray(element["memory_mask"])
    actual = np.asarray(element["memory_latents"])
    ids = np.asarray(element["memory_segment_ids"])
    require(mask.shape == (1800,) and actual.shape == (1800, 64), "在线padding不是1800")
    require(not np.any(actual[~mask]) and not np.any(ids[~mask]), "在线padding非零")
    require(np.array_equal(ids[mask], segments), "有效segment顺序不同")
    require(np.array_equal(mask, np.arange(1800) >= 1800 - len(reference)), "mask不是左padding")
    return numeric_parity(actual[mask], reference, mean, std)


def verify_wrong_contract_rejected(checkpoint: Path, config, manifest_path: Path, validator) -> int:
    """交换现有anchor类型编号，保持词表与参数形状；只认模型身份拒绝。"""
    demo_id = config.model.memory_demo_anchor_type_id
    execution_id = config.model.memory_execution_anchor_type_id
    require(demo_id != execution_id, "无法构造不同的anchor类型配置")
    wrong_model = dataclasses.replace(
        config.model,
        memory_demo_anchor_type_id=execution_id,
        memory_execution_anchor_type_id=demo_id,
    )
    wrong_config = dataclasses.replace(config, model=wrong_model)
    try:
        validator(checkpoint, wrong_config, manifest_path=manifest_path)
    except ValueError as error:
        require("model_config_sha256" in str(error), f"错误配置并非因模型合同身份被拒绝：{error}")
        return 1
    raise ValueError("同形状的错误anchor配置未被保存件合同拒绝")


def online_check(env_file: Path, checkpoint: Path, pair_batch_size: int) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from openpi.models import model as model_lib
    from openpi.models.pi0 import pool_anchor_tokens
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies import policy_config
    from openpi.shared import ltmpi_contract, nnx_utils
    from scripts import precompute_dreamdojo_memory as cache
    from scripts.serve import MemorySession, _checkpoint_is_committed

    require(jax.default_backend() == "gpu", "真实在线验收需要GPU")
    config = effective_config(env_file)
    manifest_path = Path(os.environ["ROBOMME_MEMORY_CACHE"]) / "manifest.json"
    _checkpoint_is_committed(checkpoint)
    contract = ltmpi_contract.validate_checkpoint_contract(checkpoint, config, manifest_path=manifest_path)
    wrong_contract_rejected = verify_wrong_contract_rejected(
        checkpoint, config, manifest_path, ltmpi_contract.validate_checkpoint_contract
    )
    source_identity = ltmpi_contract.validate_source_input_manifest(
        Path(os.environ["ROBOMME_INPUT_MANIFEST"]), Path(config.data.repo_id)
    )
    require(source_identity["source_input_pin"] == contract["semantic"]["source_input_pin"], "恢复用源数据身份不同")
    manifest = json.loads(manifest_path.read_text())
    mean = np.asarray(manifest["latent_normalization"]["mean"])
    std = np.asarray(manifest["latent_normalization"]["std"])
    policy = policy_config.create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": 10})
    identity = cache.resolve_checkpoint_identity(
        Path(os.environ["DREAMDOJO_CHECKPOINT"]), explicit_sha256=manifest["checkpoint_sha256"], verify_sha256=True
    )
    encoder, encode, preprocess, identity = cache.load_dreamdojo_encoder(
        Path(os.environ.get("DREAMDOJO_RUNTIME_ROOT", ROOT / "third_party/dreamdojo/runtime")), identity, "cuda:0"
    )
    data, dataset = memory_dataset(config)
    spec = cache.discover_dataset(Path(data.repo_id))
    source = next(
        (
            item
            for item in spec.episodes
            if item.expected_length >= int(dataset._exec_start_by_episode[item.episode_index]) + 24
            and item.expected_length >= 48
        ),
        None,
    )
    require(source is not None, "缺少至少24帧执行尾段的真实episode，覆盖不足")
    head = cache.read_episode(source, spec.root, "image")
    wrist = cache.read_episode(source, spec.root, "wrist_image")
    rows = np.flatnonzero(dataset._episode_index == source.episode_index)
    require(np.array_equal(dataset._frame_index[rows], head["frame_index"]), "缓存和真实episode行不同")
    raw_latents, valid = np.asarray(dataset._latents[rows]), dataset._valid[rows]
    actual_boundary = int(dataset._exec_start_by_episode[source.episode_index])
    tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
    capture = CapturePolicy(policy)
    limit = max(actual_boundary + 24, 35)
    prepared = [cache.preprocess_frames(view["frames"][:limit], preprocess, chunk_size=32) for view in (head, wrist)]

    class CheckedSession(MemorySession):
        def add_buffer(self, message):
            end = self.total_frames + len(message["images"])
            self.checked_endpoints = np.arange(max(4, ((self.total_frames + 3) // 4) * 4), end, 4)
            self.checked_view = 0
            result = super().add_buffer(message)
            require(self.checked_view == (2 if len(self.checked_endpoints) else 0), "双视角帧对调用次数错误")
            return result

        def _encode_dreamdojo_pairs(self, frames):
            expected_indices = np.concatenate(([self.checked_endpoints[0] - 4], self.checked_endpoints))
            require(
                np.array_equal(frames, prepared[self.checked_view][expected_indices]),
                "真实编码器输入帧对偏离0→4→8时间线",
            )
            self.checked_view += 1
            return super()._encode_dreamdojo_pairs(frames)

    def session():
        return CheckedSession(
            policy=capture,
            dreamdojo_model=encoder,
            encode_full=encode,
            preprocess=preprocess,
            device="cuda:0",
            memory_horizon=1800,
            latent_dim=64,
            action_horizon=50,
            action_dim=config.model.action_dim,
            prompt_tokenizer=tokenizer,
            max_token_len=200,
            pair_batch_size=pair_batch_size,
            use_memory_segments=True,
            use_demo_anchor=True,
            use_execution_anchor=True,
            transition_frame_stride=4,
            two_view_memory=True,
            memory_lag=3,
            latent_storage_dtype="float32",
        )

    results, final_session, final_message, anchor_pair = [], None, None, None
    # 官方标签对拍和真实帧的人工边界回放分别留证，人工边界不冒充官方数据。
    for boundary, official in [(actual_boundary, True), (0, False), (8, False), (9, False), (10, False), (11, False)]:
        stream = session()
        maximum, last_element = boundary + 23, None
        for last in range(boundary, maximum + 1):
            start = stream.total_frames
            stream.add_buffer(
                {
                    "images": head["frames"][start : last + 1],
                    "wrist_images": wrist["frames"][start : last + 1],
                    "exec_start_idx": boundary if start == 0 else 0,
                }
            )
            sample = dataset[int(rows[last])]
            element = sample
            for transform in data.repack_transforms.inputs:
                element = transform(element)
            message = {key: value for key, value in element.items() if not key.startswith("memory_")}
            stream.infer(message)
            current = capture.element
            reference, segments, endpoints = offline_visible(raw_latents, valid, head["frame_index"], boundary, last)
            parity = compare_memory(current, reference, segments, mean, std)
            require(stream.total_transitions == last // 4, "真实Session端点数量错误")
            all_endpoints = np.arange(4, last + 1, 4)
            expected_ids = np.where(
                all_endpoints < boundary, 1, np.where((boundary > 0) & (all_endpoints == boundary), 2, 3)
            )
            require(np.array_equal(np.asarray(stream.segment_ids), expected_ids), "端点segment顺序错误")
            require(bool(current["memory_demo_start_mask"]) == (boundary > 0), "demo anchor mask错误")
            require(bool(current["memory_execution_start_mask"]), "execution anchor mask错误")
            require(
                np.array_equal(
                    current["memory_demo_start_image"],
                    head["frames"][0] if boundary else np.zeros_like(head["frames"][0]),
                ),
                "demo anchor原图不同",
            )
            require(
                np.array_equal(current["memory_execution_start_image"], head["frames"][boundary]),
                "execution anchor原图不同",
            )
            if official:
                offline_mask = np.asarray(sample["memory_mask"])
                require(np.array_equal(sample["memory_segment_ids"][offline_mask], segments), "真实离线segment不同")
                require(np.array_equal(sample["memory_latents"][offline_mask], reference), "真实离线token顺序不同")
                for key in (
                    "memory_demo_start_image",
                    "memory_execution_start_image",
                    "memory_demo_start_mask",
                    "memory_execution_start_mask",
                ):
                    require(np.array_equal(sample[key], current[key]), f"真实离线{key}不同")
                anchor_pair, final_session, final_message = (element, current), stream, message
            results.append(
                {
                    "official_boundary": official,
                    "boundary": boundary,
                    "last_frame": last,
                    "visible_endpoints": endpoints.tolist(),
                    **parity,
                }
            )
            last_element = current
        one = session()
        one.add_buffer(
            {
                "images": head["frames"][: maximum + 1],
                "wrist_images": wrist["frames"][: maximum + 1],
                "exec_start_idx": boundary,
            }
        )
        one.infer(message)
        mask = np.asarray(last_element["memory_mask"])
        compare_memory(
            capture.element, last_element["memory_latents"][mask], last_element["memory_segment_ids"][mask], mean, std
        )
        # 同batch/dtype受控重复；跨batch阈值不能代替逐位判据。
        duplicate = session()
        duplicate.add_buffer(
            {
                "images": head["frames"][: maximum + 1],
                "wrist_images": wrist["frames"][: maximum + 1],
                "exec_start_idx": boundary,
            }
        )
        require(
            np.array_equal(np.asarray(one.latents), np.asarray(duplicate.latents)), "同batch的LAM重复编码不逐位相同"
        )

    require(anchor_pair is not None and final_session is not None, "没有完成官方样本恢复")
    observations = []
    for element in anchor_pair:
        transformed = policy._input_transform(dict(element))
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None], transformed)
        observation = model_lib.Observation.from_dict(batched)
        observations.append(model_lib.preprocess_observation(None, observation, train=False))
    image_inputs = [
        jnp.concatenate([obs.memory_demo_start_image, obs.memory_execution_start_image], axis=0) for obs in observations
    ]
    vision = nnx_utils.module_jit(policy._model.PaliGemma.img.__call__, static_argnames=("train",))
    features = [vision(images, train=False)[0] for images in image_inputs]
    pooled = [pool_anchor_tokens(feature, 4) for feature in features]
    require(
        pooled[0].shape[1] == 16 and np.array_equal(np.asarray(pooled[0]), np.asarray(pooled[1])),
        "双anchor池化不逐位相同",
    )
    require(all(np.isfinite(np.asarray(value, dtype=np.float32)).all() for value in pooled), "anchor池化非有限")
    capture.forward = True
    actions = np.asarray(final_session.infer(final_message)["actions"])
    require(actions.shape == (50, 8) and np.isfinite(actions).all(), f"真实动作块错误：{actions.shape}")
    return {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "episode_index": source.episode_index,
        "policy_loaded": True,
        "dreamdojo_loaded": True,
        "wrong_contract_rejected": wrong_contract_rejected,
        "global_relative_l2_tolerance": GLOBAL_RELATIVE_L2_TOLERANCE,
        "normalized_token_relative_l2_tolerance": NORMALIZED_TOKEN_RELATIVE_L2_TOLERANCE,
        "actions_shape": list(actions.shape),
        "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "boundary_residues": [0, 1, 2, 3],
        "controlled_boundaries": [0, 8, 9, 10, 11],
        "real_encoder_frame_pairs_checked": True,
        "same_batch_lam_bitwise": True,
        "same_batch_anchor_pool_bitwise": True,
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "cache-smoke", "train-smoke", "online", "restore"), default="smoke")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--timeout-hours", type=float, default=6)
    parser.add_argument("--worker", choices=("cache", "checkpoint", "online"), help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    env = read_env(args.env_file)
    if args.worker:
        # 环境文件可含8卡训练设置；worker的隔离设置由父进程确定，不允许被覆盖。
        isolation = {
            key: os.environ[key]
            for key in (
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_COMPILATION_CACHE_DIR",
            )
            if key in os.environ
        }
        os.environ.update(env)
        os.environ.update(isolation)
        if args.worker == "cache":
            result = cache_check(args.env_file)
        elif args.worker == "checkpoint":
            result = checkpoint_counters(args.checkpoint)
        else:
            result = online_check(args.env_file, args.checkpoint, args.pair_batch_size)
        write_json(args.result, result)
        return 0
    require(env.get("ROBOMME_CONFIG") == CONFIG, "只支持本任务新配置")
    require(args.pair_batch_size > 0 and args.timeout_hours > 0, "batch/timeout必须为正")
    output = (args.output_dir or Path(env["RUN_BASE_DIR"]) / env["ROBOMME_EXP"] / "runtime-verification").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "mode": args.mode,
        "fork_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "stages": {},
    }
    report_path = output / (args.mode + "-result.json")
    python = env.get("PYTHON_BIN", sys.executable)
    checkpoint_root = Path(env["CHECKPOINT_BASE_DIR"]) / CONFIG / env["ROBOMME_EXP"]
    hf_cache = Path(env.get("HF_CACHE", "/scratch/hongze/.cache/huggingface"))
    arrow_cache = (
        output / f"arrow-cache-{args.mode}"
        if args.mode in ("smoke", "cache-smoke", "train-smoke")
        else Path(env.get("HF_DATASETS_CACHE", str(hf_cache / "datasets")))
    )
    watch = {"checkpoint": checkpoint_root, "hf_cache": hf_cache, "arrow_cache": arrow_cache}
    initial_arrow_bytes = directory_bytes(arrow_cache)
    stage_measurements = []
    base_env = {
        **env,
        "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "JAX_COMPILATION_CACHE_DIR": "/scratch/hongze/.cache/jax",
        "HF_HOME": str(hf_cache),
        "HF_DATASETS_CACHE": str(arrow_cache),
    }

    def invoke(stage: str, env_file: Path, checkpoint: Path | None = None):
        result_path = output / f"{stage}-{checkpoint.name if checkpoint else 'cache'}.json"
        command = [
            python,
            str(Path(__file__).resolve()),
            "--env-file",
            str(env_file),
            "--worker",
            stage,
            "--result",
            str(result_path),
            "--pair-batch-size",
            str(args.pair_batch_size),
        ]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        child_env = {
            **base_env,
            "JAX_PLATFORMS": "cuda" if stage == "online" else "cpu",
            "CUDA_VISIBLE_DEVICES": "0" if stage == "online" else "",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        stage_measurements.append(
            run_command(
                command,
                env=child_env,
                log=result_path.with_suffix(".log"),
                watch=watch,
                timeout=args.timeout_hours * 3600,
            )
        )
        require(result_path.is_file(), "子阶段没有产出结果")
        return json.loads(result_path.read_text())

    try:
        if args.mode in ("smoke", "cache-smoke", "train-smoke"):
            require(initial_arrow_bytes == 0, f"容量实测需要空的独立Arrow缓存，请使用新的--output-dir：{arrow_cache}")
            require(env.get("ROBOMME_UPLOAD_RECEIPTS_DIR", "") == "", "冒烟必须关闭上传回执")
            require("smoke" in env["ROBOMME_EXP"].lower(), "冒烟run名必须含smoke，避免误用正式目录")
            common = {
                "ROBOMME_NUM_TRAIN_STEPS": "20",
                "ROBOMME_LR_DECAY_STEPS": "60000",
                "ROBOMME_SAVE_STEPS": "20",
                "ROBOMME_LOG_INTERVAL": "1",
                "ROBOMME_RESUME": "0",
                "ROBOMME_UPLOAD_RECEIPTS_DIR": "",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
                "CACHE_GPUS": "0,1,2,3,4,5,6,7",
                "TRANSITION_FRAME_STRIDE": "4",
                "HF_DATASETS_CACHE": str(arrow_cache),
            }
            smoke_env = overlay_env(args.env_file, output / "smoke20.env", common)
            # train.sh允许调用方覆盖环境文件，故必须同时覆盖父环境和overlay。
            build_env = {**base_env, **common, "JAX_PLATFORMS": "cuda", "ROBOMME_ENV_FILE": str(smoke_env)}
            if args.mode != "train-smoke" and not (Path(env["ROBOMME_MEMORY_CACHE"]) / "manifest.json").exists():
                run_command(
                    ["bash", "bin/prepare.sh", "memory"],
                    env=build_env,
                    log=output / "cache-build.log",
                    watch=watch,
                    timeout=args.timeout_hours * 3600,
                )
            report["stages"]["cache"] = invoke("cache", smoke_env)
            print("S4_CACHE_SMOKE=PASS scope=complete_smoke_dataset", flush=True)
            write_json(report_path, report)
            if args.mode != "cache-smoke":
                require(not checkpoint_root.exists(), f"冒烟checkpoint目录已存在，不覆盖：{checkpoint_root}")
                first = run_command(
                    ["bash", "bin/train.sh", "run"],
                    env=build_env,
                    log=output / "train20.log",
                    watch=watch,
                    timeout=args.timeout_hours * 3600,
                )
                first_metrics = verify_train_rows(first["rows"], 0, 20)
                saved20 = invoke("checkpoint", smoke_env, checkpoint_root / "20")
                resume_env = overlay_env(
                    args.env_file,
                    output / "smoke21.env",
                    {**common, "ROBOMME_NUM_TRAIN_STEPS": "21", "ROBOMME_SAVE_STEPS": "21", "ROBOMME_RESUME": "1"},
                )
                second = run_command(
                    ["bash", "bin/train.sh", "run"],
                    env={
                        **build_env,
                        "ROBOMME_NUM_TRAIN_STEPS": "21",
                        "ROBOMME_SAVE_STEPS": "21",
                        "ROBOMME_RESUME": "1",
                        "ROBOMME_ENV_FILE": str(resume_env),
                    },
                    log=output / "resume21.log",
                    watch=watch,
                    timeout=args.timeout_hours * 3600,
                )
                verify_train_rows(second["rows"], 20, 21)
                starts = re.findall(r"LTMPI_TRAIN_STATE=(\{[^\r\n]+\})", (output / "resume21.log").read_text())
                require(len(starts) == 1, "没有唯一真实恢复状态日志")
                restored = json.loads(starts[0])
                require(restored["start_step"] == 20, "没有从20恢复")
                counts = restored["optimizer_counts"]
                require(bool(counts) and all(int(value) == 20 for value in counts.values()), "恢复优化器计数错误")
                saved21 = invoke("checkpoint", resume_env, checkpoint_root / "21")
                report["stages"]["training"] = {
                    "status": "pass",
                    "train20": first_metrics,
                    "checkpoint20": saved20,
                    "checkpoint21": saved21,
                    "resume_state": restored,
                }
                complete = max(saved20["complete_checkpoint_bytes"], saved21["complete_checkpoint_bytes"])
                peak = max(first["peak_sampled_bytes"]["checkpoint"], second["peak_sampled_bytes"]["checkpoint"])
                resources = {
                    "schema_version": 1,
                    "status": "pass",
                    "fork_commit": report["fork_commit"],
                    "config_name": CONFIG,
                    "batch_size": 128,
                    "fsdp_devices": 8,
                    "num_train_steps": 20,
                    "resumed_to": 21,
                    "lr_decay_steps": 60000,
                    "contract_sha256": json.loads(Path(env["ROBOMME_TRAINING_CONTRACT"]).read_text())[
                        "contract_sha256"
                    ],
                    "contract_file_sha256": sha256(Path(env["ROBOMME_TRAINING_CONTRACT"])),
                    "complete_checkpoint_bytes": complete,
                    "checkpoint_parts_bytes": saved21["parts_bytes"],
                    "checkpoint_tree_peak_sampled_bytes": peak,
                    "checkpoint_extra_peak_sampled_bytes": max(0, peak - 2 * complete),
                    "arrow_cache_growth_bytes": max(
                        0,
                        max(
                            measurement["peak_sampled_bytes"]["arrow_cache"]
                            for measurement in [first, second, *stage_measurements]
                        )
                        - initial_arrow_bytes,
                    ),
                    "arrow_cache_path": str(arrow_cache),
                    "arrow_cache_baseline_bytes": initial_arrow_bytes,
                    "dataset_frames": report["stages"]["cache"]["dataset_frames"],
                    "hf_cache_peak_sampled_bytes": max(
                        measurement["peak_sampled_bytes"]["hf_cache"]
                        for measurement in [first, second, *stage_measurements]
                    ),
                    "independent_restore_copy_bytes": complete,
                    "disk_safety_bytes": 10 * GIB,
                    "sampling_interval_seconds": 1.0,
                    "peak_is_lower_bound": True,
                    **first_metrics,
                }
                write_json(output / "resources.json", resources)
                print("LTMPI_S4_TRAIN20=PASS updates=20 resumed_step=20 completed_step=21", flush=True)
                write_json(report_path, report)
                if args.mode == "smoke":
                    report["stages"]["online"] = invoke("online", resume_env, checkpoint_root / "21")
                    print(
                        "S4_ONLINE_OFFLINE=PASS real_policy=1 real_dreamdojo=1 finite_actions=1 wrong_contract_rejected=1 "
                        f"global_relative_l2_tolerance={GLOBAL_RELATIVE_L2_TOLERANCE} "
                        f"normalized_token_relative_l2_tolerance={NORMALIZED_TOKEN_RELATIVE_L2_TOLERANCE}",
                        flush=True,
                    )
        else:
            require(args.checkpoint is not None, "online/restore必须指定--checkpoint")
            report["stages"]["checkpoint"] = invoke("checkpoint", args.env_file, args.checkpoint.resolve())
            report["stages"]["online"] = invoke("online", args.env_file, args.checkpoint.resolve())
            print(
                "LTMPI_S4_RESTORE=PASS real_policy=1 real_dreamdojo=1 finite_actions=1 wrong_contract_rejected=1 "
                f"global_relative_l2_tolerance={GLOBAL_RELATIVE_L2_TOLERANCE} "
                f"normalized_token_relative_l2_tolerance={NORMALIZED_TOKEN_RELATIVE_L2_TOLERANCE}",
                flush=True,
            )
        report["status"] = "pass"
        write_json(report_path, report)
        print(f"LTMPI_S4_RUNTIME=PASS mode={args.mode} report={report_path}", flush=True)
        return 0
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(report_path, report)
        print(f"LTMPI_S4_RUNTIME=FAIL mode={args.mode} error={error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
