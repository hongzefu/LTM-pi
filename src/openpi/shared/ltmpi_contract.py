"""本轮 stride-4 数据身份、训练合同和保存回执的共同定义。"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

S4_CONFIG_NAME = "pi05_robomme_16x100_dreamdojo_lmv_causal_fixedlag3_stride4_anchor4x4_h50_default"
_OPERATIONAL_FIELDS = frozenset(
    {
        "num_train_steps",
        "save_steps",
        "save_interval",
        "keep_period",
        "log_interval",
        "resume",
        "overwrite",
        "training_contract_path",
        "checkpoint_upload_receipts_dir",
        "checkpoint_base_dir",
        "exp_name",
        "project_name",
        "wandb_enabled",
        "restore_checkpoint_dir",
        "restore_checkpoint_step",
    }
)
_EXCLUDED_INPUT_NAMES = frozenset({"input_manifest.json", "source_input_manifest.json", "READY"})
_EXCLUDED_INPUT_DIRS = frozenset({"cache", "caches", "derived", "__pycache__"})


def normalize(value: Any) -> Any:
    """把配置转为稳定 JSON，避免对象地址进入合同。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return normalize(value.value)
    if isinstance(value, os.PathLike):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {field.name: normalize(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if callable(getattr(value, "tolist", None)):
        return normalize(value.tolist())
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if hasattr(value, "__dict__"):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "fields": normalize(vars(value))}
    raise TypeError(f"合同无法稳定序列化：{type(value)}")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(normalize(value), handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(value: str | Path | Mapping) -> dict:
    document = dict(value) if isinstance(value, Mapping) else json.loads(Path(value).read_text())
    if not isinstance(document, dict):
        raise ValueError("合同或清单必须为 JSON 对象")
    return document


def validate_dataset_root(declared_root: str | Path, actual_root: str | Path | None = None) -> Path:
    root = Path(declared_root).expanduser().resolve(strict=True)
    override = os.environ.get("OPENPI_DATASET_LOCAL_ROOT", "").strip() if actual_root is None else str(actual_root)
    actual = Path(override).expanduser().resolve(strict=True) if override else root
    if actual != root:
        raise ValueError(f"实际数据根与声明不符：declared={root}, actual={actual}")
    if not root.is_dir():
        raise ValueError(f"数据根不是目录：{root}")
    return root


def _input_files(root: Path) -> list[dict]:
    paths = sorted(set((root / "data").rglob("*.parquet")) | set((root / "meta").rglob("*")))
    result = []
    for path in paths:
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.name in _EXCLUDED_INPUT_NAMES
            or any(part.startswith(".") or part in _EXCLUDED_INPUT_DIRS for part in relative.parts)
        ):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"源文件越过数据根：{path}")
        result.append({"path": relative.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not any(item["path"].startswith("data/") for item in result):
        raise ValueError(f"数据根没有 parquet：{root}")
    if not any(item["path"].startswith("meta/") for item in result):
        raise ValueError(f"数据根没有 meta：{root}")
    return result


def build_source_input_manifest(dataset_root: str | Path, *, source: dict | None = None) -> dict:
    root = validate_dataset_root(dataset_root)
    files = _input_files(root)
    return {
        "schema_version": 1,
        "kind": "ltmpi_source_input",
        "dataset_root": str(root),
        "source": source or {},
        "files": files,
        "source_input_pin": canonical_sha256(files),
    }


def validate_source_input_manifest(
    manifest_or_path: str | Path | Mapping, dataset_root: str | Path | None = None
) -> dict:
    manifest = _read_json(manifest_or_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "ltmpi_source_input":
        raise ValueError("不支持的输入清单格式")
    root = validate_dataset_root(dataset_root or manifest["dataset_root"])
    if root != Path(manifest["dataset_root"]).expanduser().resolve():
        raise ValueError("输入清单数据根不符")
    files = _input_files(root)
    if files != manifest.get("files") or canonical_sha256(files) != manifest.get("source_input_pin"):
        raise ValueError("源文件内容或集合与输入清单不符")
    return manifest


def _complete_manifest(path: str | Path, pin: str) -> dict:
    manifest = _read_json(path)
    if manifest.get("status") != "complete" or manifest.get("missing_episodes") not in (None, []):
        raise ValueError(f"产物未完整提交：{path}")
    if manifest.get("source_input_pin") != pin:
        raise ValueError(f"产物输入 pin 不匹配：{path}")
    return manifest


def _action_norm_path(config: Any) -> Path:
    assets = config.data.assets
    root = Path(assets.assets_dir or config.assets_dirs)
    return root / (assets.asset_id or config.data.repo_id) / "norm_stats.json"


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def build_training_contract(
    config: Any,
    source_manifest: Mapping,
    *,
    memory_manifest_path: str | Path,
    action_norm_path: str | Path,
    anchor_manifest_path: str | Path,
    main_commit: str,
    fork_commit: str,
) -> dict:
    pin = source_manifest["source_input_pin"]
    memory = _complete_manifest(memory_manifest_path, pin)
    _complete_manifest(anchor_manifest_path, pin)
    model = normalize(config.model)
    latent_norm = memory.get("latent_normalization")
    if not isinstance(latent_norm, dict):
        raise ValueError("memory manifest 缺少归一化")
    if model.get("memory_latent_mean") != latent_norm.get("mean") or model.get("memory_latent_std") != latent_norm.get(
        "std"
    ):
        raise ValueError("有效模型 LAM 归一化与 manifest 不一致")
    runtime = normalize(config)
    semantic = {
        "config_name": config.name,
        "model_config_sha256": canonical_sha256(config.model),
        "data_config_sha256": canonical_sha256(config.data),
        "source_input_pin": pin,
        "memory_manifest_sha256": sha256_file(memory_manifest_path),
        "anchor_manifest_sha256": sha256_file(anchor_manifest_path),
        "action_norm_sha256": canonical_sha256(_read_json(action_norm_path)),
        "latent_normalization_sha256": canonical_sha256(latent_norm),
        "lr_schedule": normalize(config.lr_schedule),
        "training": {key: value for key, value in runtime.items() if key not in _OPERATIONAL_FIELDS},
    }
    contract = {
        "schema_version": 1,
        "kind": "ltmpi_training_contract",
        "semantic": semantic,
        "semantic_sha256": canonical_sha256(semantic),
        "runtime_config": runtime,
        "provenance": {"main_commit": main_commit, "fork_commit": fork_commit},
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_training_contract(contract_or_path: str | Path | Mapping) -> dict:
    contract = _read_json(contract_or_path)
    if contract.get("schema_version") != 1 or contract.get("kind") != "ltmpi_training_contract":
        raise ValueError("不支持的训练合同格式")
    unhashed = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if canonical_sha256(unhashed) != contract.get("contract_sha256"):
        raise ValueError("训练合同内容 hash 不符")
    if canonical_sha256(contract["semantic"]) != contract.get("semantic_sha256"):
        raise ValueError("训练合同语义 hash 不符")
    return contract


def prepare_training_contract(config: Any) -> dict | None:
    path = getattr(config, "training_contract_path", None)
    receipts = getattr(config, "checkpoint_upload_receipts_dir", None)
    if not path:
        if config.name == S4_CONFIG_NAME or receipts:
            raise ValueError("新配置或上传等待必须设置 training_contract_path")
        return None
    if config.overwrite:
        raise ValueError("启用训练合同时禁止覆盖既有训练目录")
    root = validate_dataset_root(config.data.repo_id)
    source_path = os.environ.get("ROBOMME_INPUT_MANIFEST", str(root / "input_manifest.json"))
    source = validate_source_input_manifest(source_path, root)
    data = config.data.base_config
    memory_path = Path(data.memory_cache_dir) / "manifest.json"
    anchor_path = Path(data.demo_anchor_cache_dir) / "manifest.json"
    norm_path = _action_norm_path(config)
    norm_binding = norm_path.parent / "source_input_pin.json"
    if not norm_binding.is_file() or _read_json(norm_binding).get("source_input_pin") != source["source_input_pin"]:
        raise ValueError(f"动作归一化缺少匹配的输入 pin：{norm_binding}")
    if _read_json(norm_binding).get("norm_stats_sha256") != sha256_file(norm_path):
        raise ValueError("动作归一化内容与绑定不符")
    fork_root = Path(__file__).resolve().parents[3]
    main_root = Path(os.environ.get("LTMPI_MAIN_REPO", "/scratch/hongze/robomme-policy-learning-ltm"))
    contract = build_training_contract(
        config,
        source,
        memory_manifest_path=memory_path,
        action_norm_path=norm_path,
        anchor_manifest_path=anchor_path,
        main_commit=_git_commit(main_root),
        fork_commit=_git_commit(fork_root),
    )
    if Path(path).exists():
        existing = validate_training_contract(path)
        if existing["semantic_sha256"] != contract["semantic_sha256"]:
            raise ValueError("运行合同与既有合同语义不符")
    write_json_atomic(path, contract)
    return contract


def checkpoint_is_committed(path: str | Path) -> bool:
    path = Path(path)
    metadata = path / "_CHECKPOINT_METADATA"
    if not metadata.is_file():
        return False
    try:
        return int(_read_json(metadata).get("commit_timestamp_nsecs", 0)) > 0 and all(
            (path / item).is_dir() for item in ("params", "train_state", "assets")
        )
    except (ValueError, TypeError):
        return False


def validate_checkpoint_contract(checkpoint_dir: str | Path, config: Any, *, manifest_path: str | Path) -> dict | None:
    checkpoint = Path(checkpoint_dir)
    path = checkpoint / "assets" / "training_contract.json"
    if not path.is_file():
        if config.name == S4_CONFIG_NAME:
            raise ValueError("新配置的 checkpoint 缺少训练合同")
        return None
    if not checkpoint_is_committed(checkpoint):
        raise ValueError("checkpoint 尚未完整提交")
    contract = validate_training_contract(path)
    semantic = contract["semantic"]
    for key, actual in (
        ("config_name", config.name),
        ("model_config_sha256", canonical_sha256(config.model)),
        ("data_config_sha256", canonical_sha256(config.data)),
        ("memory_manifest_sha256", sha256_file(manifest_path)),
    ):
        if semantic.get(key) != actual:
            raise ValueError(f"checkpoint 合同身份不符：{key}")
    manifest = _complete_manifest(manifest_path, semantic["source_input_pin"])
    if canonical_sha256(manifest["latent_normalization"]) != semantic["latent_normalization_sha256"]:
        raise ValueError("checkpoint LAM 归一化不符")
    norm_paths = tuple((checkpoint / "assets").rglob("norm_stats.json"))
    if len(norm_paths) != 1 or canonical_sha256(_read_json(norm_paths[0])) != semantic["action_norm_sha256"]:
        raise ValueError("checkpoint 动作归一化不符")
    return contract


def checkpoint_file_manifest(checkpoint_dir: str | Path) -> list[dict]:
    root = Path(checkpoint_dir)
    if not checkpoint_is_committed(root):
        raise ValueError(f"拒绝未完整提交的 checkpoint：{root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint 不允许符号链接：{path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def validate_upload_receipt(
    receipt_or_path: str | Path | Mapping, *, step: int, contract_sha256: str, files_sha256: str
) -> dict:
    receipt = _read_json(receipt_or_path)
    if receipt.get("status") != "PASS":
        raise RuntimeError(f"上传失败或回执状态无效：step={step}, receipt={receipt}")
    expected = {"schema_version": 1, "step": step, "contract_sha256": contract_sha256, "files_sha256": files_sha256}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"上传回执不匹配：step={step}, key={key}")
    if not isinstance(receipt.get("remote_revision"), str) or not receipt["remote_revision"].strip():
        raise RuntimeError("上传回执缺少远端 revision")
    return receipt


def check_uploader_status(receipts_dir: str | Path) -> None:
    path = Path(receipts_dir) / "uploader.json"
    if not path.exists():
        return
    status = _read_json(path)
    if status.get("schema_version") != 1 or status.get("status") not in ("RUNNING", "PASS"):
        raise RuntimeError(f"上传进程失败或状态无效：{status}")


def wait_for_checkpoint_upload(
    checkpoint: str | Path, receipts_dir: str | Path, *, timeout_seconds: float = 7200, poll_seconds: float = 5
) -> dict:
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 7200:
        raise ValueError("上传等待上限必须在 (0, 7200] 秒内")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("上传轮询间隔必须为有限正数")
    check_uploader_status(receipts_dir)
    checkpoint = Path(checkpoint)
    step = int(checkpoint.name)
    contract = validate_training_contract(checkpoint / "assets" / "training_contract.json")
    files_hash = canonical_sha256(checkpoint_file_manifest(checkpoint))
    receipt_path = Path(receipts_dir) / f"{step}.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        check_uploader_status(receipts_dir)
        if receipt_path.exists():
            return validate_upload_receipt(
                receipt_path, step=step, contract_sha256=contract["contract_sha256"], files_sha256=files_hash
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待 checkpoint 上传超时：step={step}, seconds={timeout_seconds}")
        logging.info("等待上一保存点上传读回：step=%s", step)
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def check_checkpoint_capacity(
    checkpoint_dir: str | Path, *, required_bytes: int, reserve_bytes: int = 10 * 1024**3
) -> None:
    if required_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("保存空间预算必须有效")
    free = shutil.disk_usage(checkpoint_dir).free
    if free < required_bytes + reserve_bytes:
        raise RuntimeError(
            f"保存空间不足：free_bytes={free}, required_bytes={required_bytes}, "
            f"reserve_bytes={reserve_bytes}, deficit_bytes={required_bytes + reserve_bytes - free}"
        )


def before_checkpoint_save(
    checkpoint_dir: str | Path,
    receipts_dir: str | Path | None,
    *,
    required_bytes: int,
    reserve_bytes: int = 10 * 1024**3,
    timeout_seconds: float = 7200,
    poll_seconds: float = 5,
) -> None:
    """等待上一保存点交付且旧件清理完，再检查本次保存空间；不删除任何文件。"""
    if not receipts_dir:
        return
    if (
        required_bytes <= 0
        or reserve_bytes < 0
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 7200
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0
    ):
        raise ValueError("保存预算与等待参数必须有效，上传等待不超过 7200 秒")
    check_uploader_status(receipts_dir)
    root = Path(checkpoint_dir)
    started = time.monotonic()
    steps = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdecimal()), key=lambda path: int(path.name)
    )
    if any(not checkpoint_is_committed(path) for path in steps):
        raise RuntimeError("保存前存在未完成的数字 checkpoint 目录")
    if steps:
        wait_for_checkpoint_upload(steps[-1], receipts_dir, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)
    while len([path for path in root.iterdir() if path.is_dir() and path.name.isdecimal()]) > 1:
        check_uploader_status(receipts_dir)
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("上传已完成，但旧 checkpoint 尚未清理，停止新增保存")
        time.sleep(min(poll_seconds, max(0, timeout_seconds - (time.monotonic() - started))))
    check_checkpoint_capacity(root, required_bytes=required_bytes, reserve_bytes=reserve_bytes)
