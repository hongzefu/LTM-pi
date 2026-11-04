#!/usr/bin/env python3
"""从固定官方 revision 的显式 episode 集合建立独立、完整的冒烟库。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from openpi.shared import ltmpi_contract as contract

OFFICIAL_REPO = "Yinpei/robomme_data_lerobot"
OFFICIAL_REVISION = "1510653cccb4d9e5165fb3141c06d88053decc20"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _source_record(root: Path, path: Path) -> dict:
    """核对 hf download 的固定 revision 与内容 etag，并留下官方源文件摘要。"""
    relative = path.relative_to(root).as_posix()
    metadata = root / ".cache" / "huggingface" / "download" / (relative + ".metadata")
    if not metadata.is_file():
        raise ValueError(f"缺少固定 revision 下载凭据：{metadata}")
    lines = metadata.read_text().splitlines()
    if len(lines) < 2 or lines[0] != OFFICIAL_REVISION:
        raise ValueError(f"官方 revision 不匹配：{relative}")
    sha256 = contract.sha256_file(path)
    etag = lines[1].strip('"')
    if len(etag) == 64:
        actual_etag = sha256
    elif len(etag) == 40:
        payload = path.read_bytes()
        actual_etag = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    else:
        raise ValueError(f"无法验证官方源文件 etag：{relative}")
    if actual_etag != etag:
        raise ValueError(f"官方源文件内容与下载凭据不一致：{relative}")
    return {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256}


def _relative_parquet(root: Path, info: dict, episode: int) -> Path:
    relative = info["data_path"].format(episode_index=episode, episode_chunk=episode // int(info["chunks_size"]))
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix != ".parquet":
        raise ValueError(f"非法 episode 路径：{relative}")
    return path


def _scalar(table: pa.Table, key: str) -> np.ndarray:
    values = np.asarray(table[key].to_pylist())
    if values.ndim != 1:
        raise ValueError(f"{key} 必须为标量列")
    return values


def _validate_episode(table: pa.Table, *, episode: int, length: int, features: dict) -> int:
    required = {
        "episode_index",
        "index",
        "frame_index",
        "timestamp",
        "task_index",
        "exec_start_idx",
        "is_demo",
        "actions",
    }
    if required - set(table.column_names):
        raise ValueError(f"episode {episode} 缺少列：{sorted(required - set(table.column_names))}")
    if table.num_rows != length or length <= 0:
        raise ValueError(f"episode {episode} 帧数与 metadata 不匹配")
    frame = _scalar(table, "frame_index")
    if not np.array_equal(frame, np.arange(length)):
        raise ValueError(f"episode {episode} frame_index 必须连续且从零开始")
    if not np.all(_scalar(table, "episode_index") == episode):
        raise ValueError(f"episode {episode} 的 episode_index 不一致")
    indices = _scalar(table, "index")
    if np.any(np.diff(indices) != 1):
        raise ValueError(f"episode {episode} 全局 index 不连续")
    timestamp = _scalar(table, "timestamp")
    if not np.all(np.isfinite(timestamp)) or np.any(np.diff(timestamp) < 0):
        raise ValueError(f"episode {episode} timestamp 非法")
    start = _scalar(table, "exec_start_idx")
    if np.any(start != start[0]) or not 0 <= start[0] < length:
        raise ValueError(f"episode {episode} exec_start_idx 非法")
    is_demo = _scalar(table, "is_demo")
    if is_demo.dtype != np.bool_ or not np.array_equal(is_demo, frame < start[0]):
        raise ValueError(f"episode {episode} is_demo 与执行边界不一致")
    for key in ("image", "wrist_image"):
        if features.get(key, {}).get("dtype") != "image" or key not in table.column_names:
            raise ValueError(f"冒烟输入必须使用 parquet 内嵌双视角图像：{key}")
        for row in table[key].to_pylist():
            if not isinstance(row, dict) or not row.get("bytes"):
                raise ValueError(f"{key} 不是内嵌图像，禁止引用源根外部媒体")
            with Image.open(io.BytesIO(row["bytes"])) as image:
                image.load()
                if list(np.asarray(image.convert("RGB")).shape) != features[key]["shape"]:
                    raise ValueError(f"{key} 解码形状与 metadata 不一致")
    action = np.asarray(table["actions"].to_pylist())
    if not np.all(np.isfinite(action)):
        raise ValueError(f"episode {episode} 动作包含非有限值")
    return int(np.count_nonzero(~is_demo))


def _episode_stats(table: pa.Table, features: dict) -> dict:
    """使用子库的实际全部帧重建 v2 统计，不沿用全库统计或旧全局索引。"""
    stats = {}
    for key in table.column_names:
        if features.get(key, {}).get("dtype") == "image":
            sums = np.zeros(3, dtype=np.float64)
            squares = np.zeros(3, dtype=np.float64)
            minima = np.full(3, np.inf)
            maxima = np.full(3, -np.inf)
            pixels = 0
            for row in table[key].to_pylist():
                with Image.open(io.BytesIO(row["bytes"])) as image:
                    values = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
                sums += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                minima = np.minimum(minima, values.min(axis=0))
                maxima = np.maximum(maxima, values.max(axis=0))
                pixels += len(values)
            mean = sums / pixels
            stats[key] = {
                "min": minima.reshape(3, 1, 1).tolist(),
                "max": maxima.reshape(3, 1, 1).tolist(),
                "mean": mean.reshape(3, 1, 1).tolist(),
                "std": np.sqrt(np.maximum(squares / pixels - mean**2, 0)).reshape(3, 1, 1).tolist(),
                "count": [table.num_rows],
            }
            continue
        else:
            array = np.asarray(table[key].to_pylist())
            if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
                continue
            array = array.astype(np.float64)
            if array.ndim == 1:
                array = array[:, None]
            axes = 0
            shape = array.shape[1:]
        stats[key] = {
            name: getattr(array, name)(axis=axes).reshape(shape).tolist() for name in ("min", "max", "mean", "std")
        }
        stats[key]["count"] = [table.num_rows]
    return stats


def _aggregate_stats(episodes_stats: list[dict]) -> dict:
    result = {}
    for key in episodes_stats[0]["stats"]:
        parts = [row["stats"][key] for row in episodes_stats]
        counts = np.asarray([part["count"][0] for part in parts], dtype=np.float64)
        means = np.asarray([part["mean"] for part in parts])
        stds = np.asarray([part["std"] for part in parts])
        weights = (counts / counts.sum()).reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weights).sum(axis=0)
        variance = ((stds**2 + (means - mean) ** 2) * weights).sum(axis=0)
        result[key] = {
            "min": np.min([part["min"] for part in parts], axis=0).tolist(),
            "max": np.max([part["max"] for part in parts], axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(counts.sum())],
        }
    return result


def prepare_smoke_dataset(
    source_root: Path,
    output_root: Path,
    *,
    episode_indices: list[int],
    min_execution_samples: int = 128,
) -> dict:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"输出已存在，禁止覆盖：{output_root}")
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("源根与子库输出根必须独立且不得互相嵌套")
    if min_execution_samples < 128:
        raise ValueError("冒烟至少需要 128 个执行样本")
    if not episode_indices or len(set(episode_indices)) != len(episode_indices) or min(episode_indices) < 0:
        raise ValueError("必须显式选择非空、无重复的 episode 集合")
    episode_indices = sorted(episode_indices)
    meta_paths = [source_root / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl")]
    source_files = [_source_record(source_root, path) for path in meta_paths]
    info = json.loads(meta_paths[0].read_text())
    if not str(info.get("codebase_version", "")).startswith("v2"):
        raise ValueError("只支持官方 LeRobot v2 源库")
    episode_rows = _jsonl(meta_paths[1])
    by_episode = {int(row["episode_index"]): row for row in episode_rows}
    if len(by_episode) != len(episode_rows) or len(episode_rows) != info["total_episodes"]:
        raise ValueError("源库 episode metadata 不完整或包含重复")
    if sum(int(row["length"]) for row in episode_rows) != info["total_frames"]:
        raise ValueError("源库帧数 metadata 不一致")
    tasks = _jsonl(meta_paths[2])
    if [row["task_index"] for row in tasks] != list(range(len(tasks))) or len(tasks) != info["total_tasks"]:
        raise ValueError("源库任务索引或任务总数不一致")
    tables = []
    execution_samples = 0
    mapping = []
    offset = 0
    for new_episode, old_episode in enumerate(episode_indices):
        if old_episode not in by_episode:
            raise ValueError(f"所选 episode 不在源 metadata 中：{old_episode}")
        path = _relative_parquet(source_root, info, old_episode)
        source_files.append(_source_record(source_root, path))
        table = pq.read_table(path)
        execution_samples += _validate_episode(
            table, episode=old_episode, length=by_episode[old_episode]["length"], features=info["features"]
        )
        task_ids = _scalar(table, "task_index")
        if np.any(task_ids < 0) or np.any(task_ids >= len(tasks)):
            raise ValueError(f"episode {old_episode} 引用未知任务")
        mapping.append(
            {
                "source_episode_index": old_episode,
                "episode_index": new_episode,
                "source_first_index": int(_scalar(table, "index")[0]),
                "first_index": offset,
            }
        )
        for key, values in (
            ("episode_index", np.full(table.num_rows, new_episode)),
            ("index", np.arange(offset, offset + table.num_rows)),
        ):
            field = table.schema.field(key)
            table = table.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
        tables.append(table)
        offset += table.num_rows
    if execution_samples < min_execution_samples:
        raise ValueError(f"所选集合只有 {execution_samples} 个执行样本，需要至少 {min_execution_samples}")

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "meta").mkdir()
    output_info = copy.deepcopy(info)
    output_info.update(
        total_episodes=len(tables),
        total_frames=offset,
        total_chunks=(len(tables) - 1) // int(info["chunks_size"]) + 1,
        total_videos=0,
        splits={"train": f"0:{len(tables)}"},
    )
    output_info.pop("video_path", None)
    output_episodes = []
    output_stats = []
    for new_episode, table in enumerate(tables):
        path = _relative_parquet(output_root, output_info, new_episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        stored = pq.read_table(path)
        if not table.equals(stored):
            raise ValueError(f"子库 parquet 写入后数值发生变化：episode {new_episode}")
        _validate_episode(stored, episode=new_episode, length=table.num_rows, features=info["features"])
        row = copy.deepcopy(by_episode[episode_indices[new_episode]])
        row.update(episode_index=new_episode, length=table.num_rows)
        output_episodes.append(row)
        output_stats.append({"episode_index": new_episode, "stats": _episode_stats(stored, info["features"])})
    contract.write_json_atomic(output_root / "meta" / "info.json", output_info)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_stats)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", tasks)
    contract.write_json_atomic(output_root / "meta" / "stats.json", _aggregate_stats(output_stats))
    for record in source_files:
        if _source_record(source_root, source_root / record["path"]) != record:
            raise ValueError("子库构建期间官方源文件发生变化")
    manifest = contract.build_source_input_manifest(
        output_root,
        source={
            "repo_id": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION,
            "source_root": str(source_root),
            "selected_source_files": sorted(source_files, key=lambda row: row["path"]),
            "episode_mapping": mapping,
            "execution_samples": execution_samples,
        },
    )
    contract.write_json_atomic(output_root / "input_manifest.json", manifest)
    contract.validate_source_input_manifest(output_root / "input_manifest.json", dataset_root=output_root)
    (output_root / "_READY").write_text(manifest["source_input_pin"] + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-execution-samples", type=int, default=128)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode-indices", help="逗号分隔的官方 episode 编号。")
    selection.add_argument("--episode-start", type=int, help="所选连续区间起点，含该 episode。")
    parser.add_argument("--episode-stop", type=int, help="所选连续区间终点，不含该 episode。")
    args = parser.parse_args()
    if args.episode_indices is not None:
        if args.episode_stop is not None:
            parser.error("--episode-indices 不得同时使用 --episode-stop")
        selected = [int(value) for value in args.episode_indices.split(",")]
    else:
        if args.episode_stop is None or args.episode_stop <= args.episode_start:
            parser.error("--episode-start 需要更大的 --episode-stop")
        selected = list(range(args.episode_start, args.episode_stop))
    manifest = prepare_smoke_dataset(
        args.source_root, args.output_root, episode_indices=selected, min_execution_samples=args.min_execution_samples
    )
    print(f"LTMPI_SMOKE_DATASET=PASS source_input_pin={manifest['source_input_pin']}")


if __name__ == "__main__":
    main()
