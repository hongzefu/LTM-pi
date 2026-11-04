"""源内容身份与合法冒烟子库的 CPU 正反例。"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from openpi.shared import ltmpi_contract as contract
from scripts import build_anchor_cache as anchors
from scripts import combine_dreamdojo_memory_caches as combine_lib
from scripts import precompute_dreamdojo_memory as cache
from scripts import prepare_ltmpi_smoke_dataset as smoke


class SourceIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": 3,
            "total_frames": 420,
            "total_tasks": 1,
            "total_chunks": 1,
            "total_videos": 0,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "splits": {"train": "0:3"},
            "features": {
                "image": {"dtype": "image", "shape": [2, 2, 3]},
                "wrist_image": {"dtype": "image", "shape": [2, 2, 3]},
                "actions": {"dtype": "float32", "shape": [2]},
            },
        }
        contract.write_json_atomic(self.source / "meta/info.json", info)
        smoke._write_jsonl(
            self.source / "meta/episodes.jsonl",
            [{"episode_index": index, "length": 140, "tasks": ["拿起红色方块"]} for index in range(3)],
        )
        smoke._write_jsonl(self.source / "meta/tasks.jsonl", [{"task_index": 0, "task": "拿起红色方块"}])
        image = io.BytesIO()
        Image.fromarray(np.arange(12, dtype=np.uint8).reshape(2, 2, 3)).save(image, format="PNG")
        table = pa.table(
            {
                "episode_index": pa.array([2] * 140, type=pa.int64()),
                "index": pa.array(np.arange(280, 420), type=pa.int64()),
                "frame_index": pa.array(np.arange(140), type=pa.int64()),
                "timestamp": pa.array(np.arange(140) / 10, type=pa.float32()),
                "task_index": pa.array([0] * 140, type=pa.int64()),
                "exec_start_idx": pa.array([8] * 140, type=pa.int64()),
                "is_demo": pa.array(np.arange(140) < 8, type=pa.bool_()),
                "actions": pa.array([[float(i), -float(i)] for i in range(140)], type=pa.list_(pa.float32(), 2)),
                "image": [{"bytes": image.getvalue(), "path": None}] * 140,
                "wrist_image": [{"bytes": image.getvalue(), "path": None}] * 140,
            }
        )
        self.source_parquet = smoke._relative_parquet(self.source, info, 2)
        self.source_parquet.parent.mkdir(parents=True)
        pq.write_table(table, self.source_parquet)
        for path in [*sorted((self.source / "meta").iterdir()), self.source_parquet]:
            relative = path.relative_to(self.source).as_posix()
            metadata = self.source / ".cache/huggingface/download" / (relative + ".metadata")
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(smoke.OFFICIAL_REVISION + "\n" + contract.sha256_file(path) + "\n0\n")
        self.dataset = self.root / "smoke"

    def _prepare(self) -> dict:
        return smoke.prepare_smoke_dataset(self.source, self.dataset, episode_indices=[2])

    def _cache(self, image_key: str, *, pinned: bool = True) -> Path:
        dataset = cache.discover_dataset(
            self.dataset, input_manifest=self.dataset / "input_manifest.json" if pinned else None
        )
        output = self.root / image_key
        source = dataset.episodes[0]
        episode = cache.read_episode(source, dataset.root, image_key)
        latent = np.random.default_rng(2).normal(size=(140, cache.LATENT_DIM)).astype(np.float32)
        checkpoint = cache.CheckpointIdentity(self.root / "checkpoint", "a" * 64, 1, cache.CHECKPOINT_FORMAT, 400000)
        arrays = cache.build_cache_arrays(
            source,
            episode,
            latent,
            dataset=dataset,
            checkpoint=checkpoint,
            image_key=image_key,
            preprocess_version=cache.PREPROCESS_VERSION,
            storage_dtype="float32",
            transition_frame_stride=4,
        )
        cache.atomic_write_npz(cache.cache_path_for(output, source), arrays)
        return output

    def _command(self, command: str, output: Path, *, pinned: bool = True) -> argparse.Namespace:
        arguments = [command, "--dataset-root", str(self.dataset), "--output-root", str(output)]
        if pinned:
            arguments += ["--input-manifest", str(self.dataset / "input_manifest.json")]
        return cache.build_parser().parse_args(arguments)

    def test_selected_download_becomes_complete_independent_dataset(self) -> None:
        before = {path: contract.sha256_file(path) for path in self.source.rglob("*") if path.is_file()}
        manifest = self._prepare()
        self.assertEqual(
            before, {path: contract.sha256_file(path) for path in self.source.rglob("*") if path.is_file()}
        )
        self.assertFalse((self.source / "_READY").exists())
        self.assertTrue((self.dataset / "_READY").is_file())
        self.assertEqual(manifest["source"]["execution_samples"], 132)
        self.assertEqual(manifest["source"]["revision"], smoke.OFFICIAL_REVISION)
        spec = cache.discover_dataset(self.dataset, input_manifest=self.dataset / "input_manifest.json")
        self.assertEqual(len(spec.episodes), 1)
        source = pq.read_table(self.source_parquet)
        result = pq.read_table(spec.episodes[0].path)
        for key in set(source.column_names) - {"index", "episode_index"}:
            self.assertTrue(source[key].equals(result[key]), key)
        self.assertEqual(result["index"].to_pylist(), list(range(140)))
        self.assertEqual(result["episode_index"].to_pylist(), [0] * 140)
        self.assertEqual(smoke._jsonl(self.dataset / "meta/episodes_stats.jsonl")[0]["stats"]["index"]["min"], [0.0])
        self.assertEqual(json.loads((self.dataset / "meta/info.json").read_text())["splits"], {"train": "0:1"})
        with self.assertRaises(FileExistsError):
            self._prepare()

    def test_source_content_checked_before_output(self) -> None:
        self.source_parquet.write_bytes(self.source_parquet.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "下载凭据"):
            self._prepare()
        self.assertFalse(self.dataset.exists())

    def test_insufficient_samples_and_missing_selected_episode_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "执行样本"):
            smoke.prepare_smoke_dataset(self.source, self.dataset, episode_indices=[2], min_execution_samples=133)
        self.assertFalse(self.dataset.exists())
        with self.assertRaises((ValueError, FileNotFoundError)):
            smoke.prepare_smoke_dataset(self.source, self.dataset, episode_indices=[0])
        self.assertFalse(self.dataset.exists())

    def test_task_change_and_wrong_root_fail_without_changing_metadata_fingerprint(self) -> None:
        self._prepare()
        before = cache.discover_dataset(self.dataset).fingerprint
        with mock.patch.dict(os.environ, {"OPENPI_DATASET_LOCAL_ROOT": str(self.source)}):
            with self.assertRaises(ValueError):
                cache.discover_dataset(self.dataset, input_manifest=self.dataset / "input_manifest.json")
        tasks = self.dataset / "meta/tasks.jsonl"
        tasks.write_text(tasks.read_text().replace("红色", "蓝色"))
        self.assertEqual(cache.discover_dataset(self.dataset).fingerprint, before)
        with self.assertRaises(ValueError):
            cache.discover_dataset(self.dataset, input_manifest=self.dataset / "input_manifest.json")

    def test_parquet_change_is_detected(self) -> None:
        self._prepare()
        spec = cache.discover_dataset(self.dataset)
        parquet = spec.episodes[0].path
        parquet.write_bytes(parquet.read_bytes() + b"x")
        self.assertEqual(cache.discover_dataset(self.dataset).fingerprint, spec.fingerprint)
        with self.assertRaises(ValueError):
            cache.discover_dataset(self.dataset, input_manifest=self.dataset / "input_manifest.json")

    def test_complete_cache_combine_anchor_bind_same_pin(self) -> None:
        manifest = self._prepare()
        external, wrist = self._cache("image"), self._cache("wrist_image")
        for root in (external, wrist):
            self.assertEqual(cache.finalize_command(self._command("finalize", root)), 0)
            self.assertEqual(cache.validate_command(self._command("validate", root)), 0)
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["source_input_pin"], manifest["source_input_pin"]
            )
        combine_lib.combine(
            external,
            wrist,
            self.root / "combined",
            input_manifest=self.dataset / "input_manifest.json",
            dataset_root=self.dataset,
        )
        self.assertEqual(
            json.loads((self.root / "combined/manifest.json").read_text())["source_input_pin"],
            manifest["source_input_pin"],
        )
        anchors.main(self.dataset, self.root / "anchors", self.dataset / "input_manifest.json")
        self.assertEqual(
            json.loads((self.root / "anchors/manifest.json").read_text())["source_input_pin"],
            manifest["source_input_pin"],
        )
        with self.assertRaisesRegex(ValueError, "source_input_pin"):
            combine_lib.combine(external, wrist, self.root / "unbound")
        with self.assertRaisesRegex(ValueError, "source_input_pin"):
            cache.validate_command(self._command("validate", external, pinned=False))

    def test_old_cache_cannot_be_signed_and_legacy_mode_stays_usable(self) -> None:
        self._prepare()
        output = self._cache("image", pinned=False)
        with self.assertRaisesRegex(ValueError, "source_input_pin"):
            cache.finalize_command(self._command("finalize", output))
        self.assertFalse((output / "manifest.json").exists())
        self.assertEqual(cache.finalize_command(self._command("finalize", output, pinned=False)), 0)
        self.assertEqual(cache.validate_command(self._command("validate", output, pinned=False)), 0)
        self.assertNotIn("source_input_pin", json.loads((output / "manifest.json").read_text()))

    def test_repair_recomputes_unknown_cache_instead_of_reusing_it(self) -> None:
        self._prepare()
        output = self._cache("image", pinned=False)
        checkpoint = self.root / "checkpoint"
        checkpoint.write_bytes(b"fixture")
        args = cache.build_parser().parse_args(
            [
                "extract",
                "--dataset-root",
                str(self.dataset),
                "--output-root",
                str(output),
                "--input-manifest",
                str(self.dataset / "input_manifest.json"),
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                "a" * 64,
                "--transition-frame-stride",
                "4",
                "--repair",
                "--dry-run",
            ]
        )
        buffer = io.StringIO()
        with mock.patch("sys.stdout", buffer):
            self.assertEqual(cache.extract_command(args), 0)
        self.assertIn("pending=1 skipped=0", buffer.getvalue())
        path = cache.cache_path_for(output, cache.discover_dataset(self.dataset).episodes[0])
        self.assertIsNone(cache.read_cache_expectation(path).source_input_pin)

    def test_partial_sources_cannot_be_combined(self) -> None:
        self._prepare()
        external, wrist = self._cache("image"), self._cache("wrist_image")
        for root in (external, wrist):
            cache.finalize_command(self._command("finalize", root))
        path = wrist / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["status"] = "partial"
        contract.write_json_atomic(path, manifest)
        with self.assertRaisesRegex(ValueError, "not complete"):
            combine_lib.combine(
                external,
                wrist,
                self.root / "partial",
                input_manifest=self.dataset / "input_manifest.json",
                dataset_root=self.dataset,
            )
        self.assertFalse((self.root / "partial").exists())

    def test_changed_input_prevents_anchor_commit(self) -> None:
        self._prepare()
        original = anchors._decode_image
        changed = False

        def mutate(value: dict, root: Path) -> np.ndarray:
            nonlocal changed
            if not changed:
                tasks = root / "meta/tasks.jsonl"
                tasks.write_text(tasks.read_text() + "\n")
                changed = True
            return original(value, root)

        with mock.patch.object(anchors, "_decode_image", mutate):
            with self.assertRaises(ValueError):
                anchors.main(self.dataset, self.root / "changed_anchors", self.dataset / "input_manifest.json")
        self.assertFalse((self.root / "changed_anchors").exists())

    def test_prepare_norm_passes_new_config_and_writes_content_binding(self) -> None:
        self._prepare()
        wrapper = self.root / "python-wrapper"
        wrapper.write_text(
            "#!"
            + sys.executable
            + "\n"
            + """
import json
from pathlib import Path
import sys
import openpi.shared
openpi.shared.__path__.insert(0, CONTRACT_DIRECTORY)
if sys.argv[1] == '-':
    sys.argv = sys.argv[1:]
    exec(sys.stdin.read())
elif sys.argv[1].endswith('compute_norm_stats.py'):
    directory = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'norm_stats.json').write_text('{}')
    (directory / 'command.json').write_text(json.dumps(sys.argv))
else:
    raise RuntimeError(sys.argv)
""".replace("CONTRACT_DIRECTORY", repr(str(Path(contract.__file__).parent)))
        )
        wrapper.chmod(0o700)
        env_file = self.root / "empty.env"
        env_file.write_text("")
        environment = {
            **os.environ,
            "ROBOMME_ENV_FILE": str(env_file),
            "PYTHON_BIN": str(wrapper),
            "ROBOMME_VIDEO_DATASET": str(self.dataset),
            "ROBOMME_INPUT_MANIFEST": str(self.dataset / "input_manifest.json"),
            "ROBOMME_CONFIG": contract.S4_CONFIG_NAME,
            "TRANSITION_FRAME_STRIDE": "4",
            "ROBOMME_ASSETS_ROOT": str(self.root / "assets"),
            "ROBOMME_ASSET_ID": "smoke",
            "OPENPI_DATASET_LOCAL_ROOT": str(self.dataset),
        }
        script = Path(__file__).resolve().parents[1] / "bin/prepare.sh"
        completed = subprocess.run(["bash", str(script), "norm-stats"], env=environment, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = self.root / "assets/smoke"
        binding = json.loads((output / "source_input_pin.json").read_text())
        self.assertEqual(
            binding["source_input_pin"],
            json.loads((self.dataset / "input_manifest.json").read_text())["source_input_pin"],
        )
        self.assertEqual(binding["norm_stats_sha256"], contract.sha256_file(output / "norm_stats.json"))
        self.assertIn(contract.S4_CONFIG_NAME, json.loads((output / "command.json").read_text()))
        environment["OPENPI_DATASET_LOCAL_ROOT"] = str(self.source)
        completed = subprocess.run(["bash", str(script), "norm-stats"], env=environment, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)

    def test_wrong_download_revision_is_rejected(self) -> None:
        relative = self.source_parquet.relative_to(self.source).as_posix()
        metadata = self.source / ".cache/huggingface/download" / (relative + ".metadata")
        metadata.write_text(metadata.read_text().replace(smoke.OFFICIAL_REVISION, "0" * 40))
        with self.assertRaisesRegex(ValueError, "revision"):
            self._prepare()
        self.assertFalse(self.dataset.exists())

    def test_smoke_metadata_is_consumable_by_lerobot(self) -> None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self._prepare()
        metadata = LeRobotDatasetMetadata("fixture", root=self.dataset, revision="v2.1")
        self.assertEqual(metadata.total_episodes, 1)
        self.assertEqual(metadata.total_frames, 140)
        self.assertEqual(list(metadata.episodes), [0])
        self.assertEqual(list(metadata.episodes_stats), [0])
        np.testing.assert_array_equal(metadata.stats["image"]["count"], [140])
        np.testing.assert_array_equal(metadata.stats["index"]["max"], [139])


if __name__ == "__main__":
    unittest.main()
