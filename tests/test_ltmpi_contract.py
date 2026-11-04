"""验证输入内容、训练恢复身份以及上传等待的正负例。"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpi.shared import ltmpi_contract as contract


@dataclasses.dataclass(frozen=True)
class Model:
    memory_latent_mean: tuple = (0.0, 1.0)
    memory_latent_std: tuple = (1.0, 2.0)
    memory_anchor_pool_grid: int = 4


@dataclasses.dataclass(frozen=True)
class Config:
    name: str = contract.S4_CONFIG_NAME
    model: Model = Model()
    data: tuple = ("declared-data", "stride4")
    lr_schedule: tuple = (1000, 2.5e-5, 60000, 2.5e-6)
    num_train_steps: int = 20
    save_steps: tuple = (20,)
    log_interval: int = 1
    resume: bool = False
    overwrite: bool = False
    training_contract_path: str | None = None
    checkpoint_upload_receipts_dir: str | None = None
    seed: int = 0


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ltmpi-contract-", dir="/scratch/hongze")
        self.root = Path(self.temporary.name)
        self.environment = mock.patch.dict(os.environ, {"OPENPI_DATASET_LOCAL_ROOT": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.temporary.cleanup)
        self.dataset = self.root / "dataset"
        (self.dataset / "data" / "chunk-000").mkdir(parents=True)
        (self.dataset / "meta").mkdir()
        self.parquet = self.dataset / "data/chunk-000/episode_000000.parquet"
        self.parquet.write_bytes(b"parquet-content")
        (self.dataset / "meta/info.json").write_text('{"total_episodes": 1}')
        (self.dataset / "meta/tasks.jsonl").write_text('{"task": "pick"}\n')
        self.source = contract.build_source_input_manifest(self.dataset, source={"revision": "fixed"})
        self.memory = self.root / "memory.json"
        self.anchor = self.root / "anchor.json"
        self.norm = self.root / "norm_stats.json"
        self.memory_value = {
            "status": "complete",
            "source_input_pin": self.source["source_input_pin"],
            "latent_normalization": {"mean": [0.0, 1.0], "std": [1.0, 2.0]},
        }
        contract.write_json_atomic(self.memory, self.memory_value)
        contract.write_json_atomic(
            self.anchor, {"status": "complete", "source_input_pin": self.source["source_input_pin"]}
        )
        contract.write_json_atomic(self.norm, {"norm_stats": {"actions": {"mean": [0], "std": [1]}}})
        self.config = Config()
        self.document = self.build_contract(self.config)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir()
        self.receipts = self.root / "receipts"
        self.receipts.mkdir()

    def build_contract(self, config):
        return contract.build_training_contract(
            config,
            self.source,
            memory_manifest_path=self.memory,
            action_norm_path=self.norm,
            anchor_manifest_path=self.anchor,
            main_commit="main-fixed",
            fork_commit="fork-fixed",
        )

    def checkpoint(self, step=20):
        root = self.checkpoints / str(step)
        for item in ("params", "train_state", "assets"):
            (root / item).mkdir(parents=True)
        (root / "params/array").write_bytes(b"parameters")
        (root / "train_state/array").write_bytes(b"optimizer-and-step")
        contract.write_json_atomic(root / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 10})
        contract.write_json_atomic(root / "assets/training_contract.json", self.document)
        shutil.copyfile(self.norm, root / "assets/norm_stats.json")
        return root

    def receipt(self, checkpoint, *, status="PASS"):
        return {
            "schema_version": 1,
            "status": status,
            "step": int(checkpoint.name),
            "contract_sha256": self.document["contract_sha256"],
            "files_sha256": contract.canonical_sha256(contract.checkpoint_file_manifest(checkpoint)),
            "remote_revision": "bucket-manifest-sha256:fixed",
        }

    def test_source_pin_includes_tasks_and_content_not_just_size(self):
        self.assertEqual(contract.validate_source_input_manifest(self.source), self.source)
        self.parquet.write_bytes(b"PARQUET-content")
        with self.assertRaisesRegex(ValueError, "源文件"):
            contract.validate_source_input_manifest(self.source)
        self.parquet.write_bytes(b"parquet-content")
        (self.dataset / "meta/tasks.jsonl").write_text('{"task": "drop"}\n')
        with self.assertRaisesRegex(ValueError, "源文件"):
            contract.validate_source_input_manifest(self.source)

    def test_source_pin_excludes_self_ready_and_derived(self):
        contract.write_json_atomic(self.dataset / "input_manifest.json", self.source)
        contract.write_json_atomic(self.dataset / "meta/input_manifest.json", self.source)
        (self.dataset / "meta/READY").write_text("ready")
        (self.dataset / "meta/derived").mkdir()
        (self.dataset / "meta/derived/arrow").write_bytes(b"derived")
        self.assertEqual(contract.validate_source_input_manifest(self.source), self.source)

    def test_wrong_actual_root_rejected_even_with_same_files(self):
        other = self.root / "other"
        shutil.copytree(self.dataset, other)
        with mock.patch.dict(os.environ, {"OPENPI_DATASET_LOCAL_ROOT": str(other)}):
            with self.assertRaisesRegex(ValueError, "实际数据根"):
                contract.validate_source_input_manifest(self.source)
        with self.assertRaisesRegex(ValueError, "数据根不符"):
            contract.validate_source_input_manifest(self.source, other)

    def test_source_added_removed_and_external_symlink_rejected(self):
        extra = self.dataset / "meta/extra.json"
        extra.write_text("{}")
        with self.assertRaises(ValueError):
            contract.validate_source_input_manifest(self.source)
        extra.unlink()
        extra.symlink_to(self.norm)
        with self.assertRaisesRegex(ValueError, "越过数据根"):
            contract.validate_source_input_manifest(self.source)

    def test_resume20_to21_keeps_semantics_and_full_runtime(self):
        resumed = self.build_contract(
            dataclasses.replace(self.config, num_train_steps=21, save_steps=(21,), resume=True)
        )
        self.assertEqual(self.document["semantic_sha256"], resumed["semantic_sha256"])
        self.assertNotEqual(self.document["contract_sha256"], resumed["contract_sha256"])
        self.assertEqual(resumed["runtime_config"]["num_train_steps"], 21)
        for config in (
            dataclasses.replace(self.config, seed=1),
            dataclasses.replace(self.config, lr_schedule=(1000, 2.5e-5, 20, 2.5e-6)),
        ):
            self.assertNotEqual(self.document["semantic_sha256"], self.build_contract(config)["semantic_sha256"])

    def test_contract_tamper_and_same_shape_model_rejected(self):
        altered = json.loads(json.dumps(self.document))
        altered["runtime_config"]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "hash"):
            contract.validate_training_contract(altered)
        checkpoint = self.checkpoint()
        self.assertEqual(
            contract.validate_checkpoint_contract(checkpoint, self.config, manifest_path=self.memory), self.document
        )
        wrong = dataclasses.replace(
            self.config, model=dataclasses.replace(self.config.model, memory_anchor_pool_grid=16)
        )
        with self.assertRaisesRegex(ValueError, "model_config"):
            contract.validate_checkpoint_contract(checkpoint, wrong, manifest_path=self.memory)
        with self.assertRaisesRegex(ValueError, "config_name"):
            contract.validate_checkpoint_contract(
                checkpoint, dataclasses.replace(self.config, name="old"), manifest_path=self.memory
            )

    def test_missing_contract_old_compatible_new_rejected(self):
        checkpoint = self.checkpoint()
        (checkpoint / "assets/training_contract.json").unlink()
        self.assertIsNone(
            contract.validate_checkpoint_contract(
                checkpoint, dataclasses.replace(self.config, name="old"), manifest_path=self.memory
            )
        )
        with self.assertRaisesRegex(ValueError, "缺少训练合同"):
            contract.validate_checkpoint_contract(checkpoint, self.config, manifest_path=self.memory)
        self.assertIsNone(contract.prepare_training_contract(dataclasses.replace(self.config, name="old")))
        with self.assertRaisesRegex(ValueError, "training_contract_path"):
            contract.prepare_training_contract(self.config)

    def test_partial_manifest_and_uncommitted_checkpoint_rejected(self):
        contract.write_json_atomic(self.memory, {**self.memory_value, "status": "partial"})
        with self.assertRaisesRegex(ValueError, "未完整"):
            self.build_contract(self.config)
        checkpoint = self.checkpoint()
        contract.write_json_atomic(checkpoint / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": None})
        self.assertFalse(contract.checkpoint_is_committed(checkpoint))
        with self.assertRaisesRegex(ValueError, "未完整"):
            contract.checkpoint_file_manifest(checkpoint)

    def test_wrong_cache_pin_and_normalization_rejected(self):
        contract.write_json_atomic(self.memory, {**self.memory_value, "source_input_pin": "different"})
        with self.assertRaisesRegex(ValueError, "pin"):
            self.build_contract(self.config)
        contract.write_json_atomic(self.memory, self.memory_value)
        wrong = dataclasses.replace(
            self.config, model=dataclasses.replace(self.config.model, memory_latent_std=(9.0, 9.0))
        )
        with self.assertRaisesRegex(ValueError, "归一化"):
            self.build_contract(wrong)

    def test_delayed_receipt_waits_then_passes(self):
        checkpoint = self.checkpoint()
        receipt = self.receipt(checkpoint)
        with mock.patch.object(
            contract.time, "sleep", side_effect=lambda _: contract.write_json_atomic(self.receipts / "20.json", receipt)
        ) as sleep:
            contract.before_checkpoint_save(
                self.checkpoints, self.receipts, required_bytes=1, reserve_bytes=0, poll_seconds=0.01, timeout_seconds=1
            )
        sleep.assert_called_once()
        self.assertTrue(checkpoint.exists())

    def test_failure_receipt_preserves_checkpoint(self):
        checkpoint = self.checkpoint()
        contract.write_json_atomic(self.receipts / "20.json", self.receipt(checkpoint, status="FAIL"))
        with self.assertRaisesRegex(RuntimeError, "上传失败"):
            contract.before_checkpoint_save(self.checkpoints, self.receipts, required_bytes=1)
        self.assertTrue(checkpoint.exists())

    def test_uploader_startup_failure_stops_first_save(self):
        contract.write_json_atomic(self.receipts / "uploader.json", {"schema_version": 1, "status": "FAIL"})
        with self.assertRaisesRegex(RuntimeError, "上传进程失败"):
            contract.before_checkpoint_save(self.checkpoints, self.receipts, required_bytes=1)

    def test_receipt_wrong_identity_rejected(self):
        checkpoint = self.checkpoint()
        receipt = self.receipt(checkpoint)
        for key, value in (("step", 21), ("contract_sha256", "bad"), ("files_sha256", "bad"), ("remote_revision", "")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                contract.validate_upload_receipt(
                    {**receipt, key: value},
                    step=20,
                    contract_sha256=self.document["contract_sha256"],
                    files_sha256=receipt["files_sha256"],
                )

    def test_wait_limit_is_bounded(self):
        checkpoint = self.checkpoint()
        for seconds in (0, -1, float("inf"), float("nan"), 7201):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                contract.wait_for_checkpoint_upload(checkpoint, self.receipts, timeout_seconds=seconds)

    def test_timeout_and_capacity_failure_keep_latest(self):
        checkpoint = self.checkpoint()
        with mock.patch.object(contract.time, "monotonic", side_effect=[0, 0, 2]):
            with self.assertRaises(TimeoutError):
                contract.before_checkpoint_save(self.checkpoints, self.receipts, required_bytes=1, timeout_seconds=1)
        contract.write_json_atomic(self.receipts / "20.json", self.receipt(checkpoint))
        with mock.patch.object(contract.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(10, 8, 2)):
            with self.assertRaisesRegex(RuntimeError, "deficit_bytes=2"):
                contract.before_checkpoint_save(self.checkpoints, self.receipts, required_bytes=3, reserve_bytes=1)
        self.assertTrue(checkpoint.exists())

    def test_third_save_waits_for_old_cleanup(self):
        old = self.checkpoint(10)
        latest = self.checkpoint(20)
        contract.write_json_atomic(self.receipts / "20.json", self.receipt(latest))
        with mock.patch.object(contract.time, "sleep", side_effect=lambda _: shutil.rmtree(old)) as sleep:
            contract.before_checkpoint_save(self.checkpoints, self.receipts, required_bytes=1, reserve_bytes=0)
        sleep.assert_called_once()
        self.assertTrue(latest.exists())

    def test_disabled_receipts_dont_read_paths_or_wait(self):
        with mock.patch.object(contract.shutil, "disk_usage", side_effect=AssertionError("不能读取磁盘")):
            contract.before_checkpoint_save("/no-such-directory", None, required_bytes=0)
            contract.before_checkpoint_save("/no-such-directory", "", required_bytes=0)

    def test_orbax_complete_save_restores_optimizer_and_embedded_contract(self):
        from types import SimpleNamespace

        import jax.numpy as jnp
        import optax
        from flax import nnx

        from openpi.shared import normalize
        from openpi.training import checkpoints, utils

        model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
        graphdef, params = nnx.split(model)
        tx = optax.adam(1e-3)
        optimizer_state = tx.init(params)
        optimizer_state = (optimizer_state[0]._replace(count=jnp.array(20)), *optimizer_state[1:])
        state = utils.TrainState(
            step=jnp.array(20), params=params, model_def=graphdef, opt_state=optimizer_state, tx=tx, ema_decay=None
        )
        data = SimpleNamespace(
            asset_id="id", norm_stats={"actions": normalize.NormStats(mean=jnp.array([0.0]), std=jnp.array([1.0]))}
        )
        loader = SimpleNamespace(data_config=lambda: data)
        output = self.root / "real-orbax"
        manager, resuming = checkpoints.initialize_checkpoint_dir(
            output, keep_period=None, overwrite=False, resume=False, preserve_checkpoints=True
        )
        self.addCleanup(manager.close)
        self.assertFalse(resuming)
        checkpoints.save_state(manager, state, loader, 20, training_contract=self.document)
        manager.wait_until_finished()
        self.assertTrue(contract.checkpoint_is_committed(output / "20"))
        self.assertEqual(
            contract.validate_training_contract(output / "20/assets/training_contract.json"), self.document
        )
        restored = checkpoints.restore_state(manager, state, loader, step=20)
        self.assertEqual(int(restored.step), 20)
        self.assertEqual(int(restored.opt_state[0].count), 20)
        checkpoints.save_state(
            manager, dataclasses.replace(state, step=jnp.array(21)), loader, 21, training_contract=self.document
        )
        manager.wait_until_finished()
        self.assertEqual(tuple(manager.all_steps()), (20, 21))
        self.assertTrue((output / "20/train_state").is_dir())


class TrainLauncherTest(unittest.TestCase):
    @mock.patch.dict(
        os.environ,
        {
            "ROBOMME_CONFIG": "inherited-config",
            "ROBOMME_NUM_TRAIN_STEPS": "20",
            "ROBOMME_LR_DECAY_STEPS": "60000",
            "ROBOMME_SAVE_STEPS": "20",
            "ROBOMME_LOG_INTERVAL": "1",
            "ROBOMME_RESUME": "1",
            "ROBOMME_TRAINING_CONTRACT": "/inherited/contract.json",
            "ROBOMME_UPLOAD_RECEIPTS_DIR": "/inherited/receipts",
        },
    )
    def test_default_and_smoke_args_preserve_independent_decay_and_empty_receipts(self):
        with tempfile.TemporaryDirectory(prefix="ltmpi-launch-", dir="/scratch/hongze") as temporary:
            root = Path(temporary)
            for name in ("dataset", "memory", "anchor", "assets/id", "params"):
                (root / name).mkdir(parents=True)
            (root / "memory/manifest.json").write_text(json.dumps({"latent_normalization": {"mean": [0], "std": [1]}}))
            (root / "anchor/manifest.json").write_text("{}")
            (root / "assets/id/norm_stats.json").write_text("{}")
            env_file = root / ".env"
            env_file.write_text("ROBOMME_UPLOAD_RECEIPTS_DIR=/formal/receipts\nROBOMME_RESUME=0\n")
            # 验收入口会导出真实 run 的 .env；夹具只继承系统环境，训练值由本例显式声明。
            inherited_env = {
                key: value for key, value in os.environ.items() if not key.startswith(("ROBOMME_", "LTMPI_"))
            }
            env = {
                **inherited_env,
                "PYTHON_BIN": sys.executable,
                "ROBOMME_ENV_FILE": str(env_file),
                "ROBOMME_VIDEO_DATASET": str(root / "dataset"),
                "ROBOMME_MEMORY_CACHE": str(root / "memory"),
                "ROBOMME_ANCHOR_CACHE": str(root / "anchor"),
                "ROBOMME_ASSETS_ROOT": str(root / "assets"),
                "ROBOMME_ASSET_ID": "id",
                "PI05_BASE_PARAMS": str(root / "params"),
                "ROBOMME_UPLOAD_RECEIPTS_DIR": "",
                "OPENPI_DATASET_LOCAL_ROOT": "",
            }
            script = Path(__file__).resolve().parents[1] / "bin/train.sh"
            args = json.loads(subprocess.check_output(["bash", str(script), "print-args"], env=env, text=True))
            self.assertEqual(args[0], "pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default")
            self.assertEqual(args[args.index("--log-interval") + 1], "20")
            self.assertNotIn("--resume", args)
            self.assertNotIn("--training-contract-path", args)
            self.assertNotIn("--checkpoint-upload-receipts-dir", args)
            self.assertEqual(args[args.index("--num-train-steps") + 1], "120000")
            self.assertEqual(args[args.index("--lr-schedule.decay-steps") + 1], "120000")
            self.assertEqual(
                args[args.index("--save-steps") + 1 :], [str(step) for step in range(10000, 120001, 10000)]
            )
            env.update(
                ROBOMME_NUM_TRAIN_STEPS="21",
                ROBOMME_LR_DECAY_STEPS="60000",
                ROBOMME_SAVE_STEPS="21",
                ROBOMME_LOG_INTERVAL="1",
                ROBOMME_RESUME="1",
                ROBOMME_TRAINING_CONTRACT="/run/contract.json",
            )
            args = json.loads(subprocess.check_output(["bash", str(script), "print-args"], env=env, text=True))
            self.assertEqual(args[args.index("--num-train-steps") + 1], "21")
            self.assertEqual(args[args.index("--lr-schedule.decay-steps") + 1], "60000")
            self.assertIn("--resume", args)
            self.assertNotIn("--checkpoint-upload-receipts-dir", args)
            self.assertEqual(args[args.index("--training-contract-path") + 1], "/run/contract.json")


if __name__ == "__main__":
    unittest.main()
