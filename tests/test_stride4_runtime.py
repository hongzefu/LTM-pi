"""验收入口CPU测试：真Orbax小状态与故障反例，不生成GPU通过结论。"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import verify_stride4_runtime as runtime


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ltmpi-runtime-cpu-", dir="/scratch/hongze")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def rows(self, first=0, stop=20):
        return [
            {
                "step": step,
                "elapsed_seconds": 100 + step * 2,
                "metrics": f"loss=1.0, grad_norm=0.1, completed_step={step + 1}, "
                f"learning_rate={2.5e-5 / 1001 + step / 1000 * (2.5e-5 - 2.5e-5 / 1001)}",
            }
            for step in range(first, stop)
        ]

    def contract_config(self):
        @dataclasses.dataclass(frozen=True)
        class Model:
            memory_demo_anchor_type_id: int = 4
            memory_execution_anchor_type_id: int = 5
            memory_segment_vocab_size: int = 6
            memory_anchor_pool_grid: int = 4

        @dataclasses.dataclass(frozen=True)
        class Config:
            model: Model
            name: str = runtime.CONFIG

        return Config(Model())

    def test_wrong_contract_mutates_only_existing_anchor_ids_and_requires_rejection(self):
        config = self.contract_config()
        checkpoint, manifest = self.root / "21", self.root / "manifest.json"

        def validator(actual_checkpoint, candidate, *, manifest_path):
            self.assertEqual(actual_checkpoint, checkpoint)
            self.assertEqual(manifest_path, manifest)
            self.assertEqual(candidate.name, config.name)
            differences = {
                key
                for key, value in dataclasses.asdict(config.model).items()
                if dataclasses.asdict(candidate.model)[key] != value
            }
            self.assertEqual(differences, {"memory_demo_anchor_type_id", "memory_execution_anchor_type_id"})
            self.assertEqual(candidate.model.memory_demo_anchor_type_id, 5)
            self.assertEqual(candidate.model.memory_execution_anchor_type_id, 4)
            raise ValueError("checkpoint 合同身份不符：model_config_sha256")

        self.assertEqual(runtime.verify_wrong_contract_rejected(checkpoint, config, manifest, validator), 1)
        self.assertEqual(config.model.memory_demo_anchor_type_id, 4)
        self.assertEqual(config.model.memory_execution_anchor_type_id, 5)

    def test_accepting_wrong_contract_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "未被保存件合同拒绝"):
            runtime.verify_wrong_contract_rejected(
                self.root / "21", self.contract_config(), self.root / "manifest.json", mock.Mock(return_value={})
            )

    def test_missing_assets_cannot_count_as_wrong_contract_rejection(self):
        for error in (FileNotFoundError("缺少保存件"), ValueError("checkpoint 尚未完整提交")):
            with self.subTest(error=str(error)), self.assertRaises(type(error)):
                runtime.verify_wrong_contract_rejected(
                    self.root / "21", self.contract_config(), self.root / "manifest.json", mock.Mock(side_effect=error)
                )

    def test_all_updates_and_resume_are_required(self):
        result = runtime.verify_train_rows(self.rows(), 0, 20)
        self.assertEqual(result["updates"], 20)
        self.assertEqual(result["stable_step_seconds_median"], 2)
        runtime.verify_train_rows(self.rows(20, 21), 20, 21)
        for bad in (self.rows()[:-1], self.rows() + self.rows()[-1:], self.rows(0, 1)):
            with self.assertRaises(ValueError):
                runtime.verify_train_rows(bad, 0, 20)

    def test_nonfinite_and_wrong_lr_fail(self):
        for replacement in ("loss=nan", "grad_norm=inf", "learning_rate=0.1", "completed_step=20"):
            rows = self.rows()
            key = replacement.split("=")[0]
            parts = [
                replacement if part.strip().startswith(key + "=") else part for part in rows[7]["metrics"].split(",")
            ]
            rows[7]["metrics"] = ",".join(parts)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                runtime.verify_train_rows(rows, 0, 20)

    def test_per_token_normalized_bound_is_not_global_bound(self):
        reference = np.array([[1e-3, 0], [1000, 1000]], dtype=np.float64)
        actual = reference.copy()
        actual[0, 0] += 1e-4
        self.assertLess(np.linalg.norm(actual - reference) / np.linalg.norm(reference), 1e-2)
        with self.assertRaises(ValueError):
            runtime.numeric_parity(actual, reference, np.zeros(2), np.ones(2))
        result = runtime.numeric_parity(reference, reference, np.zeros(2), np.ones(2))
        self.assertEqual(result["normalized_token_relative_l2_max"], 0)

    def test_zero_reference_and_nan_are_rejected(self):
        for actual in (np.full((1, 2), 1e-6), np.full((1, 2), np.nan)):
            with self.assertRaises(ValueError):
                runtime.numeric_parity(actual, np.zeros((1, 2)), np.zeros(2), np.ones(2))

    def test_padding_and_segments_are_separate_from_values(self):
        reference = np.ones((2, 64), np.float32)
        element = {
            "memory_latents": np.pad(reference, ((1798, 0), (0, 0))),
            "memory_mask": np.arange(1800) >= 1798,
            "memory_segment_ids": np.pad(np.array([1, 3]), (1798, 0)),
        }
        runtime.compare_memory(element, reference, [1, 3], np.zeros(64), np.ones(64))
        element["memory_segment_ids"][-1] = 2
        with self.assertRaises(ValueError):
            runtime.compare_memory(element, reference, [1, 3], np.zeros(64), np.ones(64))
        element["memory_segment_ids"][-1] = 3
        element["memory_latents"][0, 0] = 1
        with self.assertRaises(ValueError):
            runtime.compare_memory(element, reference, [1, 3], np.zeros(64), np.ones(64))

    def test_boundary_startup_tail_and_endpoint_order(self):
        frames = np.arange(40)
        valid = (frames > 0) & (frames % 4 == 0)
        latents = np.repeat(frames[:, None], 64, axis=1).astype(np.float32)
        for boundary in (0, 8, 9, 10, 11):
            for last in range(boundary, 35):
                values, segments, endpoints = runtime.offline_visible(latents, valid, frames, boundary, last)
                candidates = list(range(4, last + 1, 4))
                execution = [frame for frame in candidates if frame > boundary or boundary == 0]
                expected = [frame for frame in candidates if frame not in execution[-3:]]
                np.testing.assert_array_equal(endpoints, expected)
                np.testing.assert_array_equal(values[:, 0], expected)
                np.testing.assert_array_equal(
                    segments, [1 if frame < boundary else 2 if frame == boundary else 3 for frame in expected]
                )

    def test_env_overlay_wins_and_shell_literals_stay_literals(self):
        source = self.root / "environment with spaces.env"
        source.write_text("ROBOMME_NUM_TRAIN_STEPS=60000\nROBOMME_RESUME=0\n")
        value = "literal $(touch unexpected) `echo bad` ' \" $HOME"
        target = runtime.overlay_env(
            source,
            self.root / "overlay.env",
            {"ROBOMME_NUM_TRAIN_STEPS": "20", "ROBOMME_RESUME": "1", "RUNTIME_LITERAL": value},
        )
        env = runtime.read_env(target)
        self.assertEqual(env["ROBOMME_NUM_TRAIN_STEPS"], "20")
        self.assertEqual(env["ROBOMME_RESUME"], "1")
        self.assertEqual(env["RUNTIME_LITERAL"], value)

    def test_command_error_is_not_success(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "退出码3"):
                runtime.run_command(
                    ["bash", "-c", "exit 3"],
                    env=dict(os.environ),
                    log=self.root / "failed.log",
                    watch={},
                    timeout=5,
                    interval=0.05,
                )

    def test_command_metrics_and_storage(self):
        (self.root / "asset").write_bytes(b"x" * 5000)
        with contextlib.redirect_stdout(io.StringIO()):
            result = runtime.run_command(
                ["bash", "-c", "printf 'Step 0: loss=1, grad_norm=2\\n'"],
                env=dict(os.environ),
                log=self.root / "output.log",
                watch={"fixture": self.root},
                timeout=5,
                interval=0.05,
            )
        self.assertEqual(result["rows"][0]["step"], 0)
        self.assertGreaterEqual(result["peak_sampled_bytes"]["fixture"], 5000)

    def test_real_orbax_partial_counter_restore(self):
        import orbax.checkpoint as ocp

        checkpoint = self.root / "20"
        checkpoint.mkdir()
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(
                checkpoint / "train_state",
                {
                    "step": np.asarray(20, np.int32),
                    "params": {"w": np.ones(4)},
                    "opt_state": {
                        "adam": {"count": np.asarray(20, np.int32), "mu": np.zeros(4)},
                        "schedule": {"count": np.asarray(20, np.int32)},
                    },
                },
            )
            checkpointer.save(checkpoint / "params", {"params": {"w": np.ones(4)}})
        assets = checkpoint / "assets" / "test"
        assets.mkdir(parents=True)
        (assets / "norm_stats.json").write_text("{}")
        (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
        result = runtime.checkpoint_counters(checkpoint)
        self.assertEqual(result["step"], 20)
        self.assertEqual(len(result["counts"]), 3)
        self.assertEqual(set(result["counts"].values()), {20})
        renamed = checkpoint.with_name("21")
        checkpoint.rename(renamed)
        with self.assertRaisesRegex(ValueError, "计数错误"):
            runtime.checkpoint_counters(renamed)

    def test_missing_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            runtime.checkpoint_counters(self.root / "21")

    def test_parent_60k_environment_cannot_override_smoke_and_failure_stops_chain(self):
        source = self.root / "source.env"
        source.write_text("ROBOMME_NUM_TRAIN_STEPS=60000\n")
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "manifest.json").write_text("{}")
        env = {
            **os.environ,
            "ROBOMME_CONFIG": runtime.CONFIG,
            "ROBOMME_EXP": "cpu-smoke-fixture",
            "RUN_BASE_DIR": str(self.root / "logs"),
            "CHECKPOINT_BASE_DIR": str(self.root / "checkpoints"),
            "ROBOMME_MEMORY_CACHE": str(cache),
            "ROBOMME_NUM_TRAIN_STEPS": "60000",
            "ROBOMME_UPLOAD_RECEIPTS_DIR": "",
            "HF_CACHE": str(self.root / "hf"),
        }
        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            if "--worker" in argv:
                self.assertEqual(argv[argv.index("--worker") + 1], "cache")
                runtime.write_json(Path(argv[argv.index("--result") + 1]), {"status": "pass"})
                return {"peak_sampled_bytes": {"hf_cache": 0, "arrow_cache": 0}}
            self.assertEqual(argv, ["bash", "bin/train.sh", "run"])
            self.assertEqual(kwargs["env"]["ROBOMME_NUM_TRAIN_STEPS"], "20")
            self.assertEqual(kwargs["env"]["ROBOMME_LR_DECAY_STEPS"], "60000")
            self.assertEqual(kwargs["env"]["ROBOMME_RESUME"], "0")
            self.assertEqual(kwargs["env"]["ROBOMME_UPLOAD_RECEIPTS_DIR"], "")
            raise RuntimeError("CPU故障注入，禁止继续")

        output = self.root / "results"
        argv = ["verify", "--env-file", str(source), "--mode", "smoke", "--output-dir", str(output)]
        with (
            mock.patch.object(runtime, "read_env", return_value=env),
            mock.patch.object(runtime, "run_command", side_effect=command),
            mock.patch.object(runtime.sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU故障"):
                runtime.main()
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads((output / "smoke-result.json").read_text())["status"], "failed")
        self.assertFalse((output / "resources.json").exists())


if __name__ == "__main__":
    unittest.main()
