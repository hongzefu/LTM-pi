"""验收入口CPU测试：真Orbax小状态与故障反例，不生成GPU通过结论。"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def frozen_audit(self, config_name="frozen-fixture"):
        vision = {"sha256": "a" * 64, "leaf_count": 2, "parameter_count": 32, "dtypes": ["bfloat16"]}
        initial = {
            "step": 0,
            "live_vision": vision,
            "ema_vision": copy.deepcopy(vision),
            "optimizer_vision_numeric_paths": [],
            "optimizer_counts": {"adam/count": 0},
            "nonvision_probes": {"PaliGemma/llm/w": "b" * 64},
        }
        final = copy.deepcopy(initial)
        final.update(step=20, optimizer_counts={"adam/count": 20}, nonvision_probes={"PaliGemma/llm/w": "c" * 64})
        return {
            "schema_version": 1,
            "kind": "ltmpi_frozen_siglip_audit",
            "status": "PASS",
            "config_name": config_name,
            "contract_sha256": "fixture",
            "expected_steps": 20,
            "metrics_count": 20,
            "initial": initial,
            "final": final,
            "checks": {
                key: True
                for key in (
                    "vision_unchanged",
                    "ema_vision_unchanged",
                    "vision_bfloat16",
                    "optimizer_vision_empty",
                    "nonvision_updated",
                    "optimizer_counts_match_step",
                    "completed_expected_steps",
                    "metrics_finite",
                )
            },
        }

    def write_checkpoint_files(self, checkpoint, contract):
        for name in ("params", "train_state"):
            (checkpoint / name).mkdir(parents=True)
            for filename in ("_METADATA", "manifest.ocdbt"):
                (checkpoint / name / filename).write_text("fixture")
            for folder in ("d", "ocdbt.process_0/d"):
                directory = checkpoint / name / folder
                directory.mkdir(parents=True)
                (directory / "fixture").write_bytes(b"payload")
        (checkpoint / "assets").mkdir()
        (checkpoint / "assets" / "norm_stats.json").write_text("{}")
        (checkpoint / "assets" / "training_contract.json").write_bytes(contract.read_bytes())
        runtime.write_json(checkpoint / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 1})

    def frozen_environment(self):
        from openpi.shared.ltmpi_contract import FROZEN_S4_CONFIG_NAME

        dataset, memory, anchors = (self.root / name for name in ("dataset", "memory", "anchors"))
        for path in (dataset, memory, anchors):
            path.mkdir()
        manifest = {
            "status": "complete",
            "dataset_root": str(dataset),
            "source_input_pin": "fixture",
            "transition_frame_stride": 4,
            "latent_dim": 64,
            "cached_total_frames": 482,
        }
        anchor_manifest = {key: manifest[key] for key in ("status", "dataset_root", "source_input_pin")}
        for root, item, keys in (
            (memory, manifest, ("global_latents", "global_frame_index")),
            (
                anchors,
                anchor_manifest,
                ("images_file", "execution_images_file", "episode_to_anchor_file", "exec_start_idx_file"),
            ),
        ):
            for key in keys:
                path = root / (key + ".bin")
                path.write_bytes(b"cache-fixture")
                item[key] = path.name
                item[key.removesuffix("_file") + "_sha256"] = runtime.sha256(path)
            runtime.write_json(root / "manifest.json", item)
        contract = self.root / "training_contract.json"
        runtime.write_json(contract, {"contract_sha256": "fixture"})
        return {
            **os.environ,
            "ROBOMME_CONFIG": FROZEN_S4_CONFIG_NAME,
            "ROBOMME_EXP": "frozen-cpu-smoke-fixture",
            "RUN_BASE_DIR": str(self.root / "logs"),
            "CHECKPOINT_BASE_DIR": str(self.root / "checkpoints"),
            "ROBOMME_MEMORY_CACHE": str(memory),
            "ROBOMME_ANCHOR_CACHE": str(anchors),
            "ROBOMME_VIDEO_DATASET": str(dataset),
            "ROBOMME_NUM_TRAIN_STEPS": "60000",
            "ROBOMME_UPLOAD_RECEIPTS_DIR": "",
            "HF_CACHE": str(self.root / "hf"),
            "ROBOMME_TRAINING_CONTRACT": str(contract),
        }

    def test_frozen_audit_requires_evidence_even_when_checks_claim_pass(self):
        path = self.root / "audit.json"
        good = self.frozen_audit()
        runtime.write_json(path, good)
        self.assertEqual(runtime.verify_frozen_audit(path, "frozen-fixture", "fixture"), good)
        variants = []
        for key, value in (
            ("status", "INITIALIZED"),
            ("metrics_count", 19),
            ("config_name", "wrong"),
            ("contract_sha256", "wrong"),
        ):
            bad = copy.deepcopy(good)
            bad[key] = value
            variants.append(bad)
        for key, value in (
            ("live_vision", {**good["final"]["live_vision"], "sha256": "d" * 64}),
            ("ema_vision", {**good["final"]["ema_vision"], "dtypes": ["float32"]}),
            ("optimizer_vision_numeric_paths", ["opt_state/PaliGemma/img/w"]),
            ("optimizer_counts", {"adam/count": 19}),
            ("nonvision_probes", good["initial"]["nonvision_probes"]),
        ):
            bad = copy.deepcopy(good)
            bad["final"][key] = value
            variants.append(bad)
        for bad in variants:
            runtime.write_json(path, bad)
            with self.subTest(audit=bad), self.assertRaises(ValueError):
                runtime.verify_frozen_audit(path, "frozen-fixture", "fixture")

    def test_frozen_checkpoint_files_only_require_complete_committed_parts(self):
        contract = self.root / "contract.json"
        contract.write_text("{}")
        checkpoint = self.root / "20"
        self.write_checkpoint_files(checkpoint, contract)
        with mock.patch.object(runtime, "checkpoint_counters", side_effect=AssertionError("禁止恢复")):
            result = runtime.committed_checkpoint_files(checkpoint, contract)
        self.assertEqual(result["step"], 20)
        self.assertEqual(result["verification"], "committed_metadata_and_files_only")
        self.assertGreater(result["complete_checkpoint_bytes"], 0)
        shard = checkpoint / "train_state" / "ocdbt.process_0" / "d" / "fixture"
        shard.unlink()
        with self.assertRaisesRegex(ValueError, "OCDBT数据文件"):
            runtime.committed_checkpoint_files(checkpoint, contract)
        shard.write_bytes(b"payload")
        (checkpoint / "train_state" / "manifest.ocdbt").unlink()
        with self.assertRaisesRegex(ValueError, "保存件缺少"):
            runtime.committed_checkpoint_files(checkpoint, contract)

    def test_frozen_existing_cache_must_match_hash_without_rebuild(self):
        env = self.frozen_environment()
        result = runtime.existing_frozen_cache(env)
        self.assertEqual(result["mode"], "reuse_read_only")
        (Path(env["ROBOMME_MEMORY_CACHE"]) / "global_latents.bin").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "文件身份不符"):
            runtime.existing_frozen_cache(env)

    def test_frozen_mode_runs_one_train_command_and_records_skipped_stages(self):
        env = self.frozen_environment()
        output = self.root / "results"
        source = self.root / "source.env"
        source.write_text("ROBOMME_NUM_TRAIN_STEPS=60000\n")
        checkpoint = Path(env["CHECKPOINT_BASE_DIR"]) / env["ROBOMME_CONFIG"] / env["ROBOMME_EXP"] / "20"
        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            self.assertEqual(argv, ["bash", "bin/train.sh", "run"])
            child = kwargs["env"]
            for key, value in (
                ("ROBOMME_NUM_TRAIN_STEPS", "20"),
                ("ROBOMME_SAVE_STEPS", "20"),
                ("ROBOMME_RESUME", "0"),
                ("ROBOMME_LR_DECAY_STEPS", "60000"),
            ):
                self.assertEqual(child[key], value)
            self.assertEqual(child["JAX_PLATFORMS"], "cuda")
            runtime.write_json(Path(child["LTMPI_FROZEN_AUDIT_REPORT"]), self.frozen_audit(env["ROBOMME_CONFIG"]))
            self.write_checkpoint_files(checkpoint, Path(env["ROBOMME_TRAINING_CONTRACT"]))
            return {
                "rows": self.rows(),
                "peak_sampled_bytes": {
                    "checkpoint": runtime.directory_bytes(checkpoint),
                    "hf_cache": 0,
                    "arrow_cache": 32,
                },
            }

        def config(env_file, **kwargs):
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(kwargs["command_env"]["ROBOMME_NUM_TRAIN_STEPS"], "20")
            return SimpleNamespace(num_train_steps=20, resume=False, save_steps=(20,), log_interval=1)

        argv = ["verify", "--env-file", str(source), "--mode", "frozen-train-smoke", "--output-dir", str(output)]
        with (
            mock.patch.object(runtime, "read_env", return_value=env),
            mock.patch.object(runtime, "require_clean_fork"),
            mock.patch.object(runtime, "effective_config", side_effect=config),
            mock.patch.object(runtime, "run_command", side_effect=command),
            mock.patch.object(runtime, "checkpoint_counters", side_effect=AssertionError("禁止恢复")),
            mock.patch.object(runtime, "online_check", side_effect=AssertionError("禁止在线验证")),
            mock.patch.object(runtime.sys, "argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(runtime.main(), 0)
        self.assertEqual(len(calls), 1)
        resources = json.loads((output / "resources.json").read_text())
        self.assertEqual(resources["completed_step"], 20)
        self.assertEqual(resources["updates"], 20)
        self.assertNotIn("resumed_to", resources)
        self.assertEqual(resources["resume_test"], "skipped_user_authorized")
        self.assertEqual(resources["restore_test"], "skipped_user_authorized")
        self.assertEqual(resources["independent_restore_copy_purpose"], "remote_readback_budget_only")
        self.assertEqual(resources["freeze_audit"]["status"], "PASS")
        self.assertFalse((output / "resume21.log").exists())

    def test_frozen_config_cannot_enter_legacy_workers(self):
        env = self.frozen_environment()
        for args in (("--mode", "restore"), ("--worker", "checkpoint"), ("--mode", "smoke")):
            with (
                mock.patch.object(runtime, "read_env", return_value=env),
                mock.patch.object(runtime.sys, "argv", ["verify", "--env-file", "fixture", *args]),
            ):
                with self.assertRaisesRegex(ValueError, "不执行恢复或在线worker"):
                    runtime.main()

    def test_frozen_requires_clean_producer_fork(self):
        with mock.patch.object(runtime.subprocess, "check_output", return_value=" M scripts/train.py\n"):
            with self.assertRaisesRegex(ValueError, "clean fork"):
                runtime.require_clean_fork()

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

    def test_local_1_147_percent_passes_with_global_below_one_percent(self):
        reference = np.array([[1.0, 0], [100, 100]], dtype=np.float64)
        actual = reference.copy()
        actual[0, 0] += 0.01147
        result = runtime.numeric_parity(actual, reference, np.zeros(2), np.ones(2))
        self.assertLess(result["relative_l2"], 0.01)
        self.assertAlmostEqual(result["normalized_token_relative_l2_max"], 0.01147)
        self.assertEqual(result["global_relative_l2_tolerance"], 0.01)
        self.assertEqual(result["normalized_token_relative_l2_tolerance"], 0.02)

    def test_local_above_two_percent_still_fails(self):
        reference = np.array([[1.0, 0], [100, 100]], dtype=np.float64)
        actual = reference.copy()
        actual[0, 0] += 0.02001
        self.assertLess(np.linalg.norm(actual - reference) / np.linalg.norm(reference), 0.01)
        with self.assertRaisesRegex(ValueError, "全局阈值=0.01.*逐token阈值=0.02"):
            runtime.numeric_parity(actual, reference, np.zeros(2), np.ones(2))

    def test_global_above_one_percent_still_fails_with_local_below_two_percent(self):
        reference = np.ones((2, 2), dtype=np.float64)
        actual = reference * 1.015
        normalized_local = np.linalg.norm(actual - reference, axis=1) / np.linalg.norm(reference, axis=1)
        self.assertLess(float(normalized_local.max()), 0.02)
        with self.assertRaisesRegex(ValueError, "全局阈值=0.01.*逐token阈值=0.02"):
            runtime.numeric_parity(actual, reference, np.zeros(2), np.ones(2))

    def test_shape_and_invalid_std_remain_rejected(self):
        reference = np.ones((2, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "LAM形状不同"):
            runtime.numeric_parity(reference[:1], reference, np.zeros(2), np.ones(2))
        with self.assertRaisesRegex(ValueError, "归一化形状错误"):
            runtime.numeric_parity(reference, reference, np.zeros(2), np.zeros(2))

    def test_empty_comparison_reports_both_tolerances(self):
        values = np.empty((0, 64))
        result = runtime.numeric_parity(values, values, np.zeros(64), np.ones(64))
        self.assertEqual(result["tokens"], 0)
        self.assertEqual(result["global_relative_l2_tolerance"], 0.01)
        self.assertEqual(result["normalized_token_relative_l2_tolerance"], 0.02)

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
