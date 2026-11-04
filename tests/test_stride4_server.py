from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts import serve
from scripts.precompute_dreamdojo_memory import alignment_for_transition_frame_stride


class _Tokenizer:
    def encode(self, _text, add_bos=False):
        return [1] if add_bos else [2]


class _CapturePolicy:
    def infer(self, element, noise):
        self.element = element
        return {"actions": np.zeros((50, 8), dtype=np.float32)}


def _frames(count, offset=0):
    return np.broadcast_to((np.arange(count, dtype=np.uint8) + offset)[:, None, None, None], (count, 2, 2, 3)).copy()


def _session():
    return serve.MemorySession(
        policy=_CapturePolicy(),
        dreamdojo_model=None,
        encode_full=None,
        preprocess=None,
        device="cpu",
        memory_horizon=1800,
        latent_dim=64,
        action_horizon=50,
        action_dim=8,
        prompt_tokenizer=SimpleNamespace(_tokenizer=_Tokenizer()),
        max_token_len=200,
        pair_batch_size=2,
        use_memory_segments=True,
        use_demo_anchor=True,
        use_execution_anchor=True,
        transition_frame_stride=4,
        two_view_memory=True,
        memory_lag=3,
        latent_storage_dtype="float32",
    )


def _pair_markers(frames):
    # 用输入帧号记录真正传入编码器的左右端点，不预填 session.latents。
    values = frames[:, 0, 0, 0].astype(np.float32)
    result = np.zeros((len(frames) - 1, 32), dtype=np.float32)
    result[:, 0] = values[:-1]
    result[:, 1] = values[1:]
    return result


def _identity_preprocess(frames, _preprocess, chunk_size):
    return frames


class Stride4SessionTest(unittest.TestCase):
    def _append(self, session, frames, wrists, sizes, exec_start):
        cursor = 0
        with mock.patch("scripts.precompute_dreamdojo_memory.preprocess_frames", _identity_preprocess):
            with mock.patch.object(session, "_encode_dreamdojo_pairs", side_effect=_pair_markers):
                for size in sizes:
                    result = session.add_buffer(
                        {
                            "images": frames[cursor : cursor + size],
                            "wrist_images": wrists[cursor : cursor + size],
                            "exec_start_idx": exec_start if cursor == 0 else 0,
                        }
                    )
                    self.assertEqual(result["added_frames"], size)
                    cursor += size
        self.assertEqual(cursor, len(frames))

    def test_actual_endpoints_and_all_boundary_remainders_are_chunk_invariant(self):
        count = 35
        frames, wrists = _frames(count), _frames(count, 100)
        endpoints = np.arange(4, count, 4)
        for boundary in range(8, 12):
            chunks = (
                [count],
                [boundary + 1] + [1] * (count - boundary - 1),
                [boundary + 1, 2, 7, count - boundary - 10],
            )
            for sizes in chunks:
                with self.subTest(exec_start=boundary, sizes=sizes):
                    session = _session()
                    self._append(session, frames, wrists, sizes, boundary)
                    actual = np.stack(session.latents)
                    np.testing.assert_array_equal(actual[:, 0], endpoints - 4)
                    np.testing.assert_array_equal(actual[:, 1], endpoints)
                    np.testing.assert_array_equal(actual[:, 32], endpoints - 4 + 100)
                    np.testing.assert_array_equal(actual[:, 33], endpoints + 100)
                    expected_segments = np.where(endpoints < boundary, 1, np.where(endpoints == boundary, 2, 3))
                    np.testing.assert_array_equal(session.segment_ids, expected_segments)
                    result = session.infer({"prompt": "test"})
                    element = session.policy.element
                    visible = np.ones(len(endpoints), dtype=bool)
                    visible[np.flatnonzero(endpoints > boundary)[-3:]] = False
                    valid = element["memory_mask"]
                    np.testing.assert_array_equal(element["memory_latents"][valid], actual[visible])
                    np.testing.assert_array_equal(element["memory_segment_ids"][valid], expected_segments[visible])
                    self.assertEqual(element["memory_latents"].shape, (1800, 64))
                    self.assertEqual(result["memory_timing"]["lag_hidden_transitions"], 3)
                    self.assertEqual(result["memory_timing"]["total_transitions"], len(endpoints))
                    np.testing.assert_array_equal(element["memory_demo_start_image"], frames[0])
                    np.testing.assert_array_equal(element["memory_execution_start_image"], frames[boundary])
                    self.assertTrue(element["memory_demo_start_mask"])
                    self.assertTrue(element["memory_execution_start_mask"])
                    np.testing.assert_array_equal(session.previous_stride_head_frame, frames[32])
                    np.testing.assert_array_equal(session.previous_stride_wrist_frame, wrists[32])

    def test_empty_history_and_zero_through_four_execution_tokens(self):
        for boundary in (0, 8):
            for execution_count in range(5):
                for trailing_frames in range(4):
                    count = boundary + 4 * execution_count + 1 + trailing_frames
                    with self.subTest(boundary=boundary, execution_count=execution_count, tail=trailing_frames):
                        session = _session()
                        self._append(session, _frames(count), _frames(count, 100), [count], boundary)
                        result = session.infer({"prompt": "test"})
                        element = session.policy.element
                        valid = element["memory_mask"]
                        expected = ([4, 8] if boundary else []) + ([boundary + 4] if execution_count == 4 else [])
                        np.testing.assert_array_equal(element["memory_latents"][valid, 1], expected)
                        self.assertEqual(result["memory_timing"]["lag_hidden_transitions"], min(3, execution_count))
                        self.assertEqual(bool(element["memory_demo_start_mask"]), boundary > 0)
                        self.assertEqual(element["memory_latents"].shape, (1800, 64))
                        if not expected:
                            self.assertFalse(valid.any())
                            self.assertFalse(element["memory_latents"].any())
                            self.assertFalse(element["memory_segment_ids"].any())
                        elif boundary:
                            np.testing.assert_array_equal(element["memory_segment_ids"][valid][:2], [1, 2])

    def test_first_chunk_requires_execution_anchor_and_reset_clears_history(self):
        session = _session()
        with self.assertRaisesRegex(RuntimeError, "before add_buffer"):
            session.infer({"prompt": "test"})
        with self.assertRaisesRegex(ValueError, "first add_buffer"):
            session.add_buffer({"images": _frames(4), "wrist_images": _frames(4, 100), "exec_start_idx": 4})
        self.assertIsNone(session.exec_start_idx)
        self.assertEqual(session.total_frames, 0)
        self._append(session, _frames(25), _frames(25, 100), [9, 16], 8)
        session.reset(7)
        self.assertEqual(session.latents, [])
        self.assertEqual(session.segment_ids, [])
        self.assertIsNone(session.previous_stride_head_frame)
        self.assertIsNone(session.previous_stride_wrist_frame)
        self.assertIsNone(session.demo_start_image)
        self.assertIsNone(session.execution_start_image)

    def test_actual_causal_encoder_receives_consecutive_compressed_endpoints(self):
        captured = []

        def encode_full(_model, pairs, **kwargs):
            self.assertEqual(kwargs, {"sample": False, "use_ckpt": False})
            # 真实 CPU 编帧对路径会转 bf16 并除以255，因此读取后四舍五入还原帧号。
            values = (pairs[:, :, 0, 0, 0].float() * 255).round()
            captured.extend(values.tolist())
            return {"z_mu": values[:, 1:2].repeat(1, 32)}

        session = _session()
        session.encode_full = encode_full
        with mock.patch("torch.autocast", return_value=contextlib.nullcontext()):
            result = session._encode_dreamdojo_pairs(_frames(17)[::4])
        self.assertEqual(captured, [[0, 4], [4, 8], [8, 12], [12, 16]])
        np.testing.assert_array_equal(result[:, 0], [4, 8, 12, 16])
        self.assertEqual(result.dtype, np.float32)


class Stride4ConfigurationTest(unittest.TestCase):
    @staticmethod
    def _config(lag, stride):
        return SimpleNamespace(
            data=SimpleNamespace(
                base_config=SimpleNamespace(
                    memory_random_drop_execution_tail_min=lag,
                    memory_random_drop_execution_tail_max=lag,
                )
            ),
            model=SimpleNamespace(memory_stride=stride),
            policy_metadata={"memory_random_drop_execution_tail": f"fixed_integer_{lag}_per_sample"},
        )

    def test_lag_and_stride_defaults_and_explicit_overrides(self):
        for lag, stride in ((10, 1), (3, 4)):
            config = self._config(lag, stride)
            for explicit in (False, True):
                args = argparse.Namespace(
                    memory_lag=lag if explicit else None, transition_frame_stride=stride if explicit else None
                )
                self.assertEqual(serve._configured_memory_settings(config, args), (lag, stride))
            for field, values in (("memory_lag", (-1, 0, lag + 1)), ("transition_frame_stride", (-1, 0, stride + 1))):
                for value in values:
                    args = argparse.Namespace(memory_lag=None, transition_frame_stride=None)
                    setattr(args, field, value)
                    with self.assertRaises(ValueError):
                        serve._configured_memory_settings(config, args)

    def test_variable_tail_and_inconsistent_metadata_are_rejected(self):
        config = self._config(3, 4)
        args = argparse.Namespace(memory_lag=None, transition_frame_stride=None)
        config.data.base_config.memory_random_drop_execution_tail_max = 4
        with self.assertRaises(ValueError):
            serve._configured_memory_settings(config, args)
        config.data.base_config.memory_random_drop_execution_tail_max = 3
        config.policy_metadata["memory_random_drop_execution_tail"] = "fixed_integer_10_per_sample"
        with self.assertRaises(ValueError):
            serve._configured_memory_settings(config, args)

    @staticmethod
    def _manifest(stride):
        return {
            "status": "complete",
            "missing_episodes": [],
            "latent_dim": 64,
            "latent_normalization": {"mean": [0.0] * 64, "std": [1.0] * 64},
            "alignment": alignment_for_transition_frame_stride(stride),
            "image_keys": ["image", "wrist_image"],
            "latent_layout": "concat(image_z_mu[32],wrist_image_z_mu[32])",
            "latent_dtype": "float32",
            "preprocess_version": "dreamdojo.lam400k.rgb_uint8.crop_4x3.resize_480x640.resize_240x320.v1",
            "transition_frame_stride": stride,
            "checkpoint_format": serve.EXPECTED_CHECKPOINT_FORMAT,
            "checkpoint_sha256": serve.EXPECTED_ENCODER_SHA256,
        }

    def test_manifest_requires_selected_stride_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for stride in (1, 4):
                manifest = self._manifest(stride)
                path.write_text(json.dumps(manifest))
                loaded, mean, std = serve._load_manifest(path, 64, transition_frame_stride=stride)
                self.assertEqual(loaded, manifest)
                np.testing.assert_array_equal(mean, np.zeros(64))
                np.testing.assert_array_equal(std, np.ones(64))
                with self.assertRaises(ValueError):
                    serve._load_manifest(path, 64, transition_frame_stride=5 - stride)
            for key, value in (
                ("transition_frame_stride", True),
                ("transition_frame_stride", 4.0),
                ("alignment", serve.EXPECTED_ALIGNMENT),
                ("status", "partial"),
            ):
                manifest = self._manifest(4)
                manifest[key] = value
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    serve._load_manifest(path, 64, transition_frame_stride=4)

    def test_cli_defers_lag_and_stride_to_config(self):
        with mock.patch(
            "sys.argv",
            [
                "serve.py",
                "--checkpoint",
                "checkpoint",
                "--memory-manifest",
                "manifest",
                "--dreamdojo-checkpoint",
                "encoder",
                "--config",
                serve.STRIDE4_CONFIG,
            ],
        ):
            args = serve.parse_args()
        self.assertIsNone(args.memory_lag)
        self.assertIsNone(args.transition_frame_stride)
        self.assertEqual(args.config, serve.STRIDE4_CONFIG)

    def test_shell_passes_environment_config_without_overriding_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "capture-python"
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            env = {"PATH": os.environ["PATH"], "PYTHON_BIN": str(python), "ROBOMME_CONFIG": serve.STRIDE4_CONFIG}
            script = Path(serve.__file__).resolve().parents[1] / "bin" / "serve.sh"
            for extra in ([], ["--config", serve.DEFAULT_CONFIG], [f"--config={serve.DEFAULT_CONFIG}"]):
                output = subprocess.check_output(
                    ["bash", str(script), "--checkpoint", "test", *extra], env=env, text=True
                ).splitlines()
                config_args = [arg for arg in output if arg == "--config" or arg.startswith("--config=")]
                self.assertEqual(len(config_args), 1)
                self.assertEqual(serve.STRIDE4_CONFIG in output, not extra)


class ServerCheckpointContractTest(unittest.TestCase):
    def setUp(self):
        from openpi.shared import ltmpi_contract
        from openpi.training import config as training_config

        self.contract_api = ltmpi_contract
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.checkpoint = root / "checkpoint"
        (self.checkpoint / "params").mkdir(parents=True)
        (self.checkpoint / "train_state").mkdir()
        (self.checkpoint / "params" / "_METADATA").write_text("{}")
        (self.checkpoint / "params" / "manifest.ocdbt").write_text("fixture")
        (self.checkpoint / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
        self.norm_path = self.checkpoint / "assets" / "fixture" / "norm_stats.json"
        self.norm_path.parent.mkdir(parents=True)
        self.norm_path.write_text('{"norm_stats": {"actions": {"mean": [0], "std": [1]}}}')
        self.manifest_path = root / "manifest.json"
        self.manifest = Stride4ConfigurationTest._manifest(4)
        self.manifest["source_input_pin"] = "a" * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        anchor_path = root / "anchor-manifest.json"
        anchor_path.write_text(json.dumps({"status": "complete", "source_input_pin": "a" * 64}))
        old = training_config.get_config(serve.DEFAULT_CONFIG)
        model_values = {"memory_stride": 4, "memory_latent_mean": (0.0,) * 64, "memory_latent_std": (1.0,) * 64}
        if hasattr(old.model, "memory_anchor_pool_grid"):
            model_values["memory_anchor_pool_grid"] = 4
        self.config = dataclasses.replace(
            old,
            name=serve.STRIDE4_CONFIG,
            exp_name="server-contract-fixture",
            model=dataclasses.replace(old.model, **model_values),
            data=dataclasses.replace(
                old.data,
                base_config=dataclasses.replace(
                    old.data.base_config,
                    memory_random_drop_execution_tail_min=3,
                    memory_random_drop_execution_tail_max=3,
                ),
            ),
            policy_metadata={**old.policy_metadata, "memory_random_drop_execution_tail": "fixed_integer_3_per_sample"},
        )
        self.contract = ltmpi_contract.build_training_contract(
            self.config,
            {"source_input_pin": "a" * 64},
            memory_manifest_path=self.manifest_path,
            action_norm_path=self.norm_path,
            anchor_manifest_path=anchor_path,
            main_commit="b" * 40,
            fork_commit="c" * 40,
        )
        self.contract_path = self.checkpoint / "assets" / "training_contract.json"
        ltmpi_contract.write_json_atomic(self.contract_path, self.contract)

    def _preflight(self, config):
        from openpi.training import config as training_config

        argv = [
            "serve.py",
            "--checkpoint",
            str(self.checkpoint),
            "--memory-manifest",
            str(self.manifest_path),
            "--dreamdojo-checkpoint",
            "encoder",
            "--config",
            config.name,
            "--preflight-only",
        ]
        identity = SimpleNamespace(format=serve.EXPECTED_CHECKPOINT_FORMAT, sha256=serve.EXPECTED_ENCODER_SHA256)
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch("sys.argv", argv))
            stack.enter_context(mock.patch.object(training_config, "get_config", return_value=config))
            stack.enter_context(
                mock.patch("scripts.precompute_dreamdojo_memory.resolve_checkpoint_identity", return_value=identity)
            )
            encoder_loader = stack.enter_context(
                mock.patch("scripts.precompute_dreamdojo_memory.load_dreamdojo_encoder")
            )
            stack.enter_context(contextlib.redirect_stdout(output))
            serve.main()
            encoder_loader.assert_not_called()
        return json.loads(output.getvalue())

    def test_preflight_verifies_contract_without_claiming_model_restore(self):
        result = self._preflight(self.config)
        self.assertTrue(result["training_contract_verified"])
        self.assertFalse(result["models_loaded"])
        self.assertEqual(result["memory_lag"], 3)
        self.assertEqual(result["transition_frame_stride"], 4)

    def test_new_checkpoint_without_contract_is_rejected(self):
        self.contract_path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self._preflight(self.config)

    def test_same_shape_wrong_semantics_and_old_config_name_cannot_bypass_contract(self):
        # 类型编号不改变参数形状，仍须按模型语义拒绝；改回旧名字也不能跳过合同。
        wrong_model = dataclasses.replace(self.config.model, memory_execution_anchor_type_id=4)
        for wrong in (
            dataclasses.replace(self.config, model=wrong_model),
            dataclasses.replace(self.config, name=serve.DEFAULT_CONFIG),
        ):
            with self.subTest(config=wrong.name, anchor_type=wrong.model.memory_execution_anchor_type_id):
                with self.assertRaises(ValueError):
                    self._preflight(wrong)

    def test_changed_manifest_and_action_norm_are_rejected(self):
        self.manifest["source_input_pin"] = "d" * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            self._preflight(self.config)
        self.manifest["source_input_pin"] = "a" * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.norm_path.write_text('{"norm_stats": {"actions": {"mean": [9], "std": [1]}}}')
        with self.assertRaises(ValueError):
            self._preflight(self.config)

    def test_old_checkpoint_without_contract_retains_original_path(self):
        from openpi.training import config as training_config

        self.contract_path.unlink()
        self.manifest_path.write_text(json.dumps(Stride4ConfigurationTest._manifest(1)))
        result = self._preflight(training_config.get_config(serve.DEFAULT_CONFIG))
        self.assertFalse(result["training_contract_verified"])
        self.assertFalse(result["models_loaded"])
        self.assertEqual(result["memory_lag"], 10)
        self.assertEqual(result["transition_frame_stride"], 1)


if __name__ == "__main__":
    unittest.main()
