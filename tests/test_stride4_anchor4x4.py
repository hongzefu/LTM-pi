from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.training import cdlam_memory_dataset as memory_data
from openpi.training import config as config_lib

OLD_NAME = "pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default"
NEW_NAME = "pi05_robomme_16x100_dreamdojo_lmv_causal_fixedlag3_stride4_anchor4x4_h50_default"
ENV_KEYS = (
    "ROBOMME_VIDEO_DATASET",
    "ROBOMME_MEMORY_CACHE",
    "ROBOMME_ANCHOR_CACHE",
    "ROBOMME_ASSETS_ROOT",
    "ROBOMME_ASSET_ID",
)


def _changed_paths(left, right, prefix=""):
    """递归比较全部字段，防止遗漏嵌套配置中的非预期改动。"""
    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
        return set().union(
            *(
                _changed_paths(getattr(left, field.name), getattr(right, field.name), f"{prefix}.{field.name}")
                for field in dataclasses.fields(left)
            )
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            set().union(
                *(
                    {f"{prefix}.{key}".lstrip(".")}
                    if key not in left or key not in right
                    else _changed_paths(left[key], right[key], f"{prefix}.{key}")
                    for key in left.keys() | right.keys()
                )
            )
            if left or right
            else set()
        )
    return set() if left == right else {prefix.lstrip(".")}


class PoolTest(unittest.TestCase):
    def test_known_block_means_and_dtype(self):
        tokens = jnp.arange(256, dtype=jnp.float32).reshape(1, 256, 1)
        expected = np.asarray([[25.5 + row * 64 + col * 4 for col in range(4)] for row in range(4)])
        np.testing.assert_array_equal(pi0.pool_anchor_tokens(tokens, 4)[0, :, 0], expected.reshape(16))
        for dtype in (jnp.float32, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                tokens = jnp.asarray(np.random.default_rng(7).normal(size=(2, 256, 7)), dtype=dtype)
                expected = nnx.avg_pool(tokens.reshape(2, 16, 16, 7), (4, 4), strides=(4, 4), padding="VALID").reshape(
                    2, 16, 7
                )
                actual = pi0.pool_anchor_tokens(tokens, 4)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(np.asarray(actual).tobytes(), np.asarray(expected).tobytes())
                self.assertIs(pi0.pool_anchor_tokens(tokens, 0), tokens)
                self.assertEqual(
                    np.asarray(jax.jit(pi0.pool_anchor_tokens, static_argnums=1)(tokens, 4)).tobytes(),
                    np.asarray(expected).tobytes(),
                )

    def test_invalid_grids_and_shapes(self):
        for grid in (-1, True, 2.5, "4", 3, 32):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                pi0.pool_anchor_tokens(jnp.zeros((1, 256, 2)), grid)
        for shape in ((1, 255, 2), (1, 0, 2), (256, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                pi0.pool_anchor_tokens(jnp.zeros(shape), 4)


class ConfigTest(unittest.TestCase):
    def test_frozen_config_only_changes_vision_training_semantics(self):
        base = config_lib.get_config(NEW_NAME)
        frozen = config_lib.get_config(
            "pi05_robomme_16x100_dreamdojo_lmv_causal_fixedlag3_stride4_anchor4x4_frozen_siglip_h50_default"
        )
        self.assertEqual(
            _changed_paths(base, frozen),
            {
                "name", "freeze_filter", "policy_metadata.frozen_siglip",
                "policy_metadata.frozen_siglip_parameter_dtype", "policy_metadata.frozen_siglip_ema_policy",
            },
        )
        parameter = nnx.Param(jnp.zeros((2,)))
        self.assertTrue(frozen.freeze_filter(("PaliGemma", "img", "embedding", "kernel"), parameter))
        for path in (
            ("PaliGemma", "llm", "kernel"), ("memory_projector", "kernel"),
            ("memory_segment_embed", "embedding"), ("action_out_proj", "kernel"),
            ("other", "PaliGemma", "img", "kernel"), ("PaliGemma", "img_adapter", "kernel"),
        ):
            self.assertFalse(frozen.freeze_filter(path, parameter), path)
            self.assertTrue(frozen.trainable_filter(path, parameter), path)

    def test_complete_config_difference_whitelist_and_defaults(self):
        with mock.patch.dict(
            os.environ, {key: value for key, value in os.environ.items() if key not in ENV_KEYS}, clear=True
        ):
            importlib.reload(config_lib)
            old, new = config_lib.get_config(OLD_NAME), config_lib.get_config(NEW_NAME)
            self.assertEqual(
                _changed_paths(old, new),
                {
                    "name",
                    "model.memory_stride",
                    "model.memory_anchor_pool_grid",
                    "data.repo_id",
                    "data.assets.asset_id",
                    "data.base_config.memory_cache_dir",
                    "data.base_config.demo_anchor_cache_dir",
                    "data.base_config.memory_random_drop_execution_tail_min",
                    "data.base_config.memory_random_drop_execution_tail_max",
                    "num_train_steps",
                    "save_steps",
                    "lr_schedule.warmup_steps",
                    "lr_schedule.peak_lr",
                    "lr_schedule.decay_steps",
                    "lr_schedule.decay_lr",
                    "policy_metadata.task",
                    "policy_metadata.memory_random_drop_execution_tail",
                    "policy_metadata.memory_stride",
                    "policy_metadata.memory_anchor_pool_grid",
                },
            )
            self.assertEqual(new.model.memory_stride, 4)
            self.assertEqual(new.model.memory_anchor_pool_grid, 4)
            self.assertEqual(new.data.base_config.memory_random_drop_execution_tail_min, 3)
            self.assertEqual(new.data.base_config.memory_random_drop_execution_tail_max, 3)
            self.assertEqual(new.data.repo_id, "external/robomme_16x100_lerobot_video_h264_v1")
            self.assertEqual(new.data.assets.asset_id, "lerobot/robomme_16x100")
            self.assertEqual(new.data.assets.assets_dir, "external/assets")
            self.assertEqual(new.policy_metadata["task"], "RoboMME_official_16_tasks_x_100_episodes")
            self.assertEqual(new.policy_metadata["memory_random_drop_execution_tail"], "fixed_integer_3_per_sample")
            self.assertEqual(new.policy_metadata["memory_stride"], 4)
            self.assertEqual(new.policy_metadata["memory_anchor_pool_grid"], 4)
            self.assertIn("16x100_", new.data.base_config.memory_cache_dir)
            self.assertIn("stride4", new.data.base_config.memory_cache_dir)
            self.assertIn("16x100_", new.data.base_config.demo_anchor_cache_dir)
            self.assertEqual(new.num_train_steps, 60000)
            self.assertEqual(new.save_steps, tuple(range(10000, 60001, 10000)))
            self.assertEqual(
                dataclasses.asdict(new.lr_schedule),
                {
                    "warmup_steps": 1000,
                    "peak_lr": 2.5e-5,
                    "decay_steps": 60000,
                    "decay_lr": 2.5e-6,
                },
            )
            schedule = new.lr_schedule.create()
            self.assertAlmostEqual(float(schedule(1000)), 2.5e-5, places=11)
            self.assertAlmostEqual(float(schedule(60000)), 2.5e-6, places=11)
            self.assertIsNone(new.training_contract_path)
            self.assertIsNone(new.checkpoint_upload_receipts_dir)
            self.assertIsNone(old.training_contract_path)
            self.assertIsNone(old.checkpoint_upload_receipts_dir)
            self.assertEqual(old.model.memory_anchor_pool_grid, 0)
            self.assertEqual(old.model.memory_stride, 1)
            self.assertEqual(old.data.base_config.memory_random_drop_execution_tail_min, 10)
            self.assertEqual(old.num_train_steps, 120000)
            self.assertEqual(old.data.assets.asset_id, "lerobot/robomme_16x1000")
        importlib.reload(config_lib)

    def test_environment_paths_and_contract_types(self):
        values = dict(
            zip(
                ENV_KEYS,
                (
                    "/scratch/hongze/fixture/datasets/robomme_data_lerobot",
                    "/scratch/hongze/fixture/artifacts/memory-s4",
                    "/scratch/hongze/fixture/artifacts/anchors",
                    "/scratch/hongze/fixture/assets",
                    "lerobot/robomme_16x100",
                ),
                strict=True,
            )
        )
        with mock.patch.dict(os.environ, values):
            importlib.reload(config_lib)
            cfg = config_lib.get_config(NEW_NAME)
            self.assertEqual(cfg.data.repo_id, values[ENV_KEYS[0]])
            self.assertEqual(cfg.data.base_config.memory_cache_dir, values[ENV_KEYS[1]])
            self.assertEqual(cfg.data.base_config.demo_anchor_cache_dir, values[ENV_KEYS[2]])
            self.assertEqual(cfg.data.assets.assets_dir, values[ENV_KEYS[3]])
            self.assertEqual(cfg.data.assets.asset_id, values[ENV_KEYS[4]])
            cfg = dataclasses.replace(
                cfg, training_contract_path=Path("contract.json"), checkpoint_upload_receipts_dir=Path("receipts")
            )
            self.assertIsInstance(cfg.training_contract_path, Path)
            self.assertIsInstance(cfg.checkpoint_upload_receipts_dir, Path)
        importlib.reload(config_lib)

    def test_incompatible_anchor_modes(self):
        model = config_lib.get_config(NEW_NAME).model
        for overrides in (
            {"memory_anchor_pool_grid": -1},
            {"memory_anchor_pool_grid": True},
            {"memory_anchor_pool_grid": 3},
            {"memory_anchor_pool_grid": 32},
            {"memory_demo_anchor": False},
            {"memory_execution_anchor": False},
            {"memory_episode_anchor": True},
            {"memory_demo_anchor_feature_pool": True},
            {"memory_anchor_only": True, "memory_horizon": 0, "memory_demo_lam_boundary_token": False},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                dataclasses.replace(model, **overrides)

    def test_full_model_parameter_tree_has_no_new_parameters(self):
        def shapes(cfg):
            return jax.eval_shape(lambda: nnx.state(cfg.create(jax.random.key(0)), nnx.Param))

        old = shapes(config_lib.get_config(OLD_NAME).model)
        new = shapes(config_lib.get_config(NEW_NAME).model)
        self.assertEqual(jax.tree.structure(old), jax.tree.structure(new))
        old_shapes = [(leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(old)]
        new_shapes = [(leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(new)]
        self.assertEqual(old_shapes, new_shapes)
        self.assertGreater(sum(np.prod(shape) for shape, _ in new_shapes), 1_000_000_000)


class _CaptureLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, tokens, *, method=None, **kwargs):
        if method == "embed":
            return jnp.repeat(tokens[..., None], 4, axis=-1).astype(jnp.float32)
        self.calls.append((tokens, kwargs))
        return [None if token is None else jnp.zeros_like(token) for token in tokens], ()


class _EmbedHarness:
    """仅替换昂贵编码器，执行真实前缀、记忆、训练及 KV 采样控制流。"""

    _embed_prefix_with_anchor_length = pi0.Pi0._embed_prefix_with_anchor_length
    embed_prefix = pi0.Pi0.embed_prefix
    embed_memory = pi0.Pi0.embed_memory
    compute_loss = pi0.Pi0.compute_loss
    sample_actions = pi0.Pi0.sample_actions

    def __init__(self, cfg, image_dtype=jnp.float32):
        self.config = cfg
        self.image_dtype = image_dtype
        self.pi05 = True
        self.use_memory = self.use_lam_memory = True
        self.use_memory_expert = self.use_future_expert = False
        self.action_horizon, self.action_dim = cfg.action_horizon, cfg.action_dim
        self.PaliGemma = SimpleNamespace(img=self.img, llm=_CaptureLLM())
        self.memory_latent_mean = SimpleNamespace(value=jnp.zeros(64))
        self.memory_latent_std = SimpleNamespace(value=jnp.ones(64))
        self.memory_projector_in = lambda values: values[..., :4]
        self.memory_projector_out = lambda values: values
        self.memory_projector_norm = lambda values: values
        self.action_out_proj = lambda values: jnp.zeros((*values.shape[:-1], self.action_dim))

    def img(self, images, *, train):
        return images.reshape(images.shape[0], 256, 4).astype(self.image_dtype), None

    def memory_segment_embed(self, ids):
        return jnp.repeat(ids[..., None], 4, axis=-1).astype(jnp.float32) * 1000

    def embed_suffix(self, obs, x_t, time):
        batch = obs.state.shape[0]
        return (
            jnp.zeros((batch, self.action_horizon, 4)),
            jnp.ones((batch, self.action_horizon), dtype=jnp.bool_),
            jnp.arange(self.action_horizon) == 0,
            None,
        )


def _observation():
    segments = jnp.asarray(
        [
            [0, 0, 1, 1, 2, 3, 3],
            [0, 0, 0, 1, 2, 3, 3],
            [0, 1, 1, 1, 3, 3, 3],
            [0, 0, 0, 0, 0, 3, 3],
        ],
        dtype=jnp.int32,
    )
    image = jnp.arange(4 * 256 * 4, dtype=jnp.float32).reshape(4, 16, 16, 4) / 32
    mask = jnp.ones(4, dtype=jnp.bool_)
    return model_lib.Observation(
        images={"base_0_rgb": image, "left_wrist_0_rgb": image + 100},
        image_masks={"base_0_rgb": mask, "left_wrist_0_rgb": mask},
        state=jnp.zeros((4, 32)),
        tokenized_prompt=jnp.ones((4, 3), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=jnp.bool_),
        memory_latents=jnp.ones((4, 7, 64)),
        memory_mask=segments > 0,
        memory_segment_ids=segments,
        memory_demo_start_image=image + 200,
        memory_demo_start_mask=jnp.asarray([1, 1, 1, 0], dtype=jnp.bool_),
        memory_execution_start_image=image + 300,
        memory_execution_start_mask=mask,
    )


class AnchorPlacementTest(unittest.TestCase):
    def test_both_anchors_pool_in_siglip_dtype_before_type_embedding(self):
        obs = _observation()
        images = jnp.asarray(np.random.default_rng(17).normal(size=(4, 16, 16, 4)), dtype=jnp.float32)
        obs = obs.replace(memory_demo_start_image=images, memory_execution_start_image=images)
        harness = _EmbedHarness(config_lib.get_config(NEW_NAME).model, image_dtype=jnp.bfloat16)
        prefix, _, _, _, _, _ = harness._embed_prefix_with_anchor_length(obs)
        memory, _, _ = harness.embed_memory(obs)
        pooled = nnx.avg_pool(images.astype(jnp.bfloat16), (4, 4), strides=(4, 4), padding="VALID")
        pooled = pooled.reshape(4, 16, 4)
        expected_demo = (pooled + jnp.asarray(4000, dtype=jnp.bfloat16)).astype(prefix.dtype)
        np.testing.assert_array_equal(prefix[:, -16:], expected_demo)
        expected_execution = pooled.astype(memory.dtype) + 5000
        for batch, segments in enumerate(np.asarray(obs.memory_segment_ids)):
            start = int(np.count_nonzero((segments == 0) | (segments == 1))) + 1
            np.testing.assert_array_equal(memory[batch, start : start + 16], expected_execution[batch])

    def test_dual_anchor_lengths_values_insertion_and_causality(self):
        obs = _observation()
        for grid, length in ((0, 256), (4, 16)):
            with self.subTest(grid=grid):
                harness = _EmbedHarness(
                    dataclasses.replace(config_lib.get_config(NEW_NAME).model, memory_anchor_pool_grid=grid)
                )
                prefix, prefix_mask, _, anchor_len, lang_start, lang_len = harness._embed_prefix_with_anchor_length(obs)
                memory, memory_mask, ar = harness.embed_memory(obs)
                self.assertEqual(anchor_len, length)
                self.assertEqual(prefix.shape, (4, 512 + 3 + length, 4))
                self.assertEqual(memory.shape, (4, 7 + 1 + length, 4))
                np.testing.assert_array_equal(prefix[:, :256], obs.images["base_0_rgb"].reshape(4, 256, 4))
                expected_demo = pi0.pool_anchor_tokens(obs.memory_demo_start_image.reshape(4, 256, 4), grid) + 4000
                np.testing.assert_array_equal(prefix[:, -length:], expected_demo)
                expected_exec = pi0.pool_anchor_tokens(obs.memory_execution_start_image.reshape(4, 256, 4), grid) + 5000
                ordered, valid, causal = pi0.make_language_memory_vision_prefix(
                    prefix,
                    prefix_mask,
                    memory,
                    memory_mask,
                    ar,
                    language_start=lang_start,
                    language_length=lang_len,
                    memory_anchor_length=anchor_len,
                )
                attn = np.asarray(pi0.make_attn_mask(valid, causal))
                for b, segments in enumerate(np.asarray(obs.memory_segment_ids)):
                    start = int(np.count_nonzero((segments == 0) | (segments == 1))) + 1
                    np.testing.assert_array_equal(memory[b, start : start + length], expected_exec[b])
                    np.testing.assert_array_equal(ar[b, start : start + length], [True] + [False] * (length - 1))
                    offset = lang_len + length
                    exe = offset + start
                    self.assertTrue(attn[b, exe : exe + length, exe : exe + length].all())
                    self.assertFalse(attn[b, :exe, exe : exe + length].any())
                    self.assertTrue(np.asarray(memory_mask[b, start : start + length]).all())
                self.assertEqual(ordered.shape[1], prefix.shape[1] + memory.shape[1])

    def test_training_and_kv_sampling_use_identical_positions(self):
        obs = _observation()
        harness = _EmbedHarness(config_lib.get_config(NEW_NAME).model)
        actions = jnp.zeros((4, harness.action_horizon, harness.action_dim))
        with mock.patch.object(
            model_lib, "preprocess_observation", side_effect=lambda _rng, observation, **_: observation
        ):
            loss = harness.compute_loss(jax.random.key(1), obs, actions)
            self.assertTrue(np.isfinite(np.asarray(loss)).all())
            train_tokens, train_call = harness.PaliGemma.llm.calls[-1]
            harness.PaliGemma.llm.calls.clear()
            with jax.disable_jit():
                sampled = harness.sample_actions(jax.random.key(1), obs, num_steps=1, noise=actions)
            self.assertTrue(np.isfinite(np.asarray(sampled)).all())
        prefix_tokens, prefix_call = harness.PaliGemma.llm.calls[0]
        suffix_tokens, suffix_call = harness.PaliGemma.llm.calls[-1]
        length = prefix_tokens[0].shape[1]
        np.testing.assert_array_equal(train_tokens[0], prefix_tokens[0])
        np.testing.assert_array_equal(train_call["positions"][:, :length], prefix_call["positions"])
        np.testing.assert_array_equal(train_call["positions"][:, length:], suffix_call["positions"])
        np.testing.assert_array_equal(train_call["mask"][:, :length, :length], prefix_call["mask"])
        np.testing.assert_array_equal(train_call["mask"][:, length:, :], suffix_call["mask"])
        self.assertIsNone(suffix_tokens[0])


class StrideCacheTest(unittest.TestCase):
    def test_real_cache_sampler_budget_endpoints_and_fixed_lag(self):
        # 五集分别覆盖无 demo 以及 execution 边界的四种模 4 余数；尾部均不足一个完整 stride。
        boundaries = [0, 8, 9, 10, 11]
        length = 31
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(json.dumps({"total_frames": length * 5, "total_episodes": 5}))
            (root / "meta/episodes.jsonl").write_text(
                "\n".join(json.dumps({"episode_index": index, "length": length}) for index in range(5))
            )
            info_sha = hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
            episodes_sha = hashlib.sha256((root / "meta/episodes.jsonl").read_bytes()).hexdigest()
            frames = np.tile(np.arange(length), 5)
            episodes = np.repeat(np.arange(5), length)
            valid = (frames > 0) & (frames % 4 == 0)
            latents = np.zeros((len(frames), 2), dtype=np.float32)
            latents[valid, 0] = frames[valid] - 4
            latents[valid, 1] = frames[valid]
            np.save(root / "latents.npy", latents)
            np.savez(
                root / "frame_index.npz",
                global_index=np.arange(len(frames)),
                episode_index=episodes,
                frame_index=frames,
                valid=valid,
            )
            manifest = {
                "schema_version": 2,
                "cache_schema_version": 1,
                "status": "complete",
                "missing_episodes": [],
                "latent_dim": 2,
                "transition_frame_stride": 4,
                "alignment": "row_{k*4}=z_mu(frame_{(k-1)*4},frame_{k*4}), k>=1; all other rows invalid_zero",
                "dataset_root": str(root),
                "dataset_info_sha256": info_sha,
                "dataset_episodes_sha256": episodes_sha,
                "dataset_fingerprint": hashlib.sha256(f"{info_sha}:{episodes_sha}".encode()).hexdigest(),
                "expected_episodes": 5,
                "cached_episodes": 5,
                "declared_total_frames": len(frames),
                "cached_total_frames": len(frames),
                "cached_valid_transitions": int(valid.sum()),
                "global_latents_sha256": hashlib.sha256((root / "latents.npy").read_bytes()).hexdigest(),
                "global_frame_index_sha256": hashlib.sha256((root / "frame_index.npz").read_bytes()).hexdigest(),
                "latent_normalization": {
                    "definition": "per_dimension_valid_transition_population",
                    "ddof": 0,
                    "epsilon": 1e-6,
                    "count": int(valid.sum()),
                    "mean": latents[valid].mean(axis=0).tolist(),
                    "std": latents[valid].std(axis=0).tolist(),
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            dataset = [
                {"index": np.int64(index), "exec_start_idx": np.int32(boundaries[episode])}
                for index, episode in enumerate(episodes)
            ]
            wrapper = memory_data.CDLAMMemoryDataset(
                dataset,
                root,
                memory_horizon=32,
                latent_dim=2,
                memory_stride=4,
                random_drop_execution_tail_min=3,
                random_drop_execution_tail_max=3,
            )
            np.testing.assert_array_equal(wrapper.memory_lengths, np.maximum(frames // 4, 1))
            seen_execution_counts = set()
            for index, frame in enumerate(frames):
                sample = wrapper[index]
                boundary = boundaries[episodes[index]]
                endpoints = np.arange(4, frame + 1, 4)
                self.assertEqual(len(endpoints), frame // 4)
                # 边界只在端点恰好命中 exec_start 时存在；跨边界 stride 的端点属于 execution。
                segments = np.where(endpoints < boundary, 1, np.where((endpoints == boundary) & (boundary > 0), 2, 3))
                execution = np.flatnonzero(segments == 3)
                seen_execution_counts.add(len(execution))
                keep = np.ones(len(endpoints), dtype=np.bool_)
                if frame >= boundary and len(execution):
                    keep[execution[-3:]] = False
                expected_endpoints = endpoints[keep]
                self.assertEqual(int(sample["memory_mask"].sum()), len(expected_endpoints))
                self.assertEqual(len(sample["memory_mask"]), max(len(expected_endpoints), 1))
                self.assertLessEqual(len(sample["memory_mask"]), wrapper.memory_lengths[index])
                np.testing.assert_array_equal(sample["memory_segment_ids"][sample["memory_mask"]], segments[keep])
                np.testing.assert_array_equal(sample["memory_latents"][sample["memory_mask"], 1], expected_endpoints)
                np.testing.assert_array_equal(
                    sample["memory_latents"][sample["memory_mask"], 0], expected_endpoints - 4
                )
                again = wrapper[index]
                for key in ("memory_latents", "memory_mask", "memory_segment_ids"):
                    np.testing.assert_array_equal(sample[key], again[key])
                if not len(expected_endpoints):
                    np.testing.assert_array_equal(sample["memory_latents"], np.zeros((1, 2)))
                    np.testing.assert_array_equal(sample["memory_segment_ids"], [0])
            self.assertTrue(set(range(5)).issubset(seen_execution_counts))


if __name__ == "__main__":
    unittest.main()
