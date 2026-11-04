"""冻结视觉、EMA 与首尾审计的 CPU 定向测试，不读取 checkpoint。"""

import dataclasses
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.shared import ltmpi_contract as contract
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import utils as training_utils

_SPEC = importlib.util.spec_from_file_location("ltmpi_frozen_train", Path(__file__).parents[1] / "scripts/train.py")
train = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = train
_SPEC.loader.exec_module(train)


class TinyModel(nnx.Module):
    def __init__(self):
        self.PaliGemma = nnx.Dict(
            img=nnx.Dict(kernel=nnx.Param(jnp.asarray([1.015625, -0.3359375], dtype=jnp.bfloat16))),
            llm=nnx.Dict(kernel=nnx.Param(jnp.asarray([2.0, -1.0], dtype=jnp.float32))),
        )
        self.action_out = nnx.Param(jnp.asarray([0.5], dtype=jnp.float32))


class FrozenStateTest(unittest.TestCase):
    def setUp(self):
        self.config = dataclasses.replace(
            config_lib.TrainConfig(name=contract.FROZEN_S4_CONFIG_NAME),
            freeze_filter=nnx_utils.PathRegex(r"PaliGemma/img/.*"),
            num_train_steps=2,
            ema_decay=0.999,
        )
        graphdef, params = nnx.split(TinyModel())
        tx = optax.adam(0.1)
        self.state = training_utils.TrainState(
            step=jnp.asarray(0), params=params, model_def=graphdef, tx=tx,
            opt_state=tx.init(params.filter(self.config.trainable_filter)),
            ema_decay=self.config.ema_decay, ema_params=params,
        )
        self.grads = jax.tree.map(jnp.ones_like, params.filter(self.config.trainable_filter))

    def test_frozen_ema_and_live_state_preserved_under_jit(self):
        initial = train._vision_audit_summary(self.state.params)
        update = jax.jit(lambda state, grads: train._apply_gradients(self.config, state, grads))
        state = self.state
        for _ in range(2):
            state, info = update(state, self.grads)
        self.assertEqual(train._vision_audit_summary(state.params), initial)
        self.assertEqual(train._vision_audit_summary(state.ema_params), initial)
        self.assertEqual(int(state.step), 2)
        self.assertTrue(np.isfinite(info["grad_norm"]))
        self.assertFalse(np.array_equal(state.params.action_out.value, self.state.params.action_out.value))
        once = update(self.state, self.grads)[0]
        expected = self.state.ema_decay * self.state.ema_params.action_out.value + (
            1 - self.state.ema_decay
        ) * once.params.action_out.value
        np.testing.assert_array_equal(once.ema_params.action_out.value, expected)

    def test_old_configuration_keeps_full_tree_ema_arithmetic(self):
        old_config = dataclasses.replace(self.config, name=contract.S4_CONFIG_NAME)
        updated, _ = train._apply_gradients(old_config, self.state, self.grads)
        expected = jax.tree.map(
            lambda old, new: self.state.ema_decay * old + (1 - self.state.ema_decay) * new,
            self.state.ema_params, updated.params,
        )
        for actual, wanted in zip(jax.tree.leaves(updated.ema_params), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(actual, wanted)

    def test_audit_reads_live_state_and_checks_optimizer_counts(self):
        with tempfile.TemporaryDirectory(prefix="frozen-audit-", dir="/scratch/hongze") as directory:
            path = str(Path(directory) / "audit.json")
            audit = train._FrozenSiglipAudit(self.config, self.state, path, {"contract_sha256": "fixture"})
            initial = json.loads(Path(path).read_text())
            self.assertEqual(initial["status"], "INITIALIZED")
            self.assertEqual(initial["initial"]["live_vision"]["dtypes"], ["bfloat16"])
            state = self.state
            for _ in range(2):
                state, info = train._apply_gradients(self.config, state, self.grads)
                audit.observe({**info, "loss": jnp.asarray(1.0)})
            audit.finish(state)
            report = json.loads(Path(path).read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["metrics_count"], 2)
            self.assertEqual(report["final"]["optimizer_vision_numeric_paths"], [])
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(all(count == 2 for count in report["final"]["optimizer_counts"].values()))
            changed = state.params.map(
                lambda path, leaf: leaf.replace(leaf.value + jnp.asarray(1, leaf.value.dtype))
                if path[:2] == ("PaliGemma", "img") else leaf
            )
            with self.assertRaisesRegex(RuntimeError, "审计失败"):
                audit.finish(dataclasses.replace(state, params=changed))
            self.assertEqual(json.loads(Path(path).read_text())["status"], "FAIL")

    def test_optimizer_visual_moments_are_detected_without_copying_them(self):
        wrong = dataclasses.replace(self.state, opt_state=self.state.tx.init(self.state.params))
        snapshot = train._frozen_audit_snapshot(wrong, (("action_out",),))
        self.assertEqual(len(snapshot["optimizer_vision_numeric_paths"]), 2)

    def test_audit_rejects_nonfinite_loss(self):
        with tempfile.TemporaryDirectory(prefix="frozen-audit-", dir="/scratch/hongze") as directory:
            path = str(Path(directory) / "audit.json")
            audit = train._FrozenSiglipAudit(self.config, self.state, path, None)
            with self.assertRaisesRegex(RuntimeError, "非有限"):
                audit.observe({"loss": jnp.asarray(float("nan")), "grad_norm": jnp.asarray(1.0)})
            self.assertEqual(json.loads(Path(path).read_text())["status"], "FAIL")


class FrozenContractTest(unittest.TestCase):
    def test_regex_pattern_and_flags_have_stable_distinct_identity(self):
        pattern = re.compile(r"PaliGemma/img/.*")
        self.assertEqual(contract.normalize(pattern), {"pattern": pattern.pattern, "flags": pattern.flags})
        first = nnx_utils.PathRegex(pattern)
        self.assertEqual(contract.canonical_sha256(first), contract.canonical_sha256(nnx_utils.PathRegex(pattern.pattern)))
        self.assertNotEqual(
            contract.canonical_sha256(first),
            contract.canonical_sha256(nnx_utils.PathRegex(re.compile(pattern.pattern, re.IGNORECASE))),
        )
        self.assertNotEqual(
            contract.canonical_sha256(first), contract.canonical_sha256(nnx_utils.PathRegex(r"PaliGemma/.*"))
        )

    def test_both_s4_names_require_a_training_contract(self):
        self.assertEqual(contract.S4_CONFIG_NAMES, (contract.S4_CONFIG_NAME, contract.FROZEN_S4_CONFIG_NAME))
        for name in contract.S4_CONFIG_NAMES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "training_contract_path"):
                contract.prepare_training_contract(config_lib.TrainConfig(name=name))

    def test_registered_frozen_configuration_has_distinct_stable_semantics(self):
        old = dataclasses.replace(config_lib.get_config(contract.S4_CONFIG_NAME), exp_name="fixture")
        frozen = dataclasses.replace(config_lib.get_config(contract.FROZEN_S4_CONFIG_NAME), exp_name="fixture")
        normalized = contract.normalize(frozen)
        self.assertEqual(
            normalized["freeze_filter"],
            {"pattern": {"pattern": r"PaliGemma/img/.*", "flags": int(re.UNICODE)}, "sep": "/"},
        )
        self.assertTrue(normalized["policy_metadata"]["frozen_siglip"])
        self.assertEqual(normalized["policy_metadata"]["frozen_siglip_ema_policy"], "copy_frozen")
        self.assertEqual(normalized["policy_metadata"]["frozen_siglip_parameter_dtype"], "bfloat16")
        self.assertNotEqual(contract.canonical_sha256(old), contract.canonical_sha256(frozen))
        self.assertEqual(contract.canonical_sha256(frozen), contract.canonical_sha256(normalized))


if __name__ == "__main__":
    unittest.main()
