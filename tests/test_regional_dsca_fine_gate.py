import unittest
import tempfile
from pathlib import Path

import torch
import yaml

from basicsr.archs import build_network
from basicsr.archs.SDAnet_arch import DSCA
from basicsr.archs.SDAnet_sads_arch import RegionalDSCA
from basicsr.models.base_model import BaseModel
from basicsr.utils.misc import make_exp_dirs


REPO_ROOT = Path(__file__).resolve().parents[1]
REGIONAL_CHECKPOINT = (
    REPO_ROOT
    / "experiments/SDANetRegionalContinue66K_spatial_split/models/net_g_66000.pth"
)


def legacy_regional_attention(module, x):
    batch, _, height, width = x.shape
    q = module.q_dw(module.q_proj(x))
    k = module.k_dw(module.k_proj(x))
    v = module.v_dw(module.v_proj(x))
    q = q.reshape(batch, module.num_heads, module.head_dim, height * width)
    k = k.reshape(batch, module.num_heads, module.head_dim, height * width)
    v = v.reshape(batch, module.num_heads, module.head_dim, height * width)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    attn = (q @ k.transpose(-2, -1)) * module.temperature
    soft_prob = torch.nn.functional.softmax(attn, dim=-1)
    gate = module.k_mlp(module.k_dconv(x))
    regions = (
        (0, height // 2, 0, width // 2),
        (0, height // 2, width // 2, width),
        (height // 2, height, 0, width // 2),
        (height // 2, height, width // 2, width),
    )
    outputs = []
    for y0, y1, x0, x1 in regions:
        factor = gate[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3))
        factor = factor.clamp(module.eps, 1.0 - module.eps)
        k_float = 1.0 + (module.head_dim - 1.0) * factor
        k_count = torch.floor(k_float).long().clamp(1, module.head_dim)
        k_ratio = (k_float / module.head_dim).view(batch, 1, 1, 1)
        hard_mask = module._build_hard_mask(attn, k_count)
        threshold = module._continuous_threshold(attn, k_ratio)
        soft_mask = torch.sigmoid(
            (attn - threshold) / max(module.ste_mask_tau, module.eps)
        )
        mask = hard_mask + soft_mask - soft_mask.detach()
        prob = soft_prob * mask
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(module.eps)
        indices = (
            torch.arange(y0, y1, device=x.device)[:, None] * width
            + torch.arange(x0, x1, device=x.device)[None, :]
        ).reshape(-1)
        values = v.index_select(-1, indices)
        out = prob @ values
        out = out * module.head_weights.view(1, module.num_heads, 1, 1)
        out = out.reshape(batch, -1, y1 - y0, x1 - x0)
        outputs.append(module.project_out(out))
    top = torch.cat(outputs[:2], dim=3)
    bottom = torch.cat(outputs[2:], dim=3)
    return torch.cat([top, bottom], dim=2)


class RegionalDSCAFineGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def make_module(self, fine_gate_enabled=False):
        return RegionalDSCA(
            dim=8,
            num_heads=2,
            reduction=4,
            ste_mask_tau=0.15,
            fine_gate_enabled=fine_gate_enabled,
        ).eval()

    def test_disabled_state_dict_and_output_match_legacy(self):
        module = self.make_module(fine_gate_enabled=False)
        dsca = DSCA(dim=8, num_heads=2, reduction=4, ste_mask_tau=0.15)
        self.assertEqual(set(module.state_dict()), set(dsca.state_dict()))
        x = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            actual = module._regional_attention(x)
            expected = legacy_regional_attention(module, x)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_initialized_fine_gate_matches_coarse_output(self):
        legacy = self.make_module(fine_gate_enabled=False)
        treatment = self.make_module(fine_gate_enabled=True)
        result = treatment.load_state_dict(legacy.state_dict(), strict=False)
        expected_missing = {
            "fine_gate.weight",
            "fine_gate.bias",
        }
        self.assertEqual(set(result.missing_keys), expected_missing)
        self.assertEqual(result.unexpected_keys, [])
        x = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            expected = legacy._regional_attention(x)
            actual = treatment._regional_attention(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_fine_gate_construction_preserves_global_rng_stream(self):
        torch.manual_seed(43)
        control = self.make_module(fine_gate_enabled=False)
        control_rng_state = torch.random.get_rng_state().clone()

        torch.manual_seed(43)
        treatment = self.make_module(fine_gate_enabled=True)
        treatment_rng_state = torch.random.get_rng_state().clone()

        self.assertTrue(torch.equal(control_rng_state, treatment_rng_state))
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, treatment.state_dict()[name]), name)

    def test_projection_receives_finite_nonzero_gradient(self):
        module = self.make_module(fine_gate_enabled=True).train()
        x = torch.randn(1, 8, 8, 8)
        output = module._regional_attention(x)
        coefficients = torch.linspace(0.1, 2.0, output.numel()).reshape_as(output)
        loss = (output * coefficients).sum()
        loss.backward()
        for name, parameter in module.fine_gate.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_child_factor_masks_match_explicit_reference(self):
        module = self.make_module(fine_gate_enabled=True)
        module.fine_gate.weight.data.normal_(mean=0.0, std=0.1)
        module.fine_gate.bias.data.fill_(0.03)
        x = torch.randn(1, 8, 8, 8)
        captured = []
        original_probabilities = module._fine_gate_probabilities

        def capture_probabilities(attn, soft_prob, factor, batch):
            result = original_probabilities(attn, soft_prob, factor, batch)
            captured.append(tuple(value.detach().clone() for value in result))
            return result

        module._fine_gate_probabilities = capture_probabilities
        with torch.no_grad():
            module._regional_attention(x)
        self.assertEqual(len(captured), 16)

        gate = module.k_mlp(module.k_dconv(x))
        quadrant_bounds = (
            (0, 4, 0, 4), (0, 4, 4, 8),
            (4, 8, 0, 4), (4, 8, 4, 8),
        )
        attn_features = module.q_dw(module.q_proj(x))
        key_features = module.k_dw(module.k_proj(x))
        attn_features = torch.nn.functional.normalize(
            attn_features.reshape(1, 2, 4, 64), dim=-1
        )
        key_features = torch.nn.functional.normalize(
            key_features.reshape(1, 2, 4, 64), dim=-1
        )
        attn = (attn_features @ key_features.transpose(-2, -1)) * module.temperature
        soft_prob = torch.nn.functional.softmax(attn, dim=-1)

        call_index = 0
        for y0, y1, x0, x1 in quadrant_bounds:
            parent = gate[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3))
            parent = parent.clamp(module.eps, 1 - module.eps)
            for cell_row in range(2):
                cy0 = y0 + cell_row * ((y1 - y0) // 2)
                cy1 = y0 + (cell_row + 1) * ((y1 - y0) // 2)
                for cell_col in range(2):
                    cx0 = x0 + cell_col * ((x1 - x0) // 2)
                    cx1 = x0 + (cell_col + 1) * ((x1 - x0) // 2)
                    pooled = gate[:, :, cy0:cy1, cx0:cx1].mean(dim=(2, 3))
                    factor = (parent + module.fine_gate(pooled).squeeze(-1)).clamp(
                        module.eps, 1 - module.eps
                    )
                    k_float = 1 + (module.head_dim - 1) * factor
                    expected_count = torch.floor(k_float).long().clamp(
                        1, module.head_dim
                    )
                    expected_ratio = (k_float / module.head_dim).view(1, 1, 1, 1)
                    actual_count, actual_ratio, hard, threshold, soft, ste, prob = captured[
                        call_index
                    ]
                    self.assertTrue(torch.equal(actual_count, expected_count))
                    torch.testing.assert_close(actual_ratio, expected_ratio)

                    max_k = int(expected_count.max())
                    top_indices = torch.topk(attn, k=max_k, dim=-1).indices
                    ranks = torch.arange(max_k).view(1, 1, 1, 1, -1)
                    valid = ranks < expected_count.view(1, 1, 1, 1, 1)
                    fill = torch.ones_like(
                        top_indices.unsqueeze(2), dtype=attn.dtype
                    ).masked_fill(~valid, 0.0)
                    expected_hard = torch.zeros_like(attn.unsqueeze(2)).scatter_(
                        -1, top_indices.unsqueeze(2), fill
                    ).squeeze(2)
                    self.assertTrue(torch.equal(hard, expected_hard))

                    expected_threshold = attn.mean(dim=-1, keepdim=True) + (
                        1 - expected_ratio
                    ) * attn.std(dim=-1, keepdim=True, unbiased=False).clamp_min(
                        module.eps
                    )
                    torch.testing.assert_close(threshold, expected_threshold)
                    expected_soft = torch.sigmoid(
                        (attn - expected_threshold)
                        / max(module.ste_mask_tau, module.eps)
                    )
                    expected_ste = expected_hard + expected_soft - expected_soft.detach()
                    torch.testing.assert_close(soft, expected_soft)
                    torch.testing.assert_close(ste, expected_ste)
                    expected_prob = soft_prob * expected_ste
                    expected_prob = expected_prob / expected_prob.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(module.eps)
                    torch.testing.assert_close(prob, expected_prob)
                    call_index += 1

    def test_child_partitions_cover_each_quadrant_once(self):
        height, width = 16, 24
        coverage = torch.zeros(height, width, dtype=torch.int32)
        for y0, y1, x0, x1 in (
            (0, 8, 0, 12), (0, 8, 12, 24),
            (8, 16, 0, 12), (8, 16, 12, 24),
        ):
            for cy0, cy1, cx0, cx1 in RegionalDSCA._fine_cell_bounds(
                y0, y1, x0, x1
            ):
                coverage[cy0:cy1, cx0:cx1] += 1
        self.assertTrue(torch.all(coverage == 1))
        with self.assertRaises(ValueError):
            RegionalDSCA._fine_cell_bounds(0, 3, 0, 4)

    def test_actual_legacy_checkpoint_loads_with_only_fine_gate_keys_missing(self):
        self.assertTrue(REGIONAL_CHECKPOINT.is_file())
        config_path = REPO_ROOT / "options/train/HSI/SDANetRegionalFineGateTreatment5KChux4.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = build_network(config["network_g"])
        expected_missing = {
            name for name in model.state_dict()
            if name.endswith("fused_attn.fine_gate.weight")
            or name.endswith("fused_attn.fine_gate.bias")
        }
        self.assertEqual(len(expected_missing), 16)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = torch.load(
                REGIONAL_CHECKPOINT, map_location="cpu", weights_only=False
            )
            checkpoint_path = Path(temp_dir) / "legacy_ema.pth"
            torch.save({"params_ema": checkpoint["params_ema"]}, checkpoint_path)
            loader = BaseModel.__new__(BaseModel)
            loader.load_network(
                model,
                str(checkpoint_path),
                strict=True,
                param_key="params_ema",
                allowed_missing_keys=expected_missing,
            )

            with self.assertRaisesRegex(RuntimeError, "Checkpoint key mismatch"):
                loader.load_network(
                    model,
                    str(checkpoint_path),
                    strict=True,
                    param_key="params_ema",
                    allowed_missing_keys=set(),
                )

    def test_make_exp_dirs_skips_list_valued_checkpoint_allowlist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_root = Path(temp_dir) / "experiment"
            make_exp_dirs({
                "is_train": True,
                "path": {
                    "experiments_root": str(experiment_root),
                    "allowed_missing_keys": [
                        "body.0.body.0.fused_attn.fine_gate.weight",
                        "body.0.body.0.fused_attn.fine_gate.bias",
                    ],
                },
            })
            self.assertTrue(experiment_root.is_dir())
            self.assertFalse(
                (Path(temp_dir) / "body.0.body.0.fused_attn.fine_gate.weight").exists()
            )


if __name__ == "__main__":
    unittest.main()