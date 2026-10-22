
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.SDAnet_arch import (
    DSCA,
    FEFeedForward,
    SDABlock,
    SDAStage,
    HSI_Upsampler,
    SDANET,
)


class RegionalDSCA(DSCA):
    """DSCA with independently estimated sparsity in a 2x2 spatial grid."""


    def _regional_attention(self, x):
        b, _, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"Expected even dimensions, got {h}x{w}")

        # Preserve the baseline's global Q/K/V attention computation.
        q = self.q_dw(self.q_proj(x))
        k = self.k_dw(self.k_proj(x))
        v = self.v_dw(self.v_proj(x))

        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        soft_prob = F.softmax(attn, dim=-1)

        gate = self.k_mlp(self.k_dconv(x))
        regions = [
            (0, h // 2, 0, w // 2),
            (0, h // 2, w // 2, w),
            (h // 2, h, 0, w // 2),
            (h // 2, h, w // 2, w),
        ]

        output_quadrants = []

        for y0, y1, x0, x1 in regions:
            # One gate factor per sample and region.
            factor = gate[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3))
            factor = factor.clamp(self.eps, 1.0 - self.eps)

            k_float = 1.0 + (self.head_dim - 1.0) * factor
            k_count = torch.floor(k_float).long().clamp(
                min=1, max=self.head_dim
            )

            k_ratio = (k_float / self.head_dim).view(b, 1, 1, 1)
            hard_mask = self._build_hard_mask(attn, k_count)
            threshold = self._continuous_threshold(attn, k_ratio)
            soft_mask = torch.sigmoid(
                (attn - threshold) / max(self.ste_mask_tau, self.eps)
            )
            mask = hard_mask + soft_mask - soft_mask.detach()

            prob = soft_prob * mask
            prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            # Apply the globally computed attention to this region's values.

            indices = (
                torch.arange(y0, y1, device=x.device)[:, None] * w
                + torch.arange(x0, x1, device=x.device)[None, :]
            ).reshape(-1)
            vr = v.index_select(-1, indices)

            out = prob @ vr
            out = out * self.head_weights.view(1, self.num_heads, 1, 1)
            out = out.reshape(b, -1, y1 - y0, x1 - x0)
            output_quadrants.append(self.project_out(out))

        top = torch.cat(output_quadrants[:2], dim=3)
        bottom = torch.cat(output_quadrants[2:], dim=3)
        return torch.cat([top, bottom], dim=2)

    def forward(self, x):
        global_out = self._regional_attention(x)

        local_out = self.local_dw(x)
        local_out = self.local_act(local_out)
        local_out = self._fgse(local_out)
        local_out = self.local_pw(local_out)

        return self.fusion(torch.cat([local_out, global_out], dim=1))


class SDABlockSADS(SDABlock):
    def __init__(
        self, n_feats, num_heads, expansion_ratio=1.0, drop_path=0.1,
        reduction=4, dropout=0.2, ste_mask_tau=0.15
    ):
        super().__init__(
            n_feats=n_feats,
            num_heads=num_heads,
            expansion_ratio=expansion_ratio,
            drop_path=drop_path,
            reduction=reduction,
            dropout=dropout,
            ste_mask_tau=ste_mask_tau,
        )
        self.fused_attn = RegionalDSCA(
            n_feats,
            num_heads=num_heads,
            reduction=reduction,
            ste_mask_tau=ste_mask_tau,
        )


class SDAStageSADS(SDAStage):
    def __init__(
        self, n_feats, n_block, num_heads, expansion_ratio,
        drop_path_rate=0.4, reduction=4, dropout=0.2,
        ste_mask_tau=0.15
    ):
        nn.Module.__init__(self)
        dpr = torch.linspace(0, drop_path_rate, n_block).tolist()
        self.body = nn.ModuleList([
            SDABlockSADS(
                n_feats=n_feats,
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                drop_path=dpr[i],
                reduction=reduction,
                dropout=dropout,
                ste_mask_tau=ste_mask_tau,
            )
            for i in range(n_block)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 1)


@ARCH_REGISTRY.register()
class SDANET_SADS(SDANET):
    """SDANet baseline with regional 2x2 dynamic sparse attention."""

    def __init__(
        self,
        n_block=(2, 2, 2, 2),
        n_group=4,
        n_colors=128,
        n_lms_colors=128,
        n_feats=128,
        scale=4,
        num_heads=4,
        expansion_ratio=1.0,
        drop_path_rate=0.4,
        reduction=4,
        dropout=0.2,
        ste_mask_tau=0.15,
    ):
        super().__init__(
            n_block=list(n_block),
            n_group=n_group,
            n_colors=n_colors,
            n_lms_colors=n_lms_colors,
            n_feats=n_feats,
            scale=scale,
            num_heads=num_heads,
            expansion_ratio=expansion_ratio,
            drop_path_rate=drop_path_rate,
            reduction=reduction,
            dropout=dropout,
            ste_mask_tau=ste_mask_tau,
        )
        self.body = nn.ModuleList([
            SDAStageSADS(
                n_feats=n_feats,
                n_block=n_block[i],
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                drop_path_rate=drop_path_rate,
                reduction=reduction,
                dropout=dropout,
                ste_mask_tau=ste_mask_tau,
            )
            for i in range(n_group)
        ])
