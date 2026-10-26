# -*- coding: utf-8 -*-
"""
Export the released FlowMatchingModel to ONNX for in-browser inference.

The whole flow-ODE solve is exported as ONE graph per step count (context and the
cross-attention K/V computed once, Euler steps unrolled), so the browser makes a
single call per re-plan. Verified against the released ICTPolicy.infer (~1e-7).
  serve_bread_rollout20.onnx       fp32, 20 Euler steps   (WebGPU)
  serve_bread_rollout10_fp16.onnx  fp16 weights, 10 steps (CPU / WASM fallback)

    python simulation/scripts/export_onnx.py --ckpt checkpoints/serve_bread/latest.pt --out simulation/web/assets/policy
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from inference.policy import ICTPolicy  # noqa: E402


class Rollout(torch.nn.Module):
    """The whole 20-step Euler flow solve as one graph, context and cross-attention K/V computed once.

    Re-implements FlowMatchingModel.forward's decoder with explicit attention (same weights,
    eval mode) so the per-step work is only the 50 action queries."""

    def __init__(self, m, steps):
        super().__init__()
        self.m, self.steps = m, steps

    @staticmethod
    def _split(x, h):   # (B, L, E) -> (B, h, L, E/h)
        B, L, E = x.shape
        return x.view(B, L, h, E // h).transpose(1, 2)

    def _mha_self(self, mha, x):
        E, h = x.shape[-1], mha.num_heads
        W, b = mha.in_proj_weight, mha.in_proj_bias
        q, k, v = torch.nn.functional.linear(x, W, b).split(E, dim=-1)
        q, k, v = self._split(q, h), self._split(k, h), self._split(v, h)
        a = torch.softmax(q @ k.transpose(-1, -2) / (E // h) ** 0.5, dim=-1) @ v
        return mha.out_proj(a.transpose(1, 2).reshape(x.shape))

    def _mha_cross(self, mha, x, K, V, bias):
        E, h = x.shape[-1], mha.num_heads
        q = torch.nn.functional.linear(x, mha.in_proj_weight[:E], mha.in_proj_bias[:E])
        q = self._split(q, h)
        a = torch.softmax(q @ K.transpose(-1, -2) / (E // h) ** 0.5 + bias, dim=-1) @ V
        return mha.out_proj(a.transpose(1, 2).reshape(x.shape))

    def forward(self, x_rgb, x_ict, ict_mask, anchor_uv, noise):
        m = self.m
        B = x_rgb.shape[0]
        vis = m.rgb_embed(x_rgb)
        N, T = vis.shape[1], x_ict.shape[1]
        ict = m.state_proj(x_ict) + m.state_pos_emb[:, :T, :]
        ctx = m.ctx_norm(torch.cat([ict, vis], dim=1))
        valid = torch.cat([ict_mask, torch.ones(B, N, dtype=ict_mask.dtype)], dim=1)       # 1 = attend
        neg = (1.0 - valid) * -1e9
        bias = neg[:, None, None, :]
        if m.use_region_attn:
            sb = m._generate_spatial_bias(anchor_uv, T, N).view(B, m.num_heads, m.pred_horizon, T + N)
            bias = bias + sb
        E, h = ctx.shape[-1], m.num_heads
        KV = []
        for blk in m.decoder:   # cross-attention keys / values of the (fixed) context, once
            W, b = blk.cross_attn.in_proj_weight, blk.cross_attn.in_proj_bias
            K = self._split(torch.nn.functional.linear(ctx, W[E:2 * E], b[E:2 * E]), h)
            V = self._split(torch.nn.functional.linear(ctx, W[2 * E:], b[2 * E:]), h)
            KV.append((K, V))
        w = valid.unsqueeze(-1)
        done = m.head_done((ctx * w).sum(1) / w.sum(1).clamp(min=1e-6))
        x = noise
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((B, 1), i * dt)
            q = m.action_proj(x) + m.pos_embed.unsqueeze(0) + m.time_mlp(t).unsqueeze(1)
            for blk, (K, V) in zip(m.decoder, KV):
                q = q + self._mha_self(blk.self_attn, blk.norm1(q))
                q = q + self._mha_cross(blk.cross_attn, blk.norm2(q), K, V, bias)
                q = q + blk.mlp(blk.norm3(q))
            x = x + m.head_v(q) * dt
        return x, done


class Full(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x_rgb, x_ict, ict_mask, x_t, t, anchor_uv):
        out = self.m(x_rgb=x_rgb, x_ict=x_ict, ict_mask=ict_mask > 0.5, x_t=x_t, t=t, anchor_uv=anchor_uv)
        return out["v_pred"], out["done_logit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/serve_bread/latest.pt")
    ap.add_argument("--out", default="simulation/web/assets/policy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pol = ICTPolicy({"ckpt": a.ckpt, "num_inference_steps": 20}, device="cpu")
    m = pol.model.eval()
    full = Full(m).eval()
    B, K, D = 1, pol.pred_horizon, pol.action_dim
    x_rgb = torch.rand(B, 3, *pol.img_size)
    x_ict = torch.randn(B, pol.max_ict, pol.ict_dim)
    mask = torch.zeros(B, pol.max_ict)
    mask[:, :3] = 1
    x_t = torch.randn(B, K, D)
    t = torch.full((B, 1), 0.35)
    uv = torch.tensor([[0.6, 0.55]])
    import onnxruntime as ort
    # ---- one-call rollout graph (what the browser runs) ----
    for steps in (20, 10):
        roll = Rollout(m, steps).eval()
        noise = pol.noise.cpu()
        pr = os.path.join(a.out, f"serve_bread_rollout{steps}.onnx")
        torch.onnx.export(roll, (x_rgb, x_ict, mask, uv, noise), pr, opset_version=17,
                          input_names=["x_rgb", "x_ict", "ict_mask", "anchor_uv", "noise"],
                          output_names=["actions", "done_logit"], dynamo=False)
        sr = ort.InferenceSession(pr, providers=["CPUExecutionProvider"])
        act, dl = sr.run(None, {"x_rgb": x_rgb.numpy(), "x_ict": x_ict.numpy(), "ict_mask": mask.numpy(),
                                "anchor_uv": uv.numpy(), "noise": noise.numpy()})
        if steps == 20:   # reference: the released ICTPolicy.infer (20 Euler steps on the original module)
            pol.noise = noise.to(pol.noise.device)
            traj, done_ref = pol.infer(x_rgb, x_ict, mask > 0.5, uv)
            ref = np.concatenate([traj["right"][0], traj["right"][1]], 1)
            print("rollout20 vs ICTPolicy.infer: max |pos/o6d| diff", float(np.abs(act[0, :, :9] - ref).max()),
                  " done", float(abs(1 / (1 + np.exp(-dl[0, 0])) - done_ref)))
        t0 = time.time()
        for _ in range(5):
            sr.run(None, {"x_rgb": x_rgb.numpy(), "x_ict": x_ict.numpy(), "ict_mask": mask.numpy(),
                          "anchor_uv": uv.numpy(), "noise": noise.numpy()})
        print(f"rollout{steps}: {os.path.getsize(pr) / 1e6:.1f} MB, CPU {(time.time() - t0) / 5 * 1e3:.0f} ms / re-plan")
    # shipped: fp32 20-step graph (WebGPU) + fp16-weight 10-step graph (CPU / WASM)
    import onnx
    from onnxconverter_common import float16
    p10 = os.path.join(a.out, "serve_bread_rollout10.onnx")
    onnx.save(float16.convert_float_to_float16(onnx.load(p10), keep_io_types=True),
              os.path.join(a.out, "serve_bread_rollout10_fp16.onnx"))
    for f in ("serve_bread_rollout10.onnx", "serve_bread_rollout10_int8.onnx", "serve_bread_rollout20_int8.onnx"):
        if os.path.exists(os.path.join(a.out, f)):
            os.remove(os.path.join(a.out, f))
    meta = {"pred_horizon": K, "action_dim": D, "img_size": list(pol.img_size), "max_ict": pol.max_ict,
            "ict_dim": pol.ict_dim, "num_inference_steps": 20, "pos_mean": pol.pos_mean.tolist(),
            "pos_std": pol.pos_std.tolist(), "single_hand": pol.single_hand, "side": pol.single_hand_side,
            "noise": pol.noise.cpu().numpy().ravel().round(6).tolist()}
    with open(os.path.join(a.out, "serve_bread_meta.json"), "w") as f:
        json.dump(meta, f)
    print("onnx size MB:", os.path.getsize(p) / 1e6)

    t0 = time.time()
    for _ in range(20):
        s.run(None, feeds)
    print("onnxruntime CPU, 20 steps: %.0f ms" % ((time.time() - t0) * 1e3))


if __name__ == "__main__":
    main()
