"""Fused Metal selective-scan kernel for Mamba on Apple Silicon (PyTorch MPS).

Adapted from the fixed-16 selective-scan kernel of SpeechLens
(https://github.com/faraday/SpeechLens, Sources/Inference/MetalSelectiveScan.swift),
Copyright 2026 Çağatay Çallı, licensed under the Apache License, Version 2.0.
See NOTICE. Changes: ported from MLX/Swift to torch.mps.compile_shader, and the
SSM state is read from, and written back to, a caller-owned buffer. That lets
the same kernel serve both the bidirectional frequency Mamba (zero initial
state) and the stateful time Mamba used for chunked streaming.

One thread owns a (batch, channel) pair, keeps its 16 state values in
registers and walks the sequence; softplus(delta), -exp(A_log), the D skip and
the SiLU(z) gate are all fused, so a whole Mamba scan is a single dispatch
instead of ~5 MPS kernels per sequence step.

Use `accelerate(model)` to route every Mamba mixer in a SEMamba model through it.
"""

import types

import torch
import torch.nn.functional as F

N_STATE = 16

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Matches torch.nn.functional.softplus(beta=1, threshold=20).
inline float scan_softplus(float x) {
    return x > 20.0f ? x : precise::log(1.0f + precise::exp(x));
}

inline float scan_silu(float x) {
    return x / (1.0f + precise::exp(-x));
}

// delta_raw, u, z, out : [B, L, D]   (delta_raw = dt_proj(dt), bias included, pre-softplus)
// Bs, Cs               : [B, L, 16]
// A_log                : [D, 16]
// Dp                   : [D]
// state                : [B, D, 16]  read as the initial state, overwritten with the final state
kernel void selective_scan_fixed16(
    device const float* delta_raw [[buffer(0)]],
    device const float* u         [[buffer(1)]],
    device const float* z         [[buffer(2)]],
    device const float* Bs        [[buffer(3)]],
    device const float* Cs        [[buffer(4)]],
    device const float* A_log     [[buffer(5)]],
    device const float* Dp        [[buffer(6)]],
    device float*       state     [[buffer(7)]],
    device float*       out       [[buffer(8)]],
    constant uint&      B_size    [[buffer(9)]],
    constant uint&      L_size    [[buffer(10)]],
    constant uint&      D_size    [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= B_size * D_size) return;
    uint b = tid / D_size;
    uint d = tid % D_size;

    float h[16];
    float a[16];
    device float* st = state + (b * D_size + d) * 16;
    for (uint n = 0; n < 16; ++n) {
        h[n] = st[n];
        a[n] = -precise::exp(A_log[d * 16 + n]);
    }
    float d_skip = Dp[d];

    uint base_ld = b * L_size * D_size + d;
    uint base_ln = b * L_size * 16;
    for (uint i = 0; i < L_size; ++i) {
        uint o = base_ld + i * D_size;
        float dt = scan_softplus(delta_raw[o]);
        float uv = u[o];
        float dtu = dt * uv;
        device const float* Bi = Bs + base_ln + i * 16;
        device const float* Ci = Cs + base_ln + i * 16;
        float y = 0.0f;
        for (uint n = 0; n < 16; ++n) {
            h[n] = precise::exp(dt * a[n]) * h[n] + dtu * Bi[n];
            y += h[n] * Ci[n];
        }
        y += uv * d_skip;
        out[o] = y * scan_silu(z[o]);
    }
    for (uint n = 0; n < 16; ++n) st[n] = h[n];
}
"""

_lib = None


def _library():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SOURCE)
    return _lib


def selective_scan(delta_raw, u, z, B, C, A_log, D, state):
    """All sequence tensors are [B, L, *] and contiguous; `state` [B, D, 16] is updated in place."""
    Bn, L, Dn = u.shape
    out = torch.empty_like(u)
    _library().selective_scan_fixed16(
        delta_raw, u, z, B, C, A_log, D, state, out, Bn, L, Dn, threads=Bn * Dn
    )
    return out


def mixer_forward(mixer, hidden_states, conv_state=None, ssm_state=None):
    """Mamba mixer forward on [B, L, d_model] using the fused scan.

    With conv_state/ssm_state (as allocated by mixer.allocate_inference_cache) the
    call continues from, and updates, that state; without them it starts from zero,
    exactly like mamba_ssm's Mamba.forward.
    """
    Bn, L, _ = hidden_states.shape
    x, z = mixer.in_proj(hidden_states).chunk(2, dim=-1)  # [B, L, D] each
    xt = x.transpose(1, 2)  # [B, D, L]
    w, b, groups = mixer.conv1d.weight, mixer.conv1d.bias, mixer.conv1d.groups
    if conv_state is None:
        xt = F.conv1d(xt, w, b, padding=mixer.d_conv - 1, groups=groups)[..., :L]
    else:
        # conv_state holds the last d_conv inputs (step() semantics); the oldest d_conv-1 are history.
        x_cat = torch.cat([conv_state[:, :, 1:], xt], dim=-1)
        conv_state.copy_(x_cat[:, :, -mixer.d_conv:])
        xt = F.conv1d(x_cat, w, b, groups=groups)
    u = F.silu(xt).transpose(1, 2).contiguous()  # [B, L, D]

    dt, Bs, Cs = torch.split(mixer.x_proj(u), [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1)
    delta_raw = mixer.dt_proj(dt).contiguous()

    if ssm_state is None:
        state = torch.zeros(Bn, u.shape[-1], N_STATE, device=u.device, dtype=torch.float32)
    else:
        state = ssm_state
    y = selective_scan(delta_raw, u, z.contiguous(), Bs.contiguous(), Cs.contiguous(),
                       mixer.A_log.float().contiguous(), mixer.D.float().contiguous(), state)
    return mixer.out_proj(y)


def accelerate(model) -> int:
    """Route every Mamba mixer of `model` through the fused Metal scan. Returns the count."""
    from mamba_ssm.modules.mamba_simple import Mamba
    from .mamba_block import MambaBlock

    n = 0
    for module in model.modules():
        if isinstance(module, Mamba):
            if module.d_state != N_STATE:
                raise ValueError(f"fused scan requires d_state == {N_STATE}, got {module.d_state}")

            def forward(self, hidden_states, inference_params=None):
                if inference_params is not None:
                    raise RuntimeError("accelerated Mamba does not use inference_params")
                return mixer_forward(self, hidden_states)

            module.forward = types.MethodType(forward, module)
            n += 1
        elif isinstance(module, MambaBlock):
            module.mixer_chunk_fn = mixer_forward  # stateful streaming path, any chunk length
    return n
