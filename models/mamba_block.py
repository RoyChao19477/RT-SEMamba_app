# Reference: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm


# Reference implementation adapted from https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model,
    cfg,
    layer_idx=0,
    rms_norm=True,
    fused_add_norm=False,
    residual_in_fp32=False,
):
    d_state = cfg["model_cfg"]["d_state"]
    d_conv = cfg["model_cfg"]["d_conv"]
    expand = cfg["model_cfg"]["expand"]
    norm_epsilon = cfg["model_cfg"]["norm_epsilon"]
    use_fast_path = cfg["model_cfg"].get("use_fast_path", False)

    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        use_fast_path=use_fast_path,
    )
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg, bidirectional=True, layer_offset=0):
        super(MambaBlock, self).__init__()
        n_layer = cfg["model_cfg"].get("inner_mamba_nlayer", 1)
        self.bidirectional = bidirectional
        self._streaming_enabled = False
        self._streaming_states = None
        self._streaming_batch = None
        # Optional fn(mixer, x, conv_state, ssm_state) for the streaming path (e.g. models.mps_kernels).
        self.mixer_chunk_fn = None
        self.forward_blocks = nn.ModuleList(
            create_block(in_channels, cfg, layer_idx=layer_offset + i) for i in range(n_layer)
        )
        self.backward_blocks = (
            nn.ModuleList(create_block(in_channels, cfg, layer_idx=layer_offset + n_layer + i) for i in range(n_layer))
            if bidirectional
            else None
        )
        self.total_layers = n_layer + (n_layer if bidirectional else 0)

        self.apply(partial(_init_weights, n_layer=n_layer))

    def set_streaming(self, enabled: bool) -> None:
        self._streaming_enabled = enabled
        if enabled:
            self.reset_streaming_state()

    def reset_streaming_state(self) -> None:
        self._streaming_states = None
        self._streaming_batch = None

    def _ensure_streaming_state(self, batch_size: int) -> None:
        if self._streaming_states is not None and self._streaming_batch == batch_size:
            return
        self._streaming_states = []
        for block in self.forward_blocks:
            conv_state, ssm_state = block.mixer.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=1,
            )
            self._streaming_states.append((conv_state, ssm_state))
        self._streaming_batch = batch_size

    @staticmethod
    def _mixer_chunk(mixer, hidden_states, conv_state, ssm_state):
        """Mamba mixer over a chunk of L>=1 steps, starting from (and updating) conv/ssm state in place.

        Numerically equivalent to L consecutive mixer.step() calls, but uses one batched
        projection/conv per chunk, which is much cheaper on GPUs with high per-kernel overhead.
        """
        seqlen = hidden_states.shape[1]
        x, z = mixer.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)  # (B, D, L)

        # conv_state holds the last d_conv inputs (step() semantics); its oldest d_conv-1 are the causal history.
        x_cat = torch.cat([conv_state[:, :, 1:], x], dim=-1)
        conv_state.copy_(x_cat[:, :, -mixer.d_conv:])
        x = F.conv1d(x_cat, mixer.conv1d.weight, mixer.conv1d.bias, groups=mixer.conv1d.groups)
        x = mixer.act(x).transpose(1, 2)  # (B, L, D)

        dt, B, C = torch.split(mixer.x_proj(x), [mixer.dt_rank, mixer.d_state, mixer.d_state], dim=-1)
        dt = F.softplus(mixer.dt_proj(dt))  # (B, L, D)
        A = -torch.exp(mixer.A_log.float())  # (D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, D, N)
        dBx = (dt * x).unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

        h = ssm_state
        ys = []
        for t in range(seqlen):
            h = dA[:, t] * h + dBx[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        ssm_state.copy_(h)

        y = torch.stack(ys, dim=1) + x * mixer.D
        y = y * mixer.act(z.transpose(1, 2))
        return mixer.out_proj(y)

    def _streaming_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            raise RuntimeError("Streaming is only supported for unidirectional Mamba blocks.")
        batch_size = x.shape[0]
        self._ensure_streaming_state(batch_size)
        hidden_states = x
        residual = None
        for idx, block in enumerate(self.forward_blocks):
            if block.fused_add_norm:
                raise RuntimeError("Streaming path does not support fused_add_norm blocks.")
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = block.norm(residual.to(dtype=block.norm.weight.dtype))
            if block.residual_in_fp32:
                residual = residual.to(torch.float32)
            conv_state, ssm_state = self._streaming_states[idx]
            if self.mixer_chunk_fn is not None:
                hidden_states = self.mixer_chunk_fn(block.mixer, hidden_states, conv_state, ssm_state)
            elif hidden_states.shape[1] == 1:
                hidden_states, _, _ = block.mixer.step(hidden_states, conv_state, ssm_state)
            else:
                hidden_states = self._mixer_chunk(block.mixer, hidden_states, conv_state, ssm_state)
        return (hidden_states + residual) if residual is not None else hidden_states

    def forward(self, x, inference_params=None):
        if self._streaming_enabled:
            return self._streaming_forward(x)
        x_forward = x.clone()
        resi_forward = None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward, inference_params=inference_params)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        if not self.bidirectional:
            return y_forward

        # Backward (adds lookahead; used only on frequency axis)
        # Note: Backward direction does NOT use inference_params (not causal)
        x_backward = torch.flip(x, [1])
        resi_backward = None
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward, inference_params=None)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)


class TFMambaBlock(nn.Module):
    """Temporal-Frequency Mamba block for sequence modelling."""

    def __init__(self, cfg, layer_offset=0):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg["model_cfg"]["hid_feature"]
        self.time_bidirectional = cfg["model_cfg"].get("time_mamba_bidirectional", False)
        self.freq_bidirectional = cfg["model_cfg"].get("freq_mamba_bidirectional", True)
        self._streaming_enabled = False

        self.time_mamba = MambaBlock(
            in_channels=self.hid_feature,
            cfg=cfg,
            bidirectional=self.time_bidirectional,
            layer_offset=layer_offset,
        )
        freq_offset = layer_offset + self.time_mamba.total_layers
        self.freq_mamba = MambaBlock(
            in_channels=self.hid_feature,
            cfg=cfg,
            bidirectional=self.freq_bidirectional,
            layer_offset=freq_offset,
        )
        self.total_layers = self.time_mamba.total_layers + self.freq_mamba.total_layers

        self.t_out_channels = self.hid_feature * (2 if self.time_bidirectional else 1)
        self.f_out_channels = self.hid_feature * (2 if self.freq_bidirectional else 1)

        self.tlinear = nn.ConvTranspose1d(self.t_out_channels, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.f_out_channels, self.hid_feature, 1, stride=1)

    def set_streaming(self, enabled: bool) -> None:
        self._streaming_enabled = enabled
        self.time_mamba.set_streaming(enabled)

    def reset_streaming_state(self) -> None:
        self.time_mamba.reset_streaming_state()

    def forward(self, x, inference_params=None):
        b, c, t, f = x.size()

        # --- Temporal Processing ---
        x_res_time = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        # Temporal Mamba is causal; allow inference_params for stateful streaming.
        if inference_params is not None and inference_params.seqlen_offset > 0 and t > 1:
            raise ValueError(
                "TFMambaBlock time_mamba expects frame-by-frame inputs (t=1) when using inference_params "
                "with non-zero seqlen_offset."
            )
        if self._streaming_enabled:
            x_time = self.time_mamba(x_res_time, inference_params=None)
        else:
            x_time = self.time_mamba(x_res_time, inference_params=inference_params)
        x_time = self.tlinear(x_time.permute(0, 2, 1)).permute(0, 2, 1)
        x = x_res_time + x_time

        # --- Frequency Processing ---
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_res_freq = x

        # Frequency is not a causal axis: this scan is bidirectional and runs over the
        # full spectrum of the current frame, so it carries no state across time steps.
        x_freq = self.freq_mamba(x, inference_params=None)
        x_freq = self.flinear(x_freq.permute(0, 2, 1)).permute(0, 2, 1)
        x = x_res_freq + x_freq

        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MLPWithNorm(nn.Module):
    """2-layer MLP + norm between TFMamba or Transformer blocks."""

    def __init__(self, cfg):
        super(MLPWithNorm, self).__init__()
        self.hid_feature = cfg["model_cfg"]["hid_feature"]
        norm_epsilon = cfg["model_cfg"]["norm_epsilon"]

        self.mlp = nn.Sequential(
            nn.Linear(self.hid_feature, self.hid_feature * 4),
            nn.GELU(),
            nn.Linear(self.hid_feature * 4, self.hid_feature),
            nn.Dropout(0.1),
        )

        self.norm = nn.LayerNorm(self.hid_feature, eps=norm_epsilon)

    def forward(self, x):
        b, c, t, f = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(b * t * f, c)
        x_mlp = self.mlp(x_flat)
        x_norm = self.norm(x_mlp)
        x_out = x_norm.view(b, t, f, c).permute(0, 3, 1, 2)
        return x + x_out
