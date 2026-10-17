"""MPGFormer main model.

This file defines the forecasting pipeline. Component implementations live in
``modules/`` and share the public model interface defined here.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import (
        MPGFormerOutput,
        DriftAwareRouter,
        DynamicGraphContrastiveEncoder,
        MultiScaleTimeFrequencyBank,
        NonStationaryAttentionLayer,
        PositionalEncoding,
        ShortTermTemporalEncoder,
        StaticContextSelector,
        TargetRevIN,
    )
except ImportError:
    # Support running train_mpgformer.py directly from the MPGFormer folder.
    from modules import (
        MPGFormerOutput,
        DriftAwareRouter,
        DynamicGraphContrastiveEncoder,
        MultiScaleTimeFrequencyBank,
        NonStationaryAttentionLayer,
        PositionalEncoding,
        ShortTermTemporalEncoder,
        StaticContextSelector,
        TargetRevIN,
    )


class MPGFormer(nn.Module):
    """A drift-aware workload forecasting model.

    MPGFormer combines four complementary signals:
        1. Non-stationary temporal attention for long-range history.
        2. Short-term convolution/GRU encoding for bursty workload changes.
        3. Multi-scale time-frequency prototypes for periodic patterns.
        4. Dynamic graph contrastive context across workloads in the batch.

    The model keeps the same input/output behavior as the previous single-file
    implementation: ``forward(..., return_dict=True)`` returns ``MPGFormerOutput``
    and ``return_dict=False`` returns only the point forecast.
    """

    def __init__(
        self,
        input_size: int,
        enc_seq_len: int = 48,
        label_len: int = 12,
        pred_len: int = 24,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        dropout: float = 0.1,
        static_dim: int = 0,
        target_index: int = 0,
        patch_sizes: Sequence[int] = (6, 12, 24, 48),
        num_prototypes: int = 32,
        moving_avg: int = 25,
        graph_topk: int = 8,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        use_revin: bool = True,
        disabled_components: Sequence[str] = (),
    ):
        super().__init__()
        allowed = {"revin", "temperature", "short_encoder", "prototype", "graph", "router",
                   "workload_graph", "context_graph", "prototype_entropy", "drift_bias"}
        unknown = set(disabled_components) - allowed
        if unknown:
            raise ValueError(f"Unknown disabled components: {sorted(unknown)}")
        self.disabled_components = frozenset(disabled_components)
        self.input_size = input_size
        self.enc_seq_len = enc_seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target_index = target_index
        self.quantiles = tuple(float(q) for q in quantiles)
        self.use_revin = use_revin and "revin" not in self.disabled_components

        # Input and context adapters.
        self.revin = TargetRevIN(target_index=target_index)
        self.input_proj = nn.Linear(input_size, d_model)
        self.decoder_proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.static_selector = StaticContextSelector(d_model, static_dim, dropout)

        # Temporal encoders: long memory attention followed by burst-sensitive local encoding.
        self.long_layers = nn.ModuleList(
            [NonStationaryAttentionLayer(d_model, n_heads, d_ff, dropout, moving_avg,
                                         use_temperature="temperature" not in self.disabled_components)
             for _ in range(e_layers)]
        )
        self.short_encoder = (nn.Identity() if "short_encoder" in self.disabled_components
                              else ShortTermTemporalEncoder(d_model, dropout))

        # Auxiliary experts used by the drift-aware router.
        self.prototype_bank = MultiScaleTimeFrequencyBank(
            d_model=d_model,
            enc_seq_len=enc_seq_len,
            pred_len=pred_len,
            patch_sizes=patch_sizes,
            num_prototypes=num_prototypes,
            dropout=dropout,
        )
        self.graph_encoder = DynamicGraphContrastiveEncoder(d_model, static_dim, dropout, graph_topk)
        self.router = DriftAwareRouter(d_model, dropout)
        self.context_proj = nn.Linear(d_model, d_model)

        # Forecast decoder. The decoder receives known labels plus generated future padding.
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=d_layers)

        # Four channels are predicted: median, lower distance, upper distance, point value.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        self.point_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build encoder history and a decoder sequence with masked future targets.

        The decoder retains the last ``label_len`` observations and any known
        future covariates. Forecast targets are never read from future input.
        """

        bsz, total_len, feat = x.shape
        if feat != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {feat}")

        # Very short sequences are left-padded so the encoder always sees enc_seq_len.
        if total_len < self.enc_seq_len:
            pad = self.enc_seq_len - total_len
            x = F.pad(x.transpose(1, 2), (pad, 0), mode="replicate").transpose(1, 2)
            total_len = x.size(1)

        enc_input = x[:, : self.enc_seq_len, :]

        if total_len >= self.enc_seq_len + self.pred_len:
            future = x[:, self.enc_seq_len : self.enc_seq_len + self.pred_len, :].clone()
        else:
            future = x[:, -1:, :].repeat(1, self.pred_len, 1)

        # Future target values must be hidden from the decoder during forecasting.
        future[..., self.target_index] = 0.0

        label = enc_input[:, -self.label_len :, :]
        dec_input = torch.cat([label, future], dim=1)
        return enc_input, dec_input

    def _normalize_inputs(
        self,
        enc_raw: torch.Tensor,
        dec_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Normalize only the target channel and retain inverse-transform stats."""

        if not self.use_revin:
            return enc_raw, dec_raw, None, None

        enc_input, mean, std = self.revin.normalize(enc_raw)
        dec_input = dec_raw.clone()
        dec_input[..., self.target_index : self.target_index + 1] = (
            dec_input[..., self.target_index : self.target_index + 1] - mean
        ) / std

        return enc_input, dec_input, mean, std

    def _encode_history(
        self,
        enc_input: torch.Tensor,
        raw_target_history: torch.Tensor,
        static_context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode history tokens using attention scales from raw target history.

        Return sequence states, target history for other branches,
        and the pooled temporal expert context.
        """

        target_history = enc_input[..., self.target_index]
        encoded = self.pos(self.input_proj(enc_input))
        encoded = self.static_selector(encoded, static_context)

        for layer in self.long_layers:
            encoded = layer(encoded, raw_target_history)

        encoded = self.short_encoder(encoded)
        time_context = encoded.mean(dim=1)
        return encoded, target_history, time_context

    def _fuse_expert_contexts(
        self,
        time_context: torch.Tensor,
        target_history: torch.Tensor,
        raw_target_history: torch.Tensor,
        static_context: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse temporal, prototype, and graph evidence for each window.

        Prototypes and workload similarity use encoder target history, which is
        normalized when RevIN is enabled. Drift cues and fallback context-graph
        statistics use the pre-RevIN target history.
        Fully disabled experts contribute zero and receive no routing weight.
        """

        use_prototype = "prototype" not in self.disabled_components
        use_workload_graph = "workload_graph" not in self.disabled_components
        use_context_graph = "context_graph" not in self.disabled_components
        use_graph = "graph" not in self.disabled_components and (use_workload_graph or use_context_graph)

        if use_prototype:
            prototype_context, future_padding, prototype_weights, _ = self.prototype_bank(target_history)
        else:
            prototype_context = torch.zeros_like(time_context)
            future_padding = target_history.new_zeros(target_history.size(0), self.pred_len)
            prototype_weights = target_history.new_zeros(
                target_history.size(0), self.prototype_bank.num_scales * self.prototype_bank.num_prototypes)
        if use_graph:
            graph_context, contrastive_loss, adj = self.graph_encoder(
                time_context, target_history, static_context,
                raw_target_history=raw_target_history,
                use_workload_graph=use_workload_graph, use_context_graph=use_context_graph)
        else:
            graph_context = torch.zeros_like(time_context)
            contrastive_loss = time_context.new_zeros(())
            adj = target_history.new_zeros(target_history.size(0), target_history.size(0))

        router_weights, drift_score = self.router(
            time_context=time_context,
            graph_context=graph_context,
            prototype_context=prototype_context,
            target_history=raw_target_history,
            prototype_weights=prototype_weights,
            active_experts=(True, use_graph, use_prototype),
            use_prototype_entropy=use_prototype and "prototype_entropy" not in self.disabled_components,
            use_drift_bias="drift_bias" not in self.disabled_components,
            fixed_average="router" in self.disabled_components,
        )

        fused_context = (
            router_weights[:, 0:1] * time_context
            + router_weights[:, 1:2] * graph_context
            + router_weights[:, 2:3] * prototype_context
        )

        return fused_context, future_padding, prototype_weights, contrastive_loss, adj, router_weights, drift_score

    def _decode_future(
        self,
        encoded: torch.Tensor,
        dec_input: torch.Tensor,
        fused_context: torch.Tensor,
        future_padding: torch.Tensor,
        static_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decode future tokens using prototype padding and fused encoder memory."""

        dec_input = dec_input.clone()
        dec_input[:, -self.pred_len :, self.target_index] = future_padding

        memory = encoded + self.context_proj(fused_context).unsqueeze(1)
        decoded = self.pos(self.decoder_proj(dec_input))
        decoded = self.static_selector(decoded, static_context)
        decoded = self.decoder(decoded, memory)[:, -self.pred_len :, :]
        return decoded

    def _ordered_quantiles(self, raw: torch.Tensor, decoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert raw decoder channels into ordered interval forecasts.

        The point forecast is softly anchored to the median quantile. This keeps
        the point head expressive, but reduces rare uncalibrated spikes that can
        disproportionately increase MSE on sparse Google-style workloads.
        """

        median = raw[..., 0]
        lower = median - F.softplus(raw[..., 1])
        upper = median + F.softplus(raw[..., 2])
        raw_point = raw[..., 3]
        point_gate = torch.sigmoid(self.point_gate(decoded)).squeeze(-1)
        point = point_gate * raw_point + (1.0 - point_gate) * median
        quantiles = torch.stack([lower, median, upper], dim=-1)
        return point, quantiles

    def forward(
        self,
        x: torch.Tensor,
        static_context: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """Forecast from a masked window and optional static context.

        Standard ``x`` has shape ``[batch, enc_seq_len + pred_len, input_size]``.
        Return ``MPGFormerOutput`` by default or only point forecasts when
        ``return_dict`` is false. Predictions are restored to the input scale.
        """
        if not torch.is_floating_point(x):
            x = x.float()
        if static_context is not None and not torch.is_floating_point(static_context):
            static_context = static_context.float()

        enc_raw, dec_raw = self._split_inputs(x)
        raw_target_history = enc_raw[..., self.target_index]
        enc_input, dec_input, mean, std = self._normalize_inputs(enc_raw, dec_raw)

        encoded, target_history, time_context = self._encode_history(
            enc_input, raw_target_history, static_context)
        fused_context, future_padding, prototype_weights, contrastive_loss, adj, router_weights, drift_score = (
            self._fuse_expert_contexts(time_context, target_history, raw_target_history, static_context)
        )

        decoded = self._decode_future(encoded, dec_input, fused_context, future_padding, static_context)
        point, quantiles = self._ordered_quantiles(self.head(decoded), decoded)

        if self.use_revin:
            point = self.revin.denormalize_target(point, mean, std)
            quantiles = self.revin.denormalize_target(quantiles, mean, std)

        lower = quantiles[..., 0]
        upper = quantiles[..., -1]

        if not return_dict:
            return point

        return MPGFormerOutput(
            point=point,
            quantiles=quantiles,
            lower=lower,
            upper=upper,
            router_weights=router_weights,
            drift_score=drift_score,
            contrastive_loss=contrastive_loss,
            prototype_weights=prototype_weights,
            graph_context=adj,
        )
