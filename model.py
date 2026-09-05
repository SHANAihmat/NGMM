import torch
import torch.nn as nn
import torch.nn.functional as F

# class MultiScaleTemporalBlock(nn.Module):
#     # 🌟 接收 kernel_size
#     def __init__(self, hidden_dim, kernel_size=3):
#         super().__init__()
#         short_dim = hidden_dim // 3
#         mid_dim = hidden_dim // 3
#         long_dim = hidden_dim - short_dim - mid_dim

#         # 原来的独立多尺度分支
#         # self.conv_short = nn.Conv1d(hidden_dim, short_dim, kernel_size, padding=kernel_size//2)
#         # self.conv_mid = nn.Conv1d(hidden_dim, mid_dim, kernel_size,
#         #                           padding=(kernel_size//2)*3, dilation=3)
#         # self.conv_long = nn.Conv1d(hidden_dim, long_dim, kernel_size,
#         #                            padding=(kernel_size//2)*7, dilation=7)

#         # 从长到短显式耦合多尺度特征。
#         self.conv_long = nn.Conv1d(hidden_dim, long_dim, kernel_size,
#                                    padding=(kernel_size//2)*7, dilation=7)
#         self.conv_mid = nn.Conv1d(hidden_dim + long_dim, mid_dim, kernel_size,
#                                   padding=(kernel_size//2)*3, dilation=3)
#         self.conv_short = nn.Conv1d(hidden_dim + mid_dim + long_dim, short_dim,
#                                     kernel_size, padding=kernel_size//2)
        
#         self.proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
#         self.act  = nn.SiLU()
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x):
#         res = x
#         x_conv = x.transpose(1, 2).contiguous() 
        
#         # 原来的独立多尺度分支
#         # h_short = self.conv_short(x_conv)
#         # h_mid = self.conv_mid(x_conv)
#         # h_long = self.conv_long(x_conv)

#         h_long = self.conv_long(x_conv)
#         h_mid = self.conv_mid(torch.cat([x_conv, h_long], dim=1))
#         h_short = self.conv_short(torch.cat([x_conv, h_mid, h_long], dim=1))
#         h = torch.cat([h_short, h_mid, h_long], dim=1)
#         h = self.act(self.proj(h))
#         h = h.transpose(1, 2).contiguous()
        
#         return self.norm(res + h)       

# class MultiScaleTemporalBlock(nn.Module):
#     def __init__(self, hidden_dim, kernel_size=3):
#         super().__init__()
#         # 🔑 关键改进：每个分支输出相同维度，避免信息瓶颈
#         branch_dim = hidden_dim // 3
        
#         # 长尺度分支：直接从输入提取
#         self.conv_long = nn.Conv1d(
#             hidden_dim, branch_dim, kernel_size,
#             padding=(kernel_size//2)*7, dilation=7
#         )
        
#         # 中尺度分支：融合原始输入 + 长尺度特征
#         self.conv_mid = nn.Conv1d(
#             hidden_dim + branch_dim, branch_dim, kernel_size,
#             padding=(kernel_size//2)*3, dilation=3
#         )
        
#         # 短尺度分支：融合原始输入 + 长中尺度特征
#         self.conv_short = nn.Conv1d(
#             hidden_dim + 2*branch_dim, branch_dim, kernel_size,
#             padding=kernel_size//2
#         )
        
#         # 🔑 融合层：将3个分支整合回hidden_dim
#         self.fusion = nn.Sequential(
#             nn.Conv1d(3*branch_dim, hidden_dim, kernel_size=1),
#             nn.SiLU()
#         )
        
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x):
#         res = x
#         x_conv = x.transpose(1, 2).contiguous()
        
#         # 级联提取
#         h_long = self.conv_long(x_conv)
#         h_mid = self.conv_mid(torch.cat([x_conv, h_long], dim=1))
#         h_short = self.conv_short(torch.cat([x_conv, h_mid, h_long], dim=1))
        
#         # 融合所有尺度
#         h = torch.cat([h_long, h_mid, h_short], dim=1)
#         h = self.fusion(h)
#         h = h.transpose(1, 2).contiguous()
        
#         return self.norm(res + h)


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale temporal modeling with selectable coarse-to-fine fusion.

    ``independent`` is a parallel multi-scale baseline. ``cascade`` forces
    long-range context into finer scales. ``gated_cascade`` is the proposed
    model, which adaptively mixes cascade and independent representations.
    """
    FUSION_MODES = {'independent', 'cascade', 'gated_cascade'}

    def __init__(self, hidden_dim, kernel_size=3, fusion_mode='gated_cascade'):
        super().__init__()
        if fusion_mode not in self.FUSION_MODES:
            raise ValueError(f'Unsupported temporal fusion mode: {fusion_mode}.')
        branch_dim = hidden_dim // 3
        self.fusion_mode = fusion_mode

        # The long-scale branch is shared by all ablations.
        self.conv_long = nn.Conv1d(hidden_dim, branch_dim, kernel_size,
                                   padding=(kernel_size//2)*7, dilation=7)
        if fusion_mode in {'cascade', 'gated_cascade'}:
            self.conv_mid = nn.Conv1d(hidden_dim + branch_dim, branch_dim, kernel_size,
                                      padding=(kernel_size//2)*3, dilation=3)
            self.conv_short = nn.Conv1d(hidden_dim + 2*branch_dim, branch_dim,
                                        kernel_size, padding=kernel_size//2)
        if fusion_mode in {'independent', 'gated_cascade'}:
            self.indep_mid = nn.Conv1d(hidden_dim, branch_dim, kernel_size,
                                       padding=(kernel_size//2)*3, dilation=3)
            self.indep_short = nn.Conv1d(hidden_dim, branch_dim, kernel_size,
                                         padding=kernel_size//2)
        if fusion_mode == 'gated_cascade':
            self.gate_mid = nn.Sequential(
                nn.Conv1d(hidden_dim + branch_dim, branch_dim, 1),
                nn.Sigmoid(),
            )
            self.gate_short = nn.Sequential(
                nn.Conv1d(hidden_dim + 2*branch_dim, branch_dim, 1),
                nn.Sigmoid(),
            )
        
        self.fusion = nn.Sequential(
            nn.Conv1d(3*branch_dim, hidden_dim, 1),
            nn.SiLU()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        res = x
        x_conv = x.transpose(1, 2).contiguous()
        
        h_long = self.conv_long(x_conv)
        if self.fusion_mode == 'independent':
            h_mid = self.indep_mid(x_conv)
            h_short = self.indep_short(x_conv)
        else:
            mid_input = torch.cat([x_conv, h_long], dim=1)
            h_mid_cascade = self.conv_mid(mid_input)
            short_input = torch.cat([x_conv, h_mid_cascade, h_long], dim=1)
            h_short_cascade = self.conv_short(short_input)
            if self.fusion_mode == 'cascade':
                h_mid, h_short = h_mid_cascade, h_short_cascade
            else:
                h_mid_indep = self.indep_mid(x_conv)
                gate_mid = self.gate_mid(mid_input)
                h_mid = gate_mid * h_mid_cascade + (1.0 - gate_mid) * h_mid_indep

                short_input = torch.cat([x_conv, h_mid, h_long], dim=1)
                h_short_cascade = self.conv_short(short_input)
                h_short_indep = self.indep_short(x_conv)
                gate_short = self.gate_short(short_input)
                h_short = gate_short * h_short_cascade + (1.0 - gate_short) * h_short_indep
        
        h = torch.cat([h_long, h_mid, h_short], dim=1)
        h = self.fusion(h)
        h = h.transpose(1, 2).contiguous()
        
        return self.norm(res + h)




class DoubleAttentionBlock(nn.Module):
    # 🌟 增加 use_time_attn 和 use_var_attn
    def __init__(self, hidden_dim, num_heads=4, kernel_size=3,
                 use_temporal_smooth=True, use_time_attn=True, use_var_attn=True,
                 temporal_fusion_mode='gated_cascade'):
        super().__init__()
        self.use_temporal_smooth = use_temporal_smooth
        self.use_time_attn = use_time_attn
        self.use_var_attn = use_var_attn
        
        # 仅在启用时实例化，节省显存
        if self.use_time_attn:
            self.time_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            
        if self.use_var_attn:
            self.var_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)
            
        if self.use_temporal_smooth:
            self.temporal_smooth = MultiScaleTemporalBlock(
                hidden_dim, kernel_size=kernel_size, fusion_mode=temporal_fusion_mode
            )
        
    def forward(self, x, B_N, L, F_dim):
        # --- 1. Time Dimension ---
        x_time = x.transpose(1, 2).contiguous().view(B_N * F_dim, L, -1)
        if self.use_time_attn:
            attn_t, _ = self.time_attn(x_time, x_time, x_time, need_weights=False)
            x_time = self.norm1(x_time + attn_t)
            
        if self.use_temporal_smooth:
            x_time = self.temporal_smooth(x_time)
        
        # --- 2. Variable Dimension ---
        x_var = x_time.view(B_N, F_dim, L, -1).transpose(1, 2).contiguous().view(B_N * L, F_dim, -1)
        if self.use_var_attn:
            attn_v, _ = self.var_attn(x_var, x_var, x_var, need_weights=False)
            x_var = self.norm2(x_var + attn_v)
        
        out = x_var.view(B_N, L, F_dim, -1)
        return out

class PhysicsInformedReconstructor(nn.Module):
    # 🌟 同样把参数透传下来
    def __init__(self, num_features=5, hidden_dim=64, num_layers=3, use_era5=True, 
                 num_heads=4, kernel_size=3, feature_weights=None, 
                 use_temporal_smooth=True, use_time_attn=True, use_var_attn=True,
                 temporal_fusion_mode='gated_cascade'):
        super().__init__()
        self.num_features = num_features
        self.use_era5 = use_era5

        in_dim = 3 if use_era5 else 2
        self.feature_embed = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            DoubleAttentionBlock(hidden_dim, num_heads=num_heads, kernel_size=kernel_size, 
                                 use_temporal_smooth=use_temporal_smooth,
                                 use_time_attn=use_time_attn,
                                 use_var_attn=use_var_attn,
                                 temporal_fusion_mode=temporal_fusion_mode)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, 1)

        # 🌟 动态加载要素权重
        if feature_weights is None:
            feature_weights = [1.0] * num_features
        weights = torch.tensor(feature_weights, dtype=torch.float32).view(1, 1, 1, num_features)
        self.register_buffer('feature_weights', weights)

    def forward(self, masked_obs, mask, era5=None):
        B, N, L, F_dim = masked_obs.shape
        B_N = B * N
        x_obs = masked_obs.reshape(B_N, L, F_dim)
        x_mask = mask.reshape(B_N, L, F_dim)
        
        has_valid_era5 = self.use_era5 and (era5 is not None) and (era5.numel() > 0)
        
        if has_valid_era5:
            x_era5 = era5.reshape(B_N, L, F_dim)
            x_in = torch.stack([x_obs, x_mask, x_era5], dim=-1)
        else:
            x_in = torch.stack([x_obs, x_mask], dim=-1)
            
        h = self.feature_embed(x_in)
        
        for layer in self.layers:
            h = layer(h, B_N, L, F_dim)
            
        out = self.output_proj(h).squeeze(-1)
        out = out.reshape(B, N, L, F_dim)
        
        if has_valid_era5:
            return era5 + out
        else:
            return out

# Historical implementation retained for reference; it is not used by training.
# def physics_loss_function(pred, gt_obs, mask, feature_weights, alpha=0.3, beta=0.1, missing_penalty=2.0):
#     loss_time = F.smooth_l1_loss(pred, gt_obs, reduction='none', beta=beta)
#     missing_weight = 1.0 - mask
#     # missing_penalty 决定了对缺失区域额外乘上的倍数
#     weighted_time_loss = loss_time * feature_weights * (1.0 + missing_penalty * missing_weight)
    
#     pred_f32 = pred.float()
#     gt_obs_f32 = gt_obs.float()
    
#     fft_pred = torch.fft.rfft(pred_f32, dim=2)
#     fft_true = torch.fft.rfft(gt_obs_f32, dim=2)
    
#     loss_freq = torch.abs(torch.abs(fft_pred) - torch.abs(fft_true))
#     loss_freq_mean = loss_freq.mean(dim=2, keepdim=True).expand(-1, -1, pred.shape[2], -1)
#     weighted_freq_loss = loss_freq_mean * feature_weights * (1.0 + missing_penalty * missing_weight)
    
#     total_loss = weighted_time_loss.mean() + alpha * weighted_freq_loss.mean()
#     return total_loss


def physics_loss_function(
    pred,
    gt_obs,
    mask,
    feature_weights,
    beta=0.1,
    missing_penalty=2.0,
    temporal_weight=0.2,
    wind_vector_weight=0.05,
    delta_scales=None,
    wind_delta_scale=None,
    pointwise_loss='huber',
    temporal_feature_weights=None,
):
    """Noise-aware loss for normalized meteorological time series.

    Feature order is temperature, virtual temperature, u-wind, v-wind, and
    pressure. It combines robust pointwise reconstruction with first-order
    temporal innovation matching and joint u/v wind-vector evolution.

    ``delta_scales`` and ``wind_delta_scale`` are robust innovation scales
    estimated from the training split. They make the temporal terms comparable
    across regions with different sampling noise and natural variability.

    ``pointwise_loss`` controls only the reconstruction term. The default
    Huber loss is the full-model setting; MSE is used for its single-factor
    ablation. ``temporal_feature_weights`` controls the relative importance of
    features inside the temporal innovation term only.
    """
    if pred.shape != gt_obs.shape or pred.shape != mask.shape:
        raise ValueError('pred, gt_obs, and mask must have identical [B, N, L, F] shapes.')
    if pred.ndim != 4 or pred.shape[-1] != 5:
        raise ValueError('pred, gt_obs, and mask must have shape [B, N, L, 5].')
    if feature_weights.shape[-1] != pred.shape[-1]:
        raise ValueError('feature_weights must contain one weight per feature.')
    if beta <= 0:
        raise ValueError('beta must be positive.')
    if missing_penalty < 0:
        raise ValueError('missing_penalty must be non-negative.')
    if temporal_weight < 0 or wind_vector_weight < 0:
        raise ValueError('temporal_weight and wind_vector_weight must be non-negative.')
    if pointwise_loss not in {'huber', 'mse'}:
        raise ValueError("pointwise_loss must be either 'huber' or 'mse'.")

    # Keep reductions in float32: AMP output can be fp16 and the sum of
    # per-point weights over a meteorological batch can otherwise overflow.
    pred_f32 = pred.float()
    gt_obs_f32 = gt_obs.float()
    feature_weights = feature_weights.to(device=pred.device, dtype=torch.float32)
    mask = mask.to(device=pred.device, dtype=torch.float32)
    missing = 1.0 - mask

    if delta_scales is None:
        delta_scales = torch.ones(pred.shape[-1], device=pred.device, dtype=torch.float32)
    else:
        delta_scales = torch.as_tensor(delta_scales, device=pred.device, dtype=torch.float32)
        if delta_scales.numel() != pred.shape[-1]:
            raise ValueError('delta_scales must contain one positive scale per feature.')
        delta_scales = delta_scales.reshape(-1)
    if delta_scales.device.type == 'cpu' and torch.any(delta_scales <= 0):
        raise ValueError('delta_scales must be positive.')
    delta_scales = delta_scales.clamp_min(1e-6)

    if wind_delta_scale is None:
        wind_delta_scale = 1.0
    wind_delta_scale = torch.as_tensor(
        wind_delta_scale, device=pred.device, dtype=torch.float32
    ).clamp_min(1e-6)

    # The noise analysis found heavy-tailed residuals, particularly for wind
    # and pressure, so the full model uses a robust pointwise fit.
    if pointwise_loss == 'huber':
        pointwise_error = F.smooth_l1_loss(
            pred_f32, gt_obs_f32, reduction='none', beta=beta
        )
    else:
        pointwise_error = F.mse_loss(pred_f32, gt_obs_f32, reduction='none')
    pointwise_weights = feature_weights * (1.0 + missing_penalty * missing)
    pointwise_loss = (pointwise_error * pointwise_weights).sum()
    pointwise_loss = pointwise_loss / pointwise_weights.sum().clamp_min(1e-8)

    if pred.shape[2] < 2:
        return pointwise_loss

    pred_delta = pred_f32[:, :, 1:] - pred_f32[:, :, :-1]
    target_delta = gt_obs_f32[:, :, 1:] - gt_obs_f32[:, :, :-1]
    pair_missing = 1.0 - mask[:, :, 1:] * mask[:, :, :-1]
    delta_weights = feature_weights * (1.0 + missing_penalty * pair_missing)

    # Temperature-like variables and pressure have persistent, low-frequency
    # structure. Wind remains less constrained because its innovations are
    # comparatively intermittent and heavy-tailed.
    if temporal_feature_weights is None:
        temporal_feature_weights = [1.0, 1.0, 0.5, 0.5, 1.0]
    temporal_feature_weights = torch.as_tensor(
        temporal_feature_weights, device=pred.device, dtype=torch.float32
    )
    if temporal_feature_weights.numel() != pred.shape[-1]:
        raise ValueError('temporal_feature_weights must contain one weight per feature.')
    temporal_feature_weights = temporal_feature_weights.reshape(1, 1, 1, -1)
    normalized_delta_error = (pred_delta - target_delta) / delta_scales.view(1, 1, 1, -1)
    delta_error = F.smooth_l1_loss(
        normalized_delta_error, torch.zeros_like(normalized_delta_error), reduction='none', beta=beta
    )
    temporal_weights = delta_weights * temporal_feature_weights
    temporal_loss = (delta_error * temporal_weights).sum()
    temporal_loss = temporal_loss / temporal_weights.sum().clamp_min(1e-8)

    # Treat u and v as one vector when matching changes in wind, which avoids
    # biasing the model toward either Cartesian component.
    wind_delta_error = torch.linalg.vector_norm(
        pred_delta[..., 2:4] - target_delta[..., 2:4], dim=-1
    )
    wind_feature_weight = feature_weights[..., 2:4].mean(dim=-1)
    wind_pair_weights = wind_feature_weight * (
        1.0 + missing_penalty * pair_missing[..., 2:4].amax(dim=-1)
    )
    wind_loss = (F.smooth_l1_loss(
                    wind_delta_error / wind_delta_scale,
                    torch.zeros_like(wind_delta_error),
                    beta=beta,
                )
                 * wind_pair_weights).sum()
    wind_loss = wind_loss / wind_pair_weights.sum().clamp_min(1e-8)

    return pointwise_loss + temporal_weight * temporal_loss + wind_vector_weight * wind_loss





def vanilla_loss_function(pred, gt_obs, mask, missing_penalty=2.0):
    loss = F.mse_loss(pred, gt_obs, reduction='none')
    missing_weight = 1.0 - mask
    weighted_loss = loss * (1.0 + missing_penalty * missing_weight)
    return weighted_loss.mean()
