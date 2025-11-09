import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config.config import ModelConfig


class Conv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)
    
def precompute_freqs_cis(max_len: int, dim_head: int, device: str = 'cpu') -> torch.Tensor:
    freqs = torch.arange(0, dim_head // 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000 ** (freqs / (dim_head // 2)))
    freqs = freqs.unsqueeze(0).unsqueeze(0)

    positions = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    theta = positions * freqs

    freqs_cis = torch.polar(torch.ones_like(theta), theta)

    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, num_heads, dim_head = x.size()

    freqs_cis_reshaped = freqs_cis[:, :seq_len, :].unsqueeze(2).expand(-1, -1, num_heads, -1)

    real = x[..., :dim_head // 2]
    imag = x[..., dim_head // 2:]

    x_complex = torch.view_as_complex(torch.stack((real, imag), dim=-1))

    x_rotated_complex = x_complex * freqs_cis_reshaped

    x_rotated_real = x_rotated_complex.real
    x_rotated_imag = x_rotated_complex.imag

    return torch.cat((x_rotated_real, x_rotated_imag), dim=-1)
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.dim_model % config.num_heads == 0, "dim_model must be divisible by num_heads"
        
        self.num_heads = config.num_heads
        self.dim_model = config.dim_model
        self.dim_head = config.dim_head
        self.scale = self.dim_head ** -0.5

        self.qkv_proj = nn.Linear(config.dim_model, config.dim_model * 3)
        self.out_proj = nn.Linear(config.dim_model, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.size()
        
        xqkv = self.qkv_proj(x).chunk(3, dim=-1)
        xq, xk, xv = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2), xqkv)

        q = apply_rotary_emb(xq, freqs_cis)
        k = apply_rotary_emb(xk, freqs_cis)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        scores = F.softmax(scores, dim=-1)

        assert not torch.isnan(scores).any(), "scores has NaN"
        assert not torch.isinf(scores).any(), "scores has Inf"

        scores = self.dropout(scores)
        
        out = torch.matmul(scores, xv)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim_model)
        
        return self.out_proj(out)
    
class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_ff),
            nn.GELU(),
            nn.Linear(config.dim_ff, config.dim_model)
        )

        self.input_layernorm = RMSNorm(config.dim_model, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(config.dim_model, eps=1e-6)

        self.peri_norm1 = RMSNorm(config.dim_model, eps=1e-6)
        self.peri_norm2 = RMSNorm(config.dim_model, eps=1e-6)
        
        self.residual_multiplier = 1.0

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor = None):
        residual = x
        norm_x_for_attention = self.input_layernorm(x)

        attn_out = self.attention(norm_x_for_attention, freqs_cis=freqs_cis, mask=mask)
        attn_out = self.peri_norm1(attn_out)
        
        x = residual + self.dropout1(attn_out) * self.residual_multiplier
        residual = x
        
        norm_x_for_mlp = self.post_attention_layernorm(x)
        ff_out = self.feed_forward(norm_x_for_mlp)
        ff_out = self.peri_norm2(ff_out)
            
        x = residual + self.dropout2(ff_out) * self.residual_multiplier
        
        return x
    

class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.max_len = config.window_size * config.input_feature_variants
        self.dim_head = config.dim_head
        self.input_size = config.input_size
        self.layer_count = config.num_layers

        self.conv1d = Conv1d(config.input_size, config.dim_model, kernel_size=3, padding=1)

        self.embedding_dropout = nn.Dropout(config.dropout)

        self.layer = TransformerLayer(config)

        self.norm = RMSNorm(config.dim_model, eps=1e-6)
        self.output_projection = nn.Linear(config.dim_model, max(config.future_offset))

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
        
    def forward(self, x, freqs_cis=None, attention_mask=None):
        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis(self.max_len, self.dim_head, device=x.device)

        x = self.conv1d(x)

        x = self.embedding_dropout(x)

        for i in range(self.layer_count):
            x = self.layer(x, freqs_cis=freqs_cis, mask=attention_mask)

        x = self.norm(x)
        last_hidden = x[:, -1, :]
        log_return = self.output_projection(last_hidden).squeeze(-1)

        predicted_prices = torch.exp(log_return)
        predicted_prices = torch.clamp(predicted_prices, min=0.3, max=2.0)

        return predicted_prices

def create_attention_mask(input, pad_token_id=0):
    B, T, D = input.shape

    is_pad = (input == pad_token_id).all(dim=-1)
    padding_mask = is_pad.unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones((T, T), dtype=torch.bool, device=input.device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    combined_mask = torch.logical_or(padding_mask, causal_mask)

    return combined_mask