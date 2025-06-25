import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwiGLUMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        ffn_hidden_size = int(2 * hidden_features / 3)
        self.gate_proj = nn.Linear(in_features, ffn_hidden_size, bias=bias, dtype=dtype)
        self.up_proj = nn.Linear(in_features, ffn_hidden_size, bias=bias, dtype=dtype)
        self.down_proj = nn.Linear(ffn_hidden_size, out_features, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

class LocalProbabilityTransform(nn.Module):
    """
    A simple adaptive sampler that applies a learned transformation to each
    log-probability value independently. It also applies a learned soft truncation.

    This model can learn simple sampling strategies like temperature scaling,
    epsilon sampling, or polynomial transformations of log-probabilities.
    """
    def __init__(self, hidden_dims: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.mlp = SwiGLUMLP(
            in_features=1,
            hidden_features=hidden_dims,
            out_features=1,
            bias=True,
            dtype=dtype
        )
        self.truncation_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.truncation_threshold = nn.Parameter(torch.tensor(10.0, dtype=dtype))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logit_dtype = logits.dtype
        logits = logits.to(self.dtype)
        log_probs = F.log_softmax(logits, dim=-1)
        modification = self.mlp(log_probs.unsqueeze(-1)).squeeze(-1)
        modified_log_probs = log_probs + modification
        truncation_penalty = F.logsigmoid(modified_log_probs * self.truncation_scale + self.truncation_threshold)
        final_log_probs = modified_log_probs + truncation_penalty
        return final_log_probs.to(logit_dtype)

class SimpleDistributionAwareTransform(nn.Module):
    """
    An adaptive sampler that uses both local log-probability information and global
    distribution statistics (max log-prob, entropy) to modify the logits.

    This allows the model to make decisions based on the overall entropy/shape of the
    distribution, enabling it to approximate sampling strategies like min-p and eta sampling
    in addition to sampling strategies enabled by LocalProbabilityTransform.

    If enabled, dynamic truncation learns truncation parameters from the distribution,
    otherwise something static.
    """
    def __init__(self, hidden_dims: int, use_dynamic_truncation: bool, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.local_mlp = nn.Sequential(
            nn.Linear(1, hidden_dims, dtype=dtype),
            nn.SiLU()
        )
        self.n_global_stats = 2
        self.global_mlp = SwiGLUMLP(
            in_features=hidden_dims + self.n_global_stats,
            hidden_features=hidden_dims,
            out_features=1,
            bias=True,
            dtype=dtype
        )
        self.use_dynamic_truncation = use_dynamic_truncation
        if self.use_dynamic_truncation:
            self.truncation_head = nn.Linear(self.n_global_stats, 2, dtype=dtype)
        else:
            self.truncation_params = nn.Parameter(torch.tensor([0.5413, 10.0], dtype=dtype))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits_dtype = logits.dtype
        logits = logits.to(self.dtype)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            log_probs_max = torch.max(log_probs, dim=-1, keepdim=True)[0]
            probs = torch.exp(log_probs)
            log_probs_entropy = torch.sum(-probs * log_probs, dim=-1, keepdim=True)
            global_stats = torch.cat([log_probs_max, log_probs_entropy], dim=-1)

        local_features = self.local_mlp(log_probs.unsqueeze(-1))
        
        global_features_broadcast = global_stats.unsqueeze(-2).expand(*local_features.shape[:-1], -1)
        
        combined_features = torch.cat([local_features, global_features_broadcast], dim=-1)
        modification = self.global_mlp(combined_features).squeeze(-1)
        
        modified_log_probs = log_probs + modification

        if self.use_dynamic_truncation:
            truncation_params = self.truncation_head(global_stats)
            scale = F.softplus(truncation_params[..., 0:1])
            threshold = F.softplus(truncation_params[..., 1:2])
        else:
            scale = F.softplus(self.truncation_params[0])
            threshold = F.softplus(self.truncation_params[1])
            
        truncation_penalty = F.logsigmoid(modified_log_probs * scale + threshold)
        final_log_probs = modified_log_probs + truncation_penalty

        return final_log_probs.to(logits_dtype)

class ElementwiseEncoder(nn.Module):
    """
    Encodes each scalar log-probability into a d_model vector.
    """
    def __init__(self, d_model: int, dtype: torch.dtype):
        super().__init__()
        self.mlp = SwiGLUMLP(in_features=1, hidden_features=d_model, out_features=d_model, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.unsqueeze(-1)).type_as(x)

class LinearAttention(nn.Module):
    """
    Implements Linear Attention with elu(x) + 1 as the feature map.
    Non-causal linear attention is used in order to deal with large vocabularies.
    """
    def __init__(self, d_model: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.d_model = d_model

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1

        kv_context = torch.matmul(k_prime.transpose(-2, -1), v)
        numerator = torch.matmul(q_prime, kv_context)

        k_sum = k_prime.sum(dim=-2, keepdim=True)
        denominator = torch.matmul(q_prime, k_sum.transpose(-2, -1))

        return numerator / (denominator + torch.tensor(1e-6, dtype=self.dtype))

class Block(nn.Module):
    """
    A Transformer-style block using Linear Attention.
    """
    def __init__(self, d_model: int, d_ff: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, dtype=dtype)
        self.attention = LinearAttention(d_model, dtype=dtype)
        self.ffn = SwiGLUMLP(in_features=d_model, hidden_features=d_ff, out_features=d_model, bias=True, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        x = self.norm1(x + self.attention(q, k, v))
        x = self.norm2(x + self.ffn(x))
        return x

class SamplingNetwork(nn.Module):
    """
    An adaptive sampler using a Transformer-like architecture with Linear Attention
    to model vocabulary-wide interactions.

    This model is designed to be expressive enough to learn complex sampling
    strategies that depend on the relationships between all tokens in the vocabulary,
    such as top-k, top-p and typical sampling, in addition to sampling strategies
    learned by the SimpleDistributionAwareTransform.

    This architecture is permutation-equivariant in order to not learn any knowledge
    about specific tokens, but to learn how to transform the probability distribution itself.
    """
    def __init__(self, d_model: int, d_ff: int, num_blocks: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.d_model = d_model
        self.encoder = ElementwiseEncoder(d_model, dtype=dtype)
        self.blocks = nn.Sequential(*[Block(d_model, d_ff, dtype=dtype) for _ in range(num_blocks)])
        self.transformation_head = SwiGLUMLP(in_features=d_model * 2, hidden_features=d_model, out_features=1, bias=True, dtype=dtype)
        self.truncation_head = SwiGLUMLP(in_features=d_model, hidden_features=d_model // 2, out_features=2, bias=True, dtype=dtype)

    def forward(self, logits: torch.Tensor):
        original_dtype = logits.dtype
        logits = logits.to(self.dtype)
        original_shape = logits.shape
        V = original_shape[-1]
        
        log_probs_flat = F.log_softmax(logits.view(-1, V), dim=-1)
        
        h0 = self.encoder(log_probs_flat)
        h_final = self.blocks(h0)
        
        c = torch.mean(h_final, dim=1)
        c_broadcast = c.unsqueeze(1).expand(-1, V, self.d_model)
        
        l_prime_mod = self.transformation_head(torch.cat([h_final, c_broadcast], dim=-1)).squeeze(-1)
        l_prime = log_probs_flat + l_prime_mod
        
        truncation_params = self.truncation_head(c)
        scale = F.softplus(truncation_params[..., 0:1])
        threshold = F.softplus(truncation_params[..., 1:2])
        
        truncation_penalty = F.logsigmoid(l_prime * scale + threshold)
        final_log_probs_flat = l_prime + truncation_penalty
        
        final_log_probs = final_log_probs_flat.view(*original_shape)
        return final_log_probs.to(original_dtype)
