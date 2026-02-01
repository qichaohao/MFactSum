import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from collections import defaultdict


class FFNExpert(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        h = self.activation(h)
        h = self.dropout(h)
        output = self.w2(h)
        return output


class TopKRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
        noise_std: float = 0.0,
        use_role_aware: bool = False,
        role_embed_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  
        self.temperature = temperature
        self.noise_std = noise_std
        self.use_role_aware = use_role_aware
        router_input_dim = d_model + role_embed_dim if use_role_aware else d_model
        self.gate = nn.Linear(router_input_dim, num_experts, bias=False)
        
        if use_role_aware:
            self.role_embedding = nn.Embedding(10, role_embed_dim)  
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.size(0)
        
        if self.use_role_aware and role_ids is not None:
            role_embeds = self.role_embedding(role_ids)  # [num_tokens, role_embed_dim]
            router_input = torch.cat([hidden_states, role_embeds], dim=-1)
        else:
            router_input = hidden_states
        
        if torch.cuda.is_available():
            ctx_manager = torch.cuda.amp.autocast(enabled=False)
        else:
            from contextlib import nullcontext
            ctx_manager = nullcontext()
        
        with ctx_manager:
            router_input_fp32 = router_input.float()
            router_logits = self.gate(router_input_fp32)  # [num_tokens, num_experts]
            router_logits = router_logits / self.temperature
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(router_logits) * self.noise_std
                router_logits = router_logits + noise
            router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]
            expert_weights, expert_indices = torch.topk(
                router_probs, k=self.top_k, dim=-1
            )  
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
            load_balancing_loss = self._compute_load_balancing_loss(
                router_probs, expert_indices, num_tokens
            )
        
        return expert_indices, expert_weights, router_logits, load_balancing_loss
    
    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:

        importance = router_probs.mean(dim=0)  # [num_experts]

        load = torch.zeros(
            self.num_experts, dtype=router_probs.dtype, device=router_probs.device
        )
        for i in range(self.num_experts):
            load[i] = (expert_indices == i).sum().float() / (num_tokens * self.top_k)
        target = 1.0 / self.num_experts
        loss = self.num_experts * (importance * load).sum()
        return loss


class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        temperature: float = 1.0,
        noise_std: float = 0.0,
        use_role_aware: bool = False,
        role_embed_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            FFNExpert(d_model, d_ff, activation, dropout)
            for _ in range(num_experts)
        ])

        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature,
            noise_std=noise_std,
            use_role_aware=use_role_aware,
            role_embed_dim=role_embed_dim,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            role_ids: [batch_size, seq_len] (optional)
        
        Returns:
            output: [batch_size, seq_len, d_model]
            aux_loss: scalar 
        """
        original_shape = hidden_states.shape
        batch_size, seq_len, d_model = original_shape
        
        # Flatten to [num_tokens, d_model]
        hidden_states_flat = hidden_states.reshape(-1, d_model)
        num_tokens = hidden_states_flat.size(0)
        
        # Flatten role_ids if provided
        role_ids_flat = None
        if role_ids is not None:
            role_ids_flat = role_ids.reshape(-1)

        expert_indices, expert_weights, router_logits, aux_loss = self.router(
            hidden_states_flat, role_ids_flat
        )
        output = torch.zeros_like(hidden_states_flat)
        for expert_id in range(self.num_experts):
            expert_mask = (expert_indices == expert_id)  # [num_tokens, top_k]
            token_indices, k_indices = torch.where(expert_mask)
            if token_indices.numel() == 0:
                continue  
            expert_input = hidden_states_flat[token_indices]  # [num_selected, d_model]
            expert_weight = expert_weights[token_indices, k_indices].unsqueeze(-1)  # [num_selected, 1]
            expert_output = self.experts[expert_id](expert_input)  # [num_selected, d_model]
            weighted_output = expert_output * expert_weight
            output.index_add_(0, token_indices, weighted_output)
        output = output.reshape(original_shape)
        return output, aux_loss


class BARTMoEEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        moe_temperature: float = 1.0,
        moe_noise_std: float = 0.0,
        use_role_aware: bool = False,
    ):
        super().__init__()
        
        from transformers.models.bart.modeling_bart import BartAttention
        self.self_attn = BartAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.moe_layer = MoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            dropout=dropout,
            temperature=moe_temperature,
            noise_std=moe_noise_std,
            use_role_aware=use_role_aware,
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:

        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states, moe_aux_loss = self.moe_layer(hidden_states, role_ids)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if hidden_states.dtype == torch.float16:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        outputs += (moe_aux_loss,)
        return outputs

def create_moe_layer(
    d_model: int,
    d_ff: int,
    num_experts: int = 4,
    top_k: int = 2,
    **kwargs
) -> MoELayer:
    return MoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )


