"""
Modern Mixture-of-Experts (MoE) Layer for BART Encoder
符合 Switch Transformer / GLaM / Expert Choice 最佳实践

主要特性:
- Top-k 稀疏激活 (k=1 或 2)
- 负载均衡损失 (load balancing auxiliary loss)
- 温度控制的路由器 (temperature-scaled routing)
- 可选噪声注入 (noisy gating for exploration)
- 标准 FFN 专家 (无退化专家如 Copy/Constant)
- 角色感知路由 (role-aware routing for dialogue)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from collections import defaultdict


class FFNExpert(nn.Module):
    """
    标准 Feed-Forward Network 专家
    结构: Linear -> Activation -> Dropout -> Linear
    """
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
        
        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 初始化权重 (BART-style)
        self._init_weights()
    
    def _init_weights(self):
        """BART 风格的权重初始化"""
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] or [num_tokens, d_model]
        Returns:
            output: same shape as x
        """
        h = self.w1(x)
        h = self.activation(h)
        h = self.dropout(h)
        output = self.w2(h)
        return output


class TopKRouter(nn.Module):
    """
    Top-K 稀疏路由器，支持负载均衡和温度控制
    
    Features:
    - Top-k gating (sparse activation)
    - Temperature scaling
    - Optional noise injection (for exploration)
    - Load balancing statistics
    - Optional role-aware routing
    """
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
        self.top_k = min(top_k, num_experts)  # 确保 k <= num_experts
        self.temperature = temperature
        self.noise_std = noise_std
        self.use_role_aware = use_role_aware
        
        # 路由器投影层
        router_input_dim = d_model + role_embed_dim if use_role_aware else d_model
        self.gate = nn.Linear(router_input_dim, num_experts, bias=False)
        
        # Role embedding (如果启用)
        if use_role_aware:
            self.role_embedding = nn.Embedding(10, role_embed_dim)  # 假设最多10个角色
        
        # 初始化：接近均匀分布
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [num_tokens, d_model]
            role_ids: [num_tokens] (optional, for role-aware routing)
        
        Returns:
            expert_indices: [num_tokens, top_k] - 选中的专家 ID
            expert_weights: [num_tokens, top_k] - 对应的权重 (已归一化)
            router_logits: [num_tokens, num_experts] - 原始 logits (用于计算 aux loss)
            load_balancing_loss: scalar - 负载均衡损失
        """
        num_tokens = hidden_states.size(0)
        
        # 角色感知 (如果启用)
        if self.use_role_aware and role_ids is not None:
            role_embeds = self.role_embedding(role_ids)  # [num_tokens, role_embed_dim]
            router_input = torch.cat([hidden_states, role_embeds], dim=-1)
        else:
            router_input = hidden_states
        
        # 计算路由 logits (在 FP32 下进行，避免数值不稳定)
        # CPU 训练时不使用 autocast
        if torch.cuda.is_available():
            ctx_manager = torch.cuda.amp.autocast(enabled=False)
        else:
            # CPU 上使用 dummy context manager
            from contextlib import nullcontext
            ctx_manager = nullcontext()
        
        with ctx_manager:
            router_input_fp32 = router_input.float()
            router_logits = self.gate(router_input_fp32)  # [num_tokens, num_experts]
            
            # 温度缩放
            router_logits = router_logits / self.temperature
            
            # 训练时加噪声 (for exploration)
            if self.training and self.noise_std > 0:
                noise = torch.randn_like(router_logits) * self.noise_std
                router_logits = router_logits + noise
            
            # Softmax 得到概率分布
            router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]
            
            # Top-k 选择
            expert_weights, expert_indices = torch.topk(
                router_probs, k=self.top_k, dim=-1
            )  # both: [num_tokens, top_k]
            
            # 归一化 top-k 权重
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
            
            # 计算负载均衡损失
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
        """
        计算负载均衡损失 (Switch Transformer style)
        
        目标: 让每个专家的"重要度"和"负载"接近均匀
        
        Args:
            router_probs: [num_tokens, num_experts]
            expert_indices: [num_tokens, top_k]
            num_tokens: int
        
        Returns:
            loss: scalar
        """
        # 重要度 (importance): 每个专家的平均概率
        importance = router_probs.mean(dim=0)  # [num_experts]
        
        # 负载 (load): 每个专家被选中的 token 比例
        load = torch.zeros(
            self.num_experts, dtype=router_probs.dtype, device=router_probs.device
        )
        for i in range(self.num_experts):
            load[i] = (expert_indices == i).sum().float() / (num_tokens * self.top_k)
        
        # 均衡目标: 每个专家应该有 1/num_experts 的负载
        target = 1.0 / self.num_experts
        
        # 损失: CV^2 (Coefficient of Variation squared) 或简单的 MSE
        # 这里使用 importance * load 的方差（Switch Transformer 风格）
        loss = self.num_experts * (importance * load).sum()
        
        return loss


class MoELayer(nn.Module):
    """
    完整的 MoE 层，包含多个专家和路由器
    
    替换 BART EncoderLayer 中的 FFN
    """
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
        
        # 创建专家 (所有专家结构相同，参数独立)
        self.experts = nn.ModuleList([
            FFNExpert(d_model, d_ff, activation, dropout)
            for _ in range(num_experts)
        ])
        
        # 创建路由器
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
            aux_loss: scalar (负载均衡损失)
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
        
        # 路由决策
        expert_indices, expert_weights, router_logits, aux_loss = self.router(
            hidden_states_flat, role_ids_flat
        )
        # expert_indices: [num_tokens, top_k]
        # expert_weights: [num_tokens, top_k]
        
        # 初始化输出
        output = torch.zeros_like(hidden_states_flat)
        
        # 按专家分组处理 (提高效率)
        for expert_id in range(self.num_experts):
            # 找到路由到该专家的所有 token
            expert_mask = (expert_indices == expert_id)  # [num_tokens, top_k]
            token_indices, k_indices = torch.where(expert_mask)
            
            if token_indices.numel() == 0:
                continue  # 该专家未被选中
            
            # 获取这些 token 的输入和权重
            expert_input = hidden_states_flat[token_indices]  # [num_selected, d_model]
            expert_weight = expert_weights[token_indices, k_indices].unsqueeze(-1)  # [num_selected, 1]
            
            # 专家前向计算
            expert_output = self.experts[expert_id](expert_input)  # [num_selected, d_model]
            
            # 加权并累加到输出
            weighted_output = expert_output * expert_weight
            output.index_add_(0, token_indices, weighted_output)
        
        # Reshape 回原始形状
        output = output.reshape(original_shape)
        
        return output, aux_loss


class BARTMoEEncoderLayer(nn.Module):
    """
    BART Encoder Layer with MoE-FFN
    
    替换原来的 BartEncoderLayer，将 FFN 部分改为 MoE
    """
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
        
        # Self-Attention (保持不变)
        from transformers.models.bart.modeling_bart import BartAttention
        self.self_attn = BartAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        
        # MoE-FFN (替换原 FFN)
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
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, 1, tgt_len, src_len]
            layer_head_mask: [num_heads]
            role_ids: [batch, seq_len] (optional)
            output_attentions: bool
        
        Returns:
            hidden_states: [batch, seq_len, d_model]
            attn_weights: [batch, num_heads, seq_len, seq_len] (if output_attentions)
            moe_aux_loss: scalar
        """
        # Self-Attention
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
        
        # MoE-FFN
        residual = hidden_states
        hidden_states, moe_aux_loss = self.moe_layer(hidden_states, role_ids)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        # 数值稳定性检查 (FP16 保护)
        if hidden_states.dtype == torch.float16:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        outputs += (moe_aux_loss,)
        
        return outputs


# 辅助函数：用于集成到现有代码
def create_moe_layer(
    d_model: int,
    d_ff: int,
    num_experts: int = 4,
    top_k: int = 2,
    **kwargs
) -> MoELayer:
    """
    工厂函数：创建 MoE 层
    
    使用示例:
        moe = create_moe_layer(d_model=768, d_ff=3072, num_experts=4, top_k=2)
        output, aux_loss = moe(hidden_states)
    """
    return MoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )

