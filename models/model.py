import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    # float为了在计算时保持更高的精度，防止数值下溢或溢出，尤其是在使用半精度（如float16）时。完成计算后再转换回原类型，既保证了计算的稳定性，又保持了模型的数据类型
    # self.eps**：加上一个很小的常数eps，避免分母为零的情况，增加数值稳定性。这类似于在归一化中的epsilon，比如LayerNorm中的eps。
    # torch.rsqrt(...)**：计算上述结果的平方根的倒数。也就是1 / sqrt(mean + eps)。rsqrt是倒数平方根函数，常用于优化计算，因为它可能比先计算平方根再取倒数更快。
    # .type_as(x)**：将结果转换回x的数据类型，比如如果x是half精度（float16），则转换回来，以保持数据类型的一致性。
    # self.weight * ...**：最后乘以一个可学习的权重参数，这个权重应该是一个可训练的参数，形状与归一化后的特征维度相匹配。这允许模型学习缩放归一化后的特征。
    # 这段代码实现了一种归一化层，类似于RMSNorm（Root Mean Square Layer Normalization）。RMSNorm与LayerNorm的不同之处在于，它仅使用均方根来归一化，而不减去均值。LayerNorm通常会减去均值并除以标准差，而这里只除以均方根（RMS），即平方的平均值的平方根，再加上epsilon。
    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)



# 得到旋转emb编码（Rotary Position Embedding，RoPE）
# RoPE通过将查询（query）和键（key）向量进行旋转操作来融入位置信息。这种方法的优点是不需要额外的参数，且能保持相对位置的关系。通常，RoPE会将位置信息编码为复数形式的旋转矩阵，然后应用到查询和键向量上。
# pos_cis.shape  seq_len,head_dim//2
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        # 得到维度数
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # 保证维度 (seq_len, head_dim//2)
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        # 调整 pos_cis 的形状，使其与输入 x 的维度对齐，便于广播（Broadcasting）。
        return pos_cis.view(*shape)
    # 将最后一维（如 head_dim）拆分为 (head_dim//2, 2)，每组两个元素分别作为复数的实部和虚部。
    # 复数转换：通过 torch.view_as_complex 将形状为 (..., head_dim//2, 2) 的张量转换为复数形式 (..., head_dim//2)。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # (1, seq_len, 1, head_dim//2)
    pos_cis = unite_shape(pos_cis, xq_)
    # 复数乘法等价于对特征向量进行旋转，编码位置信息。
    # flatten(dim)表示，从第dim个维度开始展开，将后面的维度转化为一维.也就是说，只保留dim之前的维度，其他维度的数据全都挤在dim这一维。
    # 通过 torch.view_as_real 将复数结果恢复为实数形式，形状变为 (..., head_dim//2, 2)。
    # 展平维度：flatten(3) 将最后两维合并，恢复为原始特征维度 head_dim
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# 对应n_heads和n_kv_heads，将n_kv_heads扩展为n_heads，让让多个查询头共享同一组键值头
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # 在第四个维度插入了`None`，也就是增加了一个新的维度，位置在第三维之后。这会改变张量的形状，从原来的四维变成五维，形状变为 `(bs, slen, n_kv_heads, 1, head_dim)`。
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# GQA
class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()

        # 为什么设置n_kv_heads；
        # 通过 减少键值头的数量（n_kv_heads < n_heads），让多个查询头共享同一组键值头，从而降低计算量。
        # 若 n_heads=8，n_kv_heads=2，则每4个查询头共享1个键值头。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        
        # 填充全为-inf
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        # 上三角的对角线以上不变，其他全为0
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)


    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                # Optional[...]：表示该参数可以是 None 或指定类型
                # Tuple[torch.Tensor, torch.Tensor]：一个包含两个 PyTorch 张量的元组，分别对应 历史 Key 矩阵 和 历史 Value 矩阵。
                # 首次调用：past_key_value=None，模型计算所有 token 的 Key 和 Value，并返回缓存。
                # 后续调用：将之前返回的 past_key_value 传入，模型仅计算新 token 的 Key 和 Value，与历史缓存拼接后生成注意力结果。
                # 更新缓存：新的 Key 和 Value 会被追加到 past_key_value 中，供下一步使用
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        # batch_size seq_len
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 得到旋转emb编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        
        # kv_cache实现
        if past_key_value is not None:
            # b,s,head,head_dim--->b,s*layers,head,head_dim
            # kv cache考虑以前多个seq的kv，所以dim=1
            # 在q*k*v之后的shape还是不变
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            # b,s,head,head_dim
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 序列长度=1 时：无因果掩码需求（仅一个 token，无未来信息）。
        # Flash Attention 的分块优化无法发挥优势，甚至可能因额外调度降低效率。
        if self.flash and seq_len != 1:
           
            dropout_p = self.dropout if self.training else 0.
            # 实现了高效的 因果自注意力（Causal Self-Attention），利用 PyTorch 的 scaled_dot_product_attention 函数并结合 Flash Attention 优化技术
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 加一个无穷小数
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            # 参考Transformer架构，前馈网络的隐藏层通常为输入维度的4倍
            hidden_dim = 4 * config.dim
            # 缩放
            hidden_dim = int(2 * hidden_dim / 3)
            # 向上取整，得到multiple_of的整数倍
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SiLU（Sigmoid Linear Unit）SiLU函数在接近零时具有更平滑的曲线，并且由于其使用了sigmoid函数，可以使网络的输出范围在0和1之间
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        # 激活函数
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        # a代表LeakyReLU的负值斜率。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        # token级别处理，因此b*s
        hidden_states = hidden_states.view(-1, h)# (b*s, hidden_dim)

        # 等价于线性层变化, X*W.T+b
        logits = F.linear(hidden_states, self.weight, None) # logits.size=(b*s,num_experts)
        
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # value,index
        # 如果为 True，返回的元素按从大到小排序；如果为 False，返回的元素不排序，速度更快。
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) 
        # topk_weight与topk_idx的shape都是(b*s,top_k)

        # 归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator # 让概率和为1

        # 负载均衡
        # 负载均衡目标
        # 问题：MoE 中可能出现某些专家被过度选择（“赢家通吃”），而其他专家未被充分利用。
        # 通过 辅助损失（Auxiliary Loss） 约束专家选择的分布，使其更均匀。
        if self.training and self.alpha > 0.0: # 辅助损失函数，用于负载均衡
            scores_for_aux = scores
            aux_topk = self.top_k

            # (batch_size, seq_len*topk)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # token序列级负载均衡
            # 每个批次每个专家求期望
            if self.seq_aux:
                # scores_for_seq_aux： (batch_size, seq_len, num_experts)
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # (batch_size, num_experts)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                
                # 对ce的每行的所有列累加
                # (batch_size, num_experts)
                # 即使 seq_len * topk 大于 num_experts，topk_idx_for_aux_loss 中的索引可以重复。
                # 在多个时间步或 token 上，可能会选择相同的类别索引
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                # (batch_size, seq_len*topk)
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device))
                # 将选中次数转换为期望频率
                # (batch_size, num_experts)
                ce.div_(seq_len * aux_topk / self.n_routed_experts)
                # 两个 b,experts 相乘，最后求均值
                # 计算batch的均值
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                
            # 样本级负载均衡
            # 对每个专家求均值期望
            else:        
                # mask_ce：形状为 (batch_size * seq_len * top_k, num_experts)，每个位置标记是否选中对应专家。
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # ce：沿批次和序列维度求平均，得到每个专家被选中的概率，形状 (num_experts,)，对所有行求
                ce = mask_ce.float().mean(0)
                # (b*s,num_experts)--->(num_experts,)
                # 得到期望
                fi = ce * self.n_routed_experts
                
                Pi = scores_for_aux.mean(0)
                # 损失项为专家分数均值与期望频率的点积，鼓励高分专家被频繁选中：
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        # 创建experts个feedforward
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # moe gate
        self.gate = MoEGate(config)

        # 创建一个shared_experts
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # topk_weight与topk_idx的shape都是(b*s,top_k)
        x = x.view(-1, x.shape[-1]) # (b*s,dim)
        flat_topk_idx = topk_idx.view(-1) # (b*s*top_k,)
        if self.training:
            # 训练模式下，重复输入数据；因为每b*s有tok个索引，x每b*s只有一个，因此，重复toK次让每个索引都能选中
            # repeat=self.config.num_experts_per_tok，则对应维度0，每个元素重复tok次
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)# (b*s*top_k,dim)
            y = torch.empty_like(x, dtype=torch.float16)# (b*s*top_k,dim)
            for i, expert in enumerate(self.experts):
                # flat_topk_idx == i表示取出当前专家所负责的token的索引（数量不固定）
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # 乘以权重，求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # b,seq,dim
            y = y.view(*orig_shape)
        else:
            # 推理模式
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity) # 共享专家就是一个FFN
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # x.shape = (b*s,dim) flat_expert_indices.shape=flat_expert_weights.shape=(b*s*top_k,)
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort() 
        # idxs是将flat_expert_indices由小到大排序后得到的索引数组，flat_expert_indices[idxs[0]]是flat_expert_indices中最小的元素，
        # 也就是将各个token按照所负责的专家号排列 idxs=[专家1负责的token索引序列，专家2负责的token索引序列，...，专家n负责的token索引序列] 其中每个专家负责的token索引序列的长度可能不一样
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # flat_expert_indices.bincount()得到的结果就是每个专家所负责的token的数量,第i个位置表示专家i出现的次数
        # flat_expert_indices.bincount()=[专家1负责的token数量，专家2负责的token数量，...，专家n负责的token数量]
        # cumsum(0)操作后就是每个专家所负责的token的下标范围, 累加操作

        # idxs=(b*s*top_k,)，每个都是索引
        # 每个 token 被复制 top_k 次，因此通过整除操作可还原原始索引
        # 第i个元素代表第i个token
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]

            # 取出的是token索引；token_idxs每个元素是token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出expert i对应的token索引
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            
            # weight.shape start_idx:end_idx,1
            # idxs[start_idx:end_idx]取出的是当前expert 在weight的索引位置
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作 expert_cache.shape=(b*s,dim)
            # exp_token_idx的每个元素都是token的序号，转变后每一行都是同一个token的序号
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
            # index=exp_token_idx.view(-1, 1).repeat(1, x.shape[-1])
            # expert_cache[index,:]+=expert_out 将结果加到对应的位置 
            # 使用 scatter_add_是因为有些位置可能不止被访问一次，保证索引操作不出错，实际上由于b*s个token每个token都有top_k个索引，所以b*s个位置每个位置都会被加top_k

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        # 残差跳连
        h = x + h_attn
        # # 残差跳连
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


# 计算pos
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):

    # torch.arange(0, dim, 2)[: (dim // 2)] 仅取前dim // 2个
    # 取倒数，得到频率向量freqs。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 时间向量 t:
    t = torch.arange(end, device=freqs.device)  # type: ignore

    # 计算t和freqs的外积，得到一个形状为(end, dim//2)的矩阵
    # t @ freqs.T
    freqs = torch.outer(t, freqs).float()  # type: ignore

    # 将全1矩阵和freqs转换为复数形式，其中全1矩阵表示模长，freqs表示角度（弧度制）。
    # pos_cis是一个复数矩阵，形状为(end, dim//2)，数据类型为complex64。
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)


        # 嵌入曾和输出层共享参数
        self.tok_embeddings.weight = self.output.weight


        # 用于预计算旋转位置编码的复数形式（通常是 (cosθ, sinθ) 对）。
        # dim (int): 表示嵌入维度（embedding dimension），即每个位置的编码向量的长度。
        # theta：旋转基的超参数，控制位置编码的频率分布（例如 10000.0）。
        # persistent=False
        # 表示该缓冲区 不会保存到模型的状态字典（state_dict） 中。
        # 模型保存时，pos_cis 不会被持久化（节省存储空间）。
        # 加载模型时，需确保 pos_cis 能重新生成（例如在推理时重新计算或动态加载）。
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        
        # 用于 封装因果语言模型（Causal Language Model）的输出。这是 Hugging Face Transformers 库中的标准输出类，包含模型生成结果及中间状态。
        # 内存复用：在模型前向传播时，直接更新 self.OUT 的属性（如 logits 和 past_key_values），避免每次前向传播创建新对象，减少内存分配开销。
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)

        # shape  b,seq,dim
        h = self.dropout(self.tok_embeddings(input_ids))

        # 取出输入序列的当前长度对应的位置编码
        # shape  seq,head_dim
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        # 每个layer是MiniMindBlock
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))

        # 计算moe的aux_loss
        # feed_forward得到前馈层
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))

        # 本质是一个有序字典，通过 __setitem__ 方法设置键值对时，会自动映射到对象的属性。
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    # torch.inference_mode用于进一步优化推理过程。它不仅禁用梯度计算，还对推理过程进行了一些额外的优化，以提高性能。
    # 禁用梯度计算：与 torch.no_grad() 一样，torch.inference_mode() 也禁用梯度计算。
    # 优化推理性能：torch.inference_mode() 进行了额外的优化，使推理过程更加高效。它可以进一步减少内存消耗和提高推理速度。
    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成 ; batch=1
        if stream:
            # 返回的是out迭代器
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成; batch>1
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            # non_pad.size=(1,seq_len_non_pad)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            # 这里得到的out是yield生成的，因此是一个迭代器，每次调用next就返回一个结果
            tokens_list = [tokens[:, -1:] for tokens in out] # 一直next直到生成所有token token_list 中每个元素都是size=(1,1)的结果tensor
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad # gen.size=(1,new_token_len)
            full_sequence = torch.cat([non_pad, gen], dim=-1) # 原有的token拼接上新生成的token 
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(

                # torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device) 得到pad的形状
                # cat拼接则填充至最大长度
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]# 这一步是将每一个句子在dim1上都拼接到到最大长度 
        return torch.cat(generated, dim=0)# size=(bs,max_len)

    # input_ids.shape=(bs=1,seq_len) 
    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):

        # #start 是初始序列长度，也是将要生成的第一个token所在的index 
        start, first_seq, past_kvs = input_ids.shape[1], True, None 
        
 
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                
              # self(....)相当于调用模型自身的__call__也即forward函数
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                # 输入最后一个seq；并且start_pos是最后一个seq索引位置；shape=[b,1]
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
                
            # # out.logits.shape=(bs=1,seqlen=1,vocab_size)
            # logits.shape=(bs=1,vocab_size)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
             
            #  # rp repetition_penalty _stream是处理bs=1的方法
            # 在每步时对之前出现过的词的概率做出惩罚，即降低出现过的字的采样概率，让模型趋向于解码出没出现过的词
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            # input_ids.size=(1,seq_len) tolist后是一个二维列表 用下标[0]索引后得到一个size=(seq_len,)的python列表
            # set去除重复的 token ID，避免对同一 token 多次惩罚
            

            # 温度参数的作用通常是控制输出的随机性。当温度较高时（比如大于1），softmax后的概率分布会更平缓，各个选项的概率差异减小，
            # 这样生成的结果会更随机、更多样化。相反，当温度较低时（接近0），概率分布会更尖锐，模型会更倾向于选择最高概率的选项，结果更确定。
            logits /= (temperature + 1e-9)
            # logits.shape=(1,vocab_size)
            # top_p指的是提前设定一个固定概率阈值threshold，从n个token中按概率值排序选择最大的top p个token，这p个token的概率总和不超过threshold
            if top_p is not None and top_p < 1.0:

                # logits.shape(bs=1,vocab_size)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                # sorted_logits是每行排序后的结果  sorted_indices是排序好的元素在原logits中的索引
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p #标记为False的是要保留的token
                #掩码向右位移一位，保证最后一个使得cumulative_probs>top_p的单词也被采样
                # [false,false,true,true]=>[false,false,false,true]
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() 
              
                sorted_indices_to_remove[:, 0] = False


                #  dim=1，sorted_indices对应每行的列索引，放入sorted_indices_to_remove,值为true或false
                # 最后，indices_to_remove为true的元素是递减排序后大于top_p的元素；例如 0.3,0.2,0.1,0.4，top_p=0.7，则0.1的位置为true，代表要舍去的token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    
                logits[indices_to_remove] = -float('Inf')
                
            # 输出的是选出的索引
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            # torch.multinomial 函数用于从多项分布中进行抽样
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)

            # 每次输出预测的数据
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
