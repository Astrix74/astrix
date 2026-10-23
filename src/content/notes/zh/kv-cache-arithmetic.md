---
title: KV cache 的内存账
date: 2026-10-22T15:43:00Z
references: [shazeer2019, ainslie2023, touvron2023]
---

Transformer 生成文本时，每个新 token 都要对之前每一个 token 的键（key）和值（value）做注意力。如果每一步都重新计算它们，会非常浪费，所以推理引擎会把它们存下来，这就是 **KV cache**。它的大小很容易算出来，而这笔账能解释很多事：为什么长上下文代价高昂，为什么会有分组查询注意力。

对于每个 token，每一层都要为每个键值头存一个键向量和一个值向量：

$$
\text{bytes per token} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value}
$$

开头那个 2 是因为 K 和 V 各存一份。再乘以序列中的 token 数，然后乘以批次中的序列数。

拿一个真实的形状来算。根据 LLaMA 的[论文](https://arxiv.org/abs/2302.13971)，它的 65B 模型有 80 层，宽度为 8,192，分成 64 个头，所以 $d_\text{head} = 128$。采用 16 位精度，每个值占 2 字节：

```python
def kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per_value=2):
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens  # K and V

layers, heads, head_dim = 80, 64, 8192 // 64  # LLaMA 65B's shape, from its paper
shapes = [("multi-head (64 KV heads)", 64),
          ("grouped (8 KV heads)", 8),
          ("multi-query (1 KV head)", 1)]
for name, kv_heads in shapes:
    per_token = kv_cache_bytes(layers, kv_heads, head_dim, tokens=1)
    per_seq = kv_cache_bytes(layers, kv_heads, head_dim, tokens=2048)
    print(f"{name:<26} {per_token / 2**10:>7,.0f} KiB per token   "
          f"{per_seq / 2**30:>6.2f} GiB per 2,048 tokens")
```

```text
multi-head (64 KV heads)     2,560 KiB per token     5.00 GiB per 2,048 tokens
grouped (8 KV heads)           320 KiB per token     0.62 GiB per 2,048 tokens
multi-query (1 KV head)         40 KiB per token     0.08 GiB per 2,048 tokens
```

用标准的多头注意力（这个模型用的正是它）时，一条 2,048 token 的序列需要 5 GiB 缓存，这还不算权重。16 条这样的序列组成一个批次，就需要 80 GiB。限制一台服务器能同时处理多少请求的，是内存，而不是计算。

另外两行是同一形状下的假设推演。[多查询注意力](https://arxiv.org/abs/1911.02150)（multi-query attention，MQA；Shazeer，2019）让所有查询头共享一个键值头，在这里把缓存缩小了 64 倍。[分组查询注意力](https://arxiv.org/abs/2305.13245)（grouped-query attention，GQA；Ainslie 等，2023）是两者之间的折中。MQA“大幅加快了解码器推理的速度”，但“可能导致质量下降”，因此 GQA 使用“数量居中（多于一个，少于查询头的数量）的键值头”。分成 8 组时，缓存缩小 8 倍，而每个查询头仍然保有自己的查询。

这个公式也揭示了其他几个可调的杠杆：更少的层、更小的头、更低的精度（8 位数值能让缓存减半），以及更短的上下文。
