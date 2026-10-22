---
title: KV-cache memory arithmetic
date: 2026-10-22T15:43:00Z
references: [shazeer2019, ainslie2023, touvron2023]
---

When a transformer generates text, each new token attends to the keys and values of every earlier token. Recomputing them at every step would be wasteful, so inference engines keep them: the **KV cache**. Its size is easy to work out, and it explains a lot about why long contexts are expensive and why grouped-query attention exists.

For every token, every layer stores one key vector and one value vector per key-value head:

$$
\text{bytes per token} = 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value}
$$

The leading 2 is for K and V. Multiply by the number of tokens in the sequence, then by the number of sequences in the batch.

Take a real shape. LLaMA's 65B model, per its [paper](https://arxiv.org/abs/2302.13971), has 80 layers and width 8,192 split into 64 heads, so $d_\text{head} = 128$. At 16-bit precision, 2 bytes per value:

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

With standard multi-head attention, which is what that model uses, one 2,048-token sequence needs 5 GiB of cache, on top of the weights. A batch of 16 such sequences needs 80 GiB. Memory, not arithmetic, is what limits how many requests a server can handle at once.

The other two rows are what-ifs for the same shape. [Multi-query attention](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) shares one key-value head across all query heads, cutting the cache 64-fold here. [Grouped-query attention](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) is the middle ground. MQA "drastically speeds up decoder inference" but "can lead to quality degradation", so GQA uses "an intermediate (more than one, less than number of query heads) number of key-value heads". With 8 groups, the cache shrinks 8-fold while each query head keeps its own queries.

The formula also shows the other levers: fewer layers, smaller heads, lower precision (8-bit values halve the cache), and shorter contexts.
