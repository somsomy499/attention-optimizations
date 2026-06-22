# Attention Optimizations ⚡

Optimized attention implementations for transformers: Flash, Paged, Sliding Window, Ring.

## Benchmarks

| Method | Memory | Speed | Max Seq Len |
|--------|--------|-------|-------------|
| Standard | O(N²) | 1x | 4K |
| Flash Attention 2 | O(N) | 2.4x | 128K |
| Paged Attention | O(N) | 2.1x | 128K+ |
| Ring Attention | O(N/P) | 1.8x | 1M+ |

## Quick Start

```python
from attention_opt import flash_attention

output = flash_attention(query, key, value, causal=True)
```

## License

Apache 2.0