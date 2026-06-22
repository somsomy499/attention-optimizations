"""Flash attention implementation."""
import numpy as np

def flash_attention(q, k, v, causal=False, block_size=128):
    seq_len = q.shape[0]
    scale = q.shape[-1] ** -0.5
    output = np.zeros_like(q)
    
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len if not causal else i + 1, block_size):
            q_block = q[i:i+block_size] * scale
            k_block = k[j:j+block_size]
            v_block = v[j:j+block_size]
            scores = q_block @ k_block.T
            weights = np.exp(scores) / (np.exp(scores).sum(axis=-1, keepdims=True) + 1e-8)
            output[i:i+block_size] += weights @ v_block
            
    return output
