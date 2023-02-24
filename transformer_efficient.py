import torch, math
from torch import nn

## Global Variables
n_embd = 32
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
vocab_size = 65
block_size = 8
dropout = 0.1

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2.0 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3.0)))

# Multi-headed self-attention
class MultiHeadAttentionBatch(nn.Module):
    def __init__(self, head_size, num_heads, mask = False):
        super().__init__()
        self.mask = mask
        self.num_heads = num_heads
        self.head_size = head_size
        self.input_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.output_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # block_size is the context length
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x = batch size (B), sequence length (T), embedding dimensionality (C)
        B, T, C = x.shape
        x_proj = self.input_proj(x) # B, T, 3 * (num_heads * head size)

        k, q, v = x_proj.split(self.num_heads * self.head_size, dim=-1) # B, T, (num_heads * head_size)

        K = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # B, num_heads, T, head_size
        Q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # B, num_heads, T, head_size
        V = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2) # B, 1, T, n_embd @ B, num_heads, n_embd, head_size -> B, num_heads, T, head_size

        scores = Q @ K.transpose(-2, -1) # B, num_heads, T, T

        # optionally, lower triangular mask on the scores
        if self.mask:
            scores = scores.masked_fill(self.tril == 0, float("-inf")) # note: self.tril is broadcastable with scores

        # scaled attention
        scores *= (C ** -0.5) # should be applied element wise
        scores = scores.softmax(dim=-1) # want to apply on last dim: B, num_heads, T, T
        scores = self.dropout(scores)
        attention = scores @ V # B, num_heads, T, head_size

        # want to recombine heads
        attention = attention.transpose(1, 2).contiguous().view(B, T, C) # B, T, num_heads * head_size
        return self.output_proj(attention)

# Transformer Block
class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.self_attention = MultiHeadAttentionBatch(n_embd // num_heads, num_heads, True) # B, T, n_embd
        self.ff1 = nn.Linear(n_embd, n_embd * 4)
        self.ff2 = nn.Linear(n_embd * 4, n_embd)
        self.GELU = NewGELU()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attention_norm = self.ln1(x + self.self_attention(x))
        ff_activations = self.ff2(self.GELU(self.ff1(attention_norm))) # project into 4 * n_embd and then back to n_embd
        ff_norm = self.ln2(x + ff_activations)
        return self.dropout(ff_norm)

# Tiny GPT
class miniGPT(nn.Module):
    def __init__(self, num_heads, num_blocks, vocab_size):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(num_heads) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(n_embd)
        self.ll = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        # Extract Info
        B, T = x.shape

        # Create positional and character embeddings
        vocab_emb = self.vocab_embed(x) # B, T, n_embd
        pos_embd = self.pos_embed(torch.arange(T, device=device)) # T, n_embd
        embd = vocab_emb + pos_embd # B, T, n_embd
        
        # FP into transformer blocks
        t_out = self.blocks(embd) # B, T, n_embd

        # Layer-norm transformer output layer
        t_out = self.ln(t_out)

        # Project onto vocab_size
        logits = self.ll(t_out) # B, T, vocab_size, probability of each token in vocab size

        return logits


        



        