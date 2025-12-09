import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        # x: [batch, heads, seq_len, head_dim]
        if seq_len is None:
            seq_len = x.shape[2]
            
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
        
        # Reshape to broadcast: [1, 1, seq_len, dim]
        return emb[None, None, :, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, pos_emb):
    # x: [batch, heads, seq_len, dim]
    # pos_emb: [1, 1, seq_len, dim] (cos, sin are derived from this)
    return (x * pos_emb.cos()) + (rotate_half(x) * pos_emb.sin())

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, context=None, mask=None):
        # x: [batch, seq_len, d_model]
        # context: [batch, context_len, d_model] (if None, self-attention)
        batch_size, seq_len, _ = x.shape
        
        is_cross = context is not None
        context = context if is_cross else x
        ctx_len = context.shape[1]

        # 1. Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply RoPE to Q and K
        # Generate cos/sin tables for the current sequence lengths
        q_pos = self.rope(q, seq_len)
        k_pos = self.rope(k, ctx_len)
        
        q = apply_rotary_pos_emb(q, q_pos)
        k = apply_rotary_pos_emb(k, k_pos)
        
        # 3. Scaled Dot-Product Attention
        # scores: [batch, heads, seq_len, ctx_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Mask should be [batch, 1, seq_len, ctx_len] or broadcastable
            # Important: Use a large negative number, but not -inf to avoid NaN in softmax gradients 
            # if mixed precision or just general stability.
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 4. Combine Heads
        # out: [batch, heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU() # Standard Transformer uses ReLU or GeLU

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm architecture (often more stable than Post-Norm)
        # x + Dropout(Attn(Norm(x)))
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, mask=mask)
        x = x + self.dropout(attn_out)
        
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, src_mask=None, tgt_mask=None):
        # 1. Self Attention (Masked)
        norm_x = self.norm1(x)
        self_attn_out = self.self_attn(norm_x, mask=tgt_mask)
        x = x + self.dropout(self_attn_out)
        
        # 2. Cross Attention (attend to Encoder output)
        norm_x = self.norm2(x)
        cross_attn_out = self.cross_attn(norm_x, context=context, mask=src_mask)
        x = x + self.dropout(cross_attn_out)
        
        # 3. Feed Forward
        norm_x = self.norm3(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Pure PyTorch implementation of Transformer with RoPE, from scratch.
    No nn.Transformer or pre-built attention blocks.
    """
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        d_model=512, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6, 
        dim_feedforward=2048, 
        dropout=0.1
    ):
        super().__init__()
        
        # Embeddings (No positional encoding here, handled by RoPE in attention)
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.d_model = d_model
        
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, nhead, dim_feedforward, dropout)
        
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch, src_len]
        # tgt: [batch, tgt_len]
        
        src_x = self.src_emb(src) * math.sqrt(self.d_model)
        tgt_x = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        
        # Encode
        memory = self.encoder(src_x, mask=src_mask)
        
        # Decode
        output = self.decoder(tgt_x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Project
        return self.out_proj(output)

# --- Utilities for masks (same as before but adapted for cleaner API) ---

def create_causal_mask(size, device):
    """Creates a triangular boolean mask for causal attention (decoder self-attention)."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return ~mask # True where allowed, False where masked

def create_padding_mask(seq, pad_idx):
    """
    Creates a boolean mask where True indicates VALID tokens, False indicates PADDING.
    Output: [batch, 1, 1, seq_len] for broadcasting with attention scores.
    """
    # seq: [batch, seq_len]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask
