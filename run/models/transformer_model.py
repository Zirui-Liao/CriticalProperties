import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ===============================
# 1. Dataset
# ===============================
class SmilesDataset(Dataset):
    """
    Dataset for SMILES strings and corresponding targets.

    - Encodes SMILES as index sequences with padding.
    - Returns x (LongTensor), y (FloatTensor), and mask (BoolTensor for non-pad tokens).
    """

    def __init__(self, smiles_list, labels, char2idx, max_len=100):
        self.data = [self.encode(smi, char2idx, max_len) for smi in smiles_list]
        self.labels = labels

    @staticmethod
    def encode(smi, char2idx, max_len):
        idxs = [char2idx.get(ch, 0) for ch in smi]
        idxs = idxs[:max_len]
        if len(idxs) < max_len:
            idxs += [char2idx["<PAD>"]] * (max_len - len(idxs))
        return idxs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask = x != 0  # True for non-PAD tokens
        return x, y, mask


# ===============================
# 4. Transformer Encoder
# ===============================
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, hidden_factor=2, dropout_model=0.1):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads=n_head, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout_model)
        self.dropout_ffn = nn.Dropout(dropout_model)

        hidden_dim = hidden_factor * d_model
        self.ffn_linear1 = nn.Linear(d_model, hidden_dim)
        self.ffn_linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x, token_mask=None):
        """
        x: (B, L, d_model)
        token_mask: (B, L) BoolTensor, True for valid tokens
        """
        # key_padding_mask: True �?PAD 位置
        key_padding_mask = ~token_mask if token_mask is not None else None

        # Self-attention block (pre-norm)
        attn_input = self.norm1(x)
        attn_output, _ = self.self_attention(
            attn_input, attn_input, attn_input, key_padding_mask=key_padding_mask
        )
        out = x + self.dropout_attn(attn_output)

        # FFN block (pre-norm)
        ffn_input = self.norm2(out)
        ffn_hidden = F.relu(self.ffn_linear1(ffn_input))
        ffn_hidden = self.dropout_ffn(ffn_hidden)
        ffn_output = self.ffn_linear2(ffn_hidden)
        out = out + ffn_output  # (B, L, d_model)

        return out


class TransformerEncoderModel(nn.Module):
    """
    A minimal Transformer encoder for SMILES regression.

    Structure:
    - Token embedding + positional encoding
    - Stack of `num_layers` self-attention + FFN blocks (pre-norm)
    - Masked mean pooling over sequence
    - MLP head to predict multiple properties
    """

    def __init__(
            self,
            vocab_size,
            max_len=100,
            d_model=32,
            hidden_factor=2,
            n_head=1,
            output_dim=4,
            dropout_pos=0.1,
            dropout_model=0.1,
            num_layers=2,        # how many EncoderBlocks
            head_hidden_factor=2,  # hidden dim of MLP head; default = d_model
            dropout_head=0.1,    # dropout in MLP head
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_len, dropout=dropout_pos
        )

        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                n_head=n_head,
                hidden_factor=hidden_factor,
                dropout_model=dropout_model,
            )
            for _ in range(num_layers)
        ])

        # Head hidden dim default
        if head_hidden_factor is None:
            head_hidden_factor = hidden_factor
        head_hidden_dim = d_model * head_hidden_factor

        # MLP head for regression
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len) with 0 as PAD id
        mask: (batch_size, seq_len) BoolTensor, True for valid tokens
        """
        # Build token mask if not provided
        if mask is None:
            token_mask = x != 0
        else:
            token_mask = mask

        # Embedding + positional encoding
        emb = self.embedding(x)  # (B, L, d_model)
        out = self.pos_encoder(emb)

        # Pass through stacked encoder blocks
        for layer in self.layers:
            out = layer(out, token_mask=token_mask)

        # Masked mean pooling over sequence
        token_mask_expanded = token_mask.unsqueeze(-1)  # (B, L, 1)
        valid_counts = token_mask_expanded.sum(dim=1).clamp(min=1)  # avoid division by zero
        pooled = (out * token_mask_expanded).sum(dim=1) / valid_counts  # (B, d_model)

        # Final MLP head
        return self.head(pooled)  # (B, output_dim)
