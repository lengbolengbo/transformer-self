import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embed size must be divisible by num_heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.num_heads pieces
        values = values.view(N, value_len, self.num_heads, self.head_dim)
        keys = keys.view(N, key_len, self.num_heads, self.head_dim)
        queries = query.view(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        forward_expansion=4,
        dropout=0.1,
        max_len=5000,
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.embed_size = embed_size

        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.pos_embedding = PositionalEncoding(embed_size, max_len)

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    forward_expansion,
                    dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    forward_expansion,
                    dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src, device):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)  # 将掩码移动到与模型相同的设备上

    def make_trg_mask(self, trg, device):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(device)  # 将掩码移动到与模型相同的设备上

    def forward(self, src, trg):
        device = src.device  # 获取输入数据所在的设备
        src_mask = self.make_src_mask(src, device)
        trg_mask = self.make_trg_mask(trg, device)
        src_embedding = self.dropout(self.pos_embedding(self.src_word_embedding(src)))
        trg_embedding = self.dropout(self.pos_embedding(self.trg_word_embedding(trg)))

        for encoder_block in self.encoder:
            src_embedding = encoder_block(src_embedding, src_embedding, src_embedding, src_mask)

        for decoder_block in self.decoder:
            trg_embedding = decoder_block(
                trg_embedding, src_embedding, src_embedding, trg_mask
            )

        out = self.fc_out(trg_embedding)
        return out

# 示例使用
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_vocab_size = 1000
    trg_vocab_size = 1000
    src_pad_idx = 0
    trg_pad_idx = 0
    embed_size = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    forward_expansion = 4
    dropout = 0.1
    max_len = 5000

    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
    ).to(device)

    src = torch.randint(0, src_vocab_size, (32, 10)).to(device)  # (batch_size, src_len)
    trg = torch.randint(0, trg_vocab_size, (32, 10)).to(device)  # (batch_size, trg_len)

    out = model(src, trg)
    print(out.shape)  # 应该输出 (batch_size, trg_len, trg_vocab_size)