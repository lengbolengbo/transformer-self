import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块：为输入序列添加位置信息"""

    def __init__(self, d_model, max_len=5000):
        # d_model: 词嵌入维度，max_len: 支持的最大序列长度
        super().__init__()

        # 初始化位置编码矩阵（max_len x d_model）
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引（0到max_len-1的列向量）
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率项：exp(-2i/d_model * log(10000))，用于正弦/余弦函数
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # 偶数位置填充正弦值，奇数位置填充余弦值
        pe[:, 0::2] = torch.sin(position * div_term)  # 从0开始，步长2
        pe[:, 1::2] = torch.cos(position * div_term)  # 从1开始，步长2

        # 调整形状：增加batch维度 -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为buffer（不会被训练，但会保存到模型参数中）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的形状：(batch_size, seq_len, d_model)
        # 将位置编码加到输入序列上（自动广播到batch维度）
        x = x + self.pe[:, :x.size(1)]  # 只取前seq_len个位置
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, h, dropout=0.1):
        # d_model: 输入维度，h: 注意力头的数量
        super().__init__()
        self.d_k = d_model // h  # 每个头的维度
        self.h = h

        # 定义四个线性变换：Q, K, V和最终输出
        self.W_q = nn.Linear(d_model, d_model)  # 查询向量变换
        self.W_k = nn.Linear(d_model, d_model)  # 键向量变换
        self.W_v = nn.Linear(d_model, d_model)  # 值向量变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

        self.dropout = nn.Dropout(dropout)  # 注意力权重dropout

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取batch大小

        # 线性变换 + 分割多头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        q = self.W_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力得分
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（将需要屏蔽的位置设为极小值）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重（softmax归一化）
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 应用dropout

        # 计算上下文向量
        output = attn_weights @ v  # (batch_size, h, seq_len, d_k)

        # 合并多头 -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k
        )

        # 最终线性变换
        return self.W_o(output)


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # 两层线性变换 + ReLU激活
        self.linear1 = nn.Linear(d_model, d_ff)  # 扩展维度
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # 恢复维度

    def forward(self, x):
        # 前向传播：Linear -> ReLU -> Dropout -> Linear
        x = self.linear1(x)  # (batch_size, seq_len, d_ff)
        x = F.relu(x)  # 非线性激活
        x = self.dropout(x)  # 随机失活
        return self.linear2(x)  # (batch_size, seq_len, d_model)


class EncoderLayer(nn.Module):
    """编码器层：包含自注意力和前馈网络"""

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)  # 自注意力
        self.ffn = FeedForward(d_model, d_ff, dropout)  # 前馈网络

        # 子层标准化和Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 残差连接 + 自注意力
        x = x + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)  # 层标准化

        # 残差连接 + 前馈网络
        x = x + self.dropout2(self.ffn(x))
        return self.norm2(x)


class DecoderLayer(nn.Module):
    """解码器层：包含自注意力、交叉注意力和前馈网络"""

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)  # 自注意力（带掩码）
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)  # 交叉注意力（编码器-解码器）
        self.ffn = FeedForward(d_model, d_ff, dropout)  # 前馈网络

        # 三个子层的标准化和Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 第一层：自注意力（带目标序列掩码）
        x = x + self.dropout1(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x)

        # 第二层：交叉注意力（使用编码器输出作为K,V）
        x = x + self.dropout2(self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.norm2(x)

        # 第三层：前馈网络
        x = x + self.dropout3(self.ffn(x))
        return self.norm3(x)


class Encoder(nn.Module):
    """编码器堆叠多个编码器层"""

    def __init__(self, num_layers, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        # 创建N个相同的编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # 顺序通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    """解码器堆叠多个解码器层"""

    def __init__(self, num_layers, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        # 创建N个相同的解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 顺序通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    """完整Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 num_layers=6, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 源语言和目标语言的词嵌入
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)

        # 编码器和解码器堆叠
        self.encoder = Encoder(num_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, h, d_ff, dropout)

        # 输出层：将解码器输出映射到目标词汇表
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器部分
        src_emb = self.encoder_embed(src)  # 词嵌入
        src_emb = self.pos_encoding(src_emb)  # 位置编码
        src_emb = self.dropout(src_emb)  # 应用Dropout
        enc_output = self.encoder(src_emb, src_mask)  # 编码器前向

        # 解码器部分
        tgt_emb = self.decoder_embed(tgt)  # 词嵌入
        tgt_emb = self.pos_encoding(tgt_emb)  # 位置编码
        tgt_emb = self.dropout(tgt_emb)  # 应用Dropout
        dec_output = self.decoder(
            tgt_emb, enc_output, src_mask, tgt_mask
        )  # 解码器前向

        # 输出映射
        return self.fc_out(dec_output)


def generate_mask(src, tgt, pad_idx=0):
    """生成源序列和目标序列的掩码"""
    # 源序列掩码（屏蔽pad位置）
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)

    # 目标序列掩码（屏蔽pad位置）
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)

    # 生成上三角掩码（防止看到未来信息）
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask.to(tgt.device)  # 合并两种掩码

    return src_mask, tgt_mask


# 使用示例
if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        num_layers=2, d_model=128, h=4, d_ff=512
    )

    # 生成测试数据（假设序列长度分别为20和25）
    src = torch.randint(0, src_vocab_size, (10, 20))  # (batch_size, src_len)
    tgt = torch.randint(0, tgt_vocab_size, (10, 25))  # (batch_size, tgt_len)

    # 生成掩码
    src_mask, tgt_mask = generate_mask(src, tgt)

    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)
    print(output.shape)  # torch.Size([10, 25, 1000])