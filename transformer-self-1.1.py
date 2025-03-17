import torch
from torch.nn import Module
from torch import nn
import math
import torch.nn.functional as F

'''位置信息编码模块：为输入序列添加位置信息'''
class PositionalEncoding(Module):
    """d_model: 词嵌入维度（也就是序列每个单元需要多少维度表示），max_len: 支持的最大序列长度"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # 继承torch.nn.Modlue类的基础方法
        # torch.nn.Modlue类： PyTorch 中所有神经网络模块的基类，
        # 所有神经网络组件（比如层、参数、计算步骤）都是它或它的子类
        super().__init__()

        # pe:编码矩阵初始化
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引position.shape(max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算频率项
        div_term = torch.exp(
            torch.arange(0,d_model,2,dtype=torch.float)
            *(math.log(10000)/-d_model))
        # 偶数位置索引
        pe[:,::2] = torch.sin(position*div_term)
        # 奇数位置索引
        pe[:,1::2] = torch.cos(position*div_term)

        # 增加batch维度，便于训练pe.shape(1,max_len,d_model)
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为buffer（不会被训练，但会保存到模型参数中）
        # 该方法将张量注册为模块的缓冲区，使得张量成为模块的一部分，并在模块保存和加载时被处理。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.shape(batch_size,seq_len,d_model)
        # pe只取前seq_len个，降低计算成本
        # self.pe[:, :x.size(1)].shape(1,seq_len,d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

'''多头注意力机制'''
class MultiHeadAttention(Module):
    def __init__(self, d_model, h, dropout=0.1):
        # d_model：输入的维度，h:注意力头数量
        super().__init__()
        # 断言d_model能被h整除，否则报错
        assert d_model % h == 0 ,"d_model must be divisible by num_heads"
        # 将每个注意力头的维度均分:
        # 为了在保持计算效率的同时，让每个注意力头能够专注于学习不同的语义特征。
        # 减少了计算量和参数量，提高了模型的泛化能力和训练效率。
        # 若保持每个注意力头的维度与原始维度一致，在理论上可以提高模型的表达能力，
        # 但在实际应用中往往会导致计算资源的浪费和训练难度的增加。
        self.d_k = d_model//h
        self.h = h

        # 线性变换矩阵Q、K、V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    """线性变换层"""
    def linear_transform(self,x,linear_layer):
        return linear_layer(x)

    """将线性变换后的结果切分为多个头"""
    def split_head(self,x,batch_size):
        # x.shape(batch_size, seq_len, d_model)
        # to (batch_size, seq_len,h, d_model)
        # 因为h*d_k=d_model,所以-1直接将原来还剩的维度存到第二维度
        x = x.view(batch_size, -1, self.h, self.d_k)
        # 将h维度转换到第二维度，便于后续将x(q\k\v)分给不同注意力头
        x = x.permute(0, 2, 1, 3)# 等价于x.transpose(1,2)
        return x

    """计算缩放点积注意力(默认不mask)"""
    def scaled_dot_attn(self,q,k,v,mask=None):
        # q\k\v.shape (batch_size, h, seq_len, d_k)
        # mask.shape (batch_size, h, seq_len, seq_len)
        # q与k^T相乘后缩小sqrt(d_k)得到
        # 注意力分数矩阵(batch_size, h, seq_len, seq_len)
        attention_score = (q@k.transpose(-2, -1))/math.sqrt(self.d_k)

        # 掩码的应用
        if mask is not None:
            # mask==0，得到bool矩阵，True位置对应需要mask的，False位置对应不mask
            # attention_score进行mask：
            # bool矩阵中True 的位置值替换为 -1e9，False保持不变
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attention_weight = F.softmax(attention_score, dim=-1)
        attention_weight = self.dropout(attention_weight)

        # 输出(batch_size, h, seq_len, d_k)
        output = attention_weight @ v
        return output, attention_weight

    """合并多个头的输出"""
    def merge_head(self, x, batch_size):
        # from (batch_size, h, seq_len, d_k)
        # to (batch_size, seq_len, h, d_k)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, -1, self.h*self.d_k)
        return x

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Self-Attention 中，q、k、v 被设定为相同的值，即q=k=v=x。
        # 因此，在代码调用时，通常直接传入 x 作为这三个参数
        # Cross-Attention（如 Decoder 中）时，q、k、v 可能不同：
        # q 来自 Decoder 的当前输入（或上一层的输出）；
        # k 和 v 来自 Encoder 的输出

        # 对查询、键、值进行线性变换之后，直接分给为多个头
        q = self.linear_transform(q, self.W_q)
        q = self.split_head(q, batch_size)

        k = self.linear_transform(k, self.W_k)
        k = self.split_head(k, batch_size)

        v = self.linear_transform(v, self.W_v)
        v = self.split_head(v, batch_size)

        # 多头注意力的计算与合并
        output,attention_weight = self.scaled_dot_attn(q, k, v, mask)
        output = self.merge_head(output, batch_size)

        # 最终输出线性变换
        output = self.W_o(output)
        return output

'''FFN前馈神经网络'''
class FeedForward(Module):
    def __init__(self, d_model,d_ff=2048, dropout=0.1):
        super().__init__()
        # 两层：第一层扩展维度(非线性在激活函数实现)，第二层压缩回维度且组合第一层非线性
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    """前向传播：Linear->ReLU->Dropout->Linear"""
    def forward(self, x):
        # (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        # 非线性关系
        x = F.relu(x)
        x = self.dropout(x)
        # (batch_size, seq_len, d_ff)
        return self.linear2(x)

'''EncoderBlock/Layer：Multi-Head Attention + FeedForward,每层后ADD&Norm'''
class EncoderLayer(Module):
    def  __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_att = MultiHeadAttention(d_model,h,dropout)
        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力的残差连接
        # self.self_att(x, x, x, mask)会自动调用MultiHeadAttention类的
        # forward方法。在PyTorch中，当你定义一个nn.Module类的实例
        # 并对其进行函数调用（即使用()）时，它会自动调用该类的forward方法
        x = x + self.dropout1(self.self_att(x, x, x, mask))
        x = self.norm1(x)

        # FFN的残差连接
        x = x + self.dropout2(self.ffn(x))
        return self.norm2(x)

'''DecoderBlock：Masked Multi-Head Attention + Multi-Head Attention + FeedForward'''
class DecoderLayer(Module):
    def __init__(self, d_model,h,d_ff,dropout=0.1):
        super().__init__()
        self.self_att = MultiHeadAttention(d_model,h,dropout)
        self.cross_att = MultiHeadAttention(d_model,h,dropout)
        self.ffn = FeedForward(d_model,d_ff,dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 第一层：自注意力（目标序列掩码）
        """自注意力使用tgt_mask"""
        x = x + self.dropout1(self.self_att(x, x, x, tgt_mask))
        x = self.norm1(x)

        # 第二层：交叉注意力
        # 使用上一个DecoderBlock的输出或x(第一个DecoderBlock时)计算Q
        # 编码器的输出生成K与V，以及mask源序列的无效位置（如padding）
        """交叉注意力使用src_mask"""
        x = x + self.dropout2(self.cross_att(x, encoder_output, encoder_output, src_mask))
        x = self.norm2(x)

        # 第三层：前馈神经网络
        x = x + self.dropout3(self.ffn(x))
        return self.norm3(x)

'''EncoderBlock堆叠'''
class Encoder(Module):
    def __init__(self, num_layers,d_model,h,d_ff, dropout=0.1):
        super().__init__()
        # 创建num_layers个EncoderBlock
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model,h,d_ff,dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

'''DecoderBlock堆叠'''
class Decoder(Module):
    def __init__(self, num_layers,d_model,h,d_ff, dropout=0.1):
        super().__init__()
        # 创建num_layers个DecoderBlock
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff,dropout) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(Module):
    def __init__(self,src_vocab_size, tgt_vocab_size, num_layers=6,
                 d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 源与目标词语向量化
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(num_layers,d_model,h,d_ff,dropout)
        self.decoder = Decoder(num_layers,d_model,h,d_ff,dropout)

        self.fc_output = nn.Linear(d_model,tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)  # 新增Softmax层
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # encoder
        src_emb = self.encoder_embed(src)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        encoder_out = self.encoder(src_emb, src_mask)

        # decoder
        tgt_emb = self.decoder_embed(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        decoder_out = self.decoder(tgt_emb, encoder_out, src_mask, tgt_mask)
        logits = self.fc_output(decoder_out)

        return self.softmax(logits)

def generate_mask(src,tgt,pad_idx=0):
    # 屏蔽padding部分
    ## 假设输入序列 src = [[1, 2, 0]]（pad_idx=0）
    ## src_mask = [[[[True, True, False]]]]  # 屏蔽第三个位置（填充）
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, src_len)




    # 屏蔽padding部分,假设目标序列 tgt = [[3, 4, 0]],pad_idx=0
    ## src_mask = [[[[True, True, False]]]]  # 屏蔽第三个位置（填充）
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # 上三角掩码
    # # triu_mask = [[[[True, False, False],
    #                 [True, True, False],
    #                 [True, True, True]]]]  # 未来信息掩码
    seq_len = tgt.size(1)
    # diagonal=0对角线为与上三角保留；1，不含对角线为的上三角保留
    # 然后1-原来的矩阵再反转
    trid_mask = (1-torch.triu(torch.ones(1,seq_len, seq_len), diagonal=1)).bool()
    # merged_tgt_mask = [[[[True, False, False],
    #                      [True, True, False],
    #                      [True, True, False]]]]  # 合并后的掩码
    tgt_mask = tgt_mask & trid_mask.to(tgt.device)

    return src_mask, tgt_mask

# 使用示例
if __name__ == "__main__":
    # 表示源语言和目标语言的词汇表各包含1000个唯一词元
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    model = Transformer(src_vocab_size, tgt_vocab_size,
                        # num_layers=2编码器和解码器各包含2个堆叠的Transformer层
                        # d_model=128决定每个词元会被映射为128维向量\
                        # h=4多头注意力机制中分4个并行计算头，每个头的维度为d_model/h=32
                        num_layers=2,d_model=128, h=4,
                        # d_ff通常设置为4*d_model
                        # 在注意力计算和FFN层中随机丢弃10%的神经元，防止过拟合
                        d_ff=512, dropout=0.1)

    # 测试数据
    # 形状：(batch_size=10, src_seq_len=20)
    src = torch.randint(0, src_vocab_size, (10, 20))
    # 形状：(batch_size=10, src_seq_len=25)
    tgt = torch.randint(0, tgt_vocab_size, (10, 25))

    # 掩码
    src_mask, tgt_mask = generate_mask(src,tgt)

    output = model(src, tgt, src_mask, tgt_mask)
    print(output.size())