from torch import nn
from torch.nn import Module
import torch
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
            torch.arange(0,d_model,dtype=torch.float)
            *(-math.log(10000)/d_model))
        # 偶数位置索引
        pe[,::2] = torch.sin(position*div_term)
        # 奇数位置索引
        pe[,1::2] = torch.cos(position*div_term)

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

class EncoderLayer(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass

class DecoderLayer(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass

class Encoder(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass

class Decoder(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass

class MultiHeadAttention(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass

class FeedForward(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass

class Transformer(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pass




