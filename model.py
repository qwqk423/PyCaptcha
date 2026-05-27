import torch
import torch.nn as nn
from collections import OrderedDict


# CRNN (Convolutional Recurrent Neural Network)：
# CNN 提取图片特征 -> LSTM 序列建模 -> FC 分类输出
# 配合 CTC Loss 实现不定长文本识别
class CRNN(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 192)):
        """
        :param n_classes:   字符类别数（包含CTC blank）
        :param input_shape: 输入图片形状 (C, H, W)
        """
        super(CRNN, self).__init__()
        self.input_shape = input_shape
        # 每个卷积块的输出通道数
        channels = [32, 64, 128, 256, 256]
        # 每个卷积块包含的卷积层数
        layers = [2, 2, 2, 2, 2]
        # 卷积核大小
        kernels = [3, 3, 3, 3, 3]
        # 池化核大小，前4个是2x2，最后一个是(2,1)只在高度上压缩、宽度不变
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        # 辅助函数：Conv2d + BatchNorm + ReLU 三个模块打包
        def cba(name, in_channels, out_channels, kernel_size):
            modules[f'conv{name}'] = nn.Conv2d(in_channels, out_channels, kernel_size,
                                               padding=(1, 1) if kernel_size == 3 else 0)
            modules[f'bn{name}'] = nn.BatchNorm2d(out_channels)
            modules[f'relu{name}'] = nn.ReLU(inplace=True)

        last_channel = 3  # 输入通道数（RGB）
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(zip(channels, layers, kernels, pools)):
            for layer in range(1, n_layer + 1):
                cba(f'{block+1}{layer}', last_channel, n_channel, n_kernel)
                last_channel = n_channel
            # 每个block后接一个MaxPool
            modules[f'pool{block + 1}'] = nn.MaxPool2d(k_pool)
        modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.cnn = nn.Sequential(modules)

        # 自动推算CNN输出特征图的通道数（展平后作为LSTM的输入维度）
        self.lstm = nn.LSTM(input_size=self.infer_features(), hidden_size=128,
                            num_layers=2, bidirectional=True)
        # 双向LSTM两个方向拼接 -> 256维 -> 全连接映射到 n_classes
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        """用一次哑前向传播推算LSTM输入维度"""
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        # (batch, channels, h, w) -> (batch, channels*h, w)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]  # 返回 channels * height

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.cnn(x)
        # (B, C, H, W) -> (B, C*H, W)  将高度维合并到通道维
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # (B, C*H, W) -> (W, B, C*H)  转为序列格式：(seq_len, batch, input_size)
        x = x.permute(2, 0, 1)
        # 双向LSTM输出
        x, _ = self.lstm(x)
        # 全连接层 -> (seq_len, batch, n_classes)  每个时间步预测一个字符分布
        x = self.fc(x)
        return x
