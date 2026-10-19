#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
 从零实现 GPT-2 (124M) 的完整教学代码
================================================================================
 本文件按照以下顺序逐步构建一个 GPT 模型：
   1. 定义 GPT-2 small 的配置
   2. 用"占位模块"搭一个空壳模型（先跑通流程）
   3. 用 tiktoken 做分词，构造 batch
   4. 手写 LayerNorm（先手动演示，再封装成类）
   5. 手写 GELU 激活函数
   6. 手写 FeedForward 前馈网络
   7. 用一个小网络演示残差连接如何缓解梯度消失
   8. 手写 MultiHeadAttention 多头因果自注意力
   9. 组装 TransformerBlock
  10. 组装完整的 GPTModel
  11. 统计参数量、计算模型大小
  12. 实现简单的文本生成函数

 运行环境：PyTorch >= 2.0，tiktoken，matplotlib
 安装依赖：pip install torch tiktoken matplotlib
================================================================================
"""

# ==============================================================================
# 第 0 部分：导入依赖
# ==============================================================================
# torch：PyTorch 主库，提供张量和自动求导
# torch.nn：神经网络模块（层、损失函数等）
import os
from time import sleep
# os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# tiktoken：OpenAI 开源的 BPE 分词器，与 GPT-2/GPT-3 完全兼容
import tiktoken

# matplotlib：用于画激活函数曲线，便于直观理解 GELU vs ReLU
import matplotlib.pyplot as plt


# ==============================================================================
# 第 1 部分：GPT-2 small (124M) 的超参数配置
# ==============================================================================
# 把模型所有超参数集中放在一个字典里，便于统一修改和传递。
# 这样设计的好处：
#   - 改一个数字就能切换模型大小（比如换成 GPT-2 medium）
#   - 所有子模块（注意力、前馈、Block）都从同一个 cfg 读取参数
#   - 避免"魔数"散落在代码各处
model_path = "/kaggle/input/datasets/mangzhang/hfgpt2traindata/hfgpt2traindata/"  # 模型文件夹路径

GPT_CONFIG_124M = {
    # 词表大小：GPT-2 使用 BPE 分词，词汇表共 50257 个 token
    # 组成 = 50000 个 merge 单元 + 256 个字节 token + 1 个 <|endoftext|>
    "vocab_size": 50260, #  # 50257 + 3 个特殊 token

    # 上下文长度：模型一次最多能"看到" 1024 个 token
    # 位置嵌入表的行数也只需 1024，因为位置 id 最大是 1023
    "context_length": 1024,

    # 嵌入维度：每个 token 被映射成 768 维向量
    # 这也是 Transformer 所有子层的"隐藏宽度"
    "emb_dim": 768,

    # 多头注意力的头数：768 / 12 = 64，每个头处理 64 维
    # 多头让模型能在不同子空间并行学习不同的注意力模式
    "n_heads": 12,

    # Transformer Block 的层数：堆叠 12 层
    # 层数越深，模型的表达能力越强，但训练也越难
    "n_layers": 12,

    # Dropout 概率：训练时随机丢弃 10% 的神经元，防止过拟合
    # 推理时必须关闭（model.eval() 会自动处理）
    "drop_rate": 0.1,

    # Q/K/V 线性层是否使用偏置：GPT-2 原版设为 False
    # 因为 LayerNorm 已经处理了均值，bias 边际收益很小
    "qkv_bias": True,
}


# ==============================================================================
# 第 2 部分：占位版 GPT 模型（先跑通整体数据流）
# ==============================================================================
# 工程习惯：先搭一个"空壳"，里面所有子模块都是恒等映射（直接返回输入）。
# 这样做的好处：
#   - 快速验证张量形状是否正确
#   - 定位 bug 时可以先排除复杂模块
#   - 后续逐步替换成真实实现，每次替换都能单独验证


class DummyTransformerBlock(nn.Module):
    """
    占位 TransformerBlock：什么都不做，原样返回输入。
    后续会被真正的 TransformerBlock 替换。
    """

    def __init__(self, cfg):
        # 调用父类 nn.Module 的构造函数，必须写
        super().__init__()
        # 没有任何可学习参数

    def forward(self, x):
        # 恒等映射：输入是什么，输出就是什么
        return x


class DummyLayerNorm(nn.Module):
    """
    占位 LayerNorm：什么都不做，原样返回输入。
    参数只是为了模拟真实 LayerNorm 的接口，实际不用。
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # 保存这些参数只是为了接口一致，占位实现里用不到
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        # 恒等映射
        return x


class DummyGPTModel(nn.Module):
    """
    占位版 GPT 模型：
      - 有完整的 token 嵌入、位置嵌入、输出头
      - 但中间的 TransformerBlock 和 LayerNorm 都是恒等映射
    目的：验证前向传播的流程和形状。
    """

    def __init__(self, cfg):
        super().__init__()

        # token 嵌入层：形状 (vocab_size, emb_dim) = (50257, 768)
        # 输入 token id，输出对应的 768 维向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # 位置嵌入层：形状 (context_length, emb_dim) = (1024, 768)
        # 输入位置 id（0~1023），输出位置向量
        # 因为自注意力本身不区分位置，必须额外加位置信息
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # 嵌入层的 dropout，防止过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠 n_layers 个占位 Block
        # nn.Sequential：按顺序依次调用子模块
        # *[...]：解包语法，把列表展开成独立的位置参数
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终 LayerNorm（占位版）
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # 输出头：把 768 维隐藏向量映射回 50257 维 logits
        # bias=False：GPT-2 原版不使用偏置
        # 注意：这个权重通常和 tok_emb.weight 共享（weight tying）
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx 形状：(batch_size, seq_len)，里面是 token id
        batch_size, seq_len = in_idx.shape

        # 1) token 嵌入：(batch, seq_len) -> (batch, seq_len, 768)
        tok_embeds = self.tok_emb(in_idx)

        # 2) 位置嵌入：
        #    - torch.arange(seq_len) 生成 [0,1,...,seq_len-1]
        #    - device=in_idx.device 保证位置张量和输入在同一设备（CPU/GPU）
        #    - 查表后形状 (seq_len, 768)，广播相加后 (batch, seq_len, 768)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # 3) 相加：token 语义 + 位置信息
        x = tok_embeds + pos_embeds

        # 4) 嵌入 dropout
        x = self.drop_emb(x)

        # 5) 经过 12 个占位 Block（原样返回）
        x = self.trf_blocks(x)

        # 6) 最终 LayerNorm（占位版）
        x = self.final_norm(x)

        # 7) 输出头：(batch, seq_len, 768) -> (batch, seq_len, 50257)
        logits = self.out_head(x)
        return logits


# ==============================================================================
# 第 3 部分：用 tiktoken 做分词，构造 batch
# ==============================================================================
# BPE（Byte Pair Encoding）：
#   - 从单个字符开始，反复合并出现频率最高的相邻对
#   - 最终得到 ~5 万个子词单元，能兼顾词表大小和未登录词
#   - GPT-2 的词表就是通过 BPE 训练得到的

# 加载 GPT-2 的分词器（会自动下载词表文件，第一次运行需要联网）
# tokenizer = tiktoken.get_encoding("gpt2")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 添加你的特殊 Token
special_tokens = ["<|persona|>", "<|user|>", "<|bot|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})


batch = []  # 用列表暂存每条样本的 token id 张量
txt1 = "Every effort moves you"  # 第一句
txt2 = "Every day holds a"       # 第二句
txt3 = "How are you today"       # 第三句


# tokenizer.encode(str) -> list[int]
# torch.tensor(list) -> 一维张量,pytorch 只接受张量作为输入
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch.append(torch.tensor(tokenizer.encode(txt3)))

# torch.stack(batch, dim=0)：把三个形状 (seq_len,) 的张量堆成 (3, seq_len)
# 第一维是 batch，第二维是序列长度
# 注意：要求两条句子 token 数相同（这里恰好都是 5）
batch = torch.stack(batch, dim=0)

print("\033[31m \n===== 输出：把两句话编码成 token，并堆叠成 batch =====\n\033[0m")
print(batch)


# ==============================================================================
# 第 4 部分：用占位模型前向传播，验证形状
# ==============================================================================
torch.manual_seed(123)  # 固定随机种子，保证每次运行结果可复现（dropout 等）

# 实例化占位模型
model_dummy = DummyGPTModel(GPT_CONFIG_124M)

# 前向传播：输入 (2, 5) 的 token id，输出 (2, 5, 50257) 的 logits
logits_dummy = model_dummy(batch)

print("\033[31m \n===== 输出：占位模型的前向输出形状与内容 =====\n\033[0m")
print("Output shape:", logits_dummy.shape)  # 应为 torch.Size([2, 5, 50257])
print(logits_dummy)


# ==============================================================================
# 第 5 部分：手写 LayerNorm 之前的"手动归一化"演示
# ==============================================================================
# 目的：先用最原始的方式理解 LayerNorm 在做什么，再封装成类。

torch.manual_seed(123)  # 固定随机种子

# 生成一个示例输入：2 个样本，每个样本 5 个特征
# torch.randn(2, 5)：从标准正态分布 N(0,1) 采样
batch_example = torch.randn(2, 5)

# 构造一个小型两层网络
# Linear(5, 6)：5 维 -> 6 维
# ReLU：负数变 0，正数不变
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

# 前向：输入 (2, 5) -> 输出 (2, 6)
out = layer(batch_example)

print("\n===== 输出：两层小网络的输出 =====\n")
print(out)


# ----- 手动求均值和方差 -----
# dim=-1：沿最后一维（特征维）求
# keepdim=True：保持形状为 (2, 1)，便于后面广播
mean = out.mean(dim=-1, keepdim=True)  # 形状 (2, 1)
var = out.var(dim=-1, keepdim=True)    # 形状 (2, 1)

print("\n===== 输出：每个样本的均值和方差 =====\n")
print("Mean:\n", mean)
print("Variance:\n", var)


# ----- 手动标准化 -----
# 公式：(x - mean) / sqrt(var)
# 广播机制：mean 和 var 形状 (2,1)，会自动扩展到 (2,6)
out_norm = (out - mean) / torch.sqrt(var)

# 再求一次均值和方差，验证是否接近 0 和 1
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

print("\033[31m \n===== 输出：手动归一化后的输出、均值、方差 =====\n\033[0m")
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)


# 关闭科学计数法，方便肉眼查看
torch.set_printoptions(sci_mode=False)

print("\033[32m \n===== 输出：关闭科学计数法后重新打印均值/方差 =====\n\033[0m")
print("Mean:\n", mean)
print("Variance:\n", var)


# ==============================================================================
# 第 6 部分：正式实现 LayerNorm
# ==============================================================================
# LayerNorm 原理：
#   1. 对每个样本（不是整个 batch）沿特征维做标准化
#   2. 再乘可学习的 scale（gamma）、加可学习的 shift（beta）
#   3. 好处：不依赖 batch 大小，适合序列模型；训练稳定
#   与 BatchNorm 区别：
#     - BatchNorm 沿 batch 维归一化，依赖 batch 大小，不适合 NLP
#     - LayerNorm 沿特征维归一化，每个样本独立，适合变长序列


class LayerNorm(nn.Module):
    """
    层归一化：对每个样本的特征维做标准化 + 仿射变换。
    """

    def __init__(self, emb_dim):
        super().__init__()
        # eps：防止除零的小常数，工程上通常取 1e-5
        self.eps = 1e-5

        # scale（gamma）：可学习的缩放参数，初始化为全 1
        # 形状 (emb_dim,)，对每个特征维度都有独立的缩放
        # nn.Parameter 会把张量注册为可学习参数，优化器会自动更新
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # shift（beta）：可学习的平移参数，初始化为全 0
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 1) 沿最后一维（特征维）求均值，keepdim 便于广播
        mean = x.mean(dim=-1, keepdim=True)

        # 2) 沿最后一维求方差
        # unbiased=False：用有偏估计（除以 N 而非 N-1）
        # 深度学习里通常用有偏方差，因为 N 通常很大，差别可忽略
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 3) 标准化：(x - mean) / sqrt(var + eps)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # 4) 仿射变换：scale * norm_x + shift
        # 让网络自己学习是否需要恢复一些原始分布
        return self.scale * norm_x + self.shift


# 实例化并测试
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)  # (2, 5) -> (2, 5)

# 验证归一化后均值和方差是否接近 0 和 1
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("\n===== 输出：LayerNorm 之后的均值与方差（应接近 0 和 1）=====\n")
print("Mean:\n", mean)
print("Variance:\n", var)


# ==============================================================================
# 第 7 部分：手写 GELU 激活函数
# ==============================================================================
# GELU 原理：
#   - 高斯误差线性单元，公式 GELU(x) = x * Φ(x)，Φ 是标准正态的 CDF
#   - 常用 tanh 近似：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#   - 相比 ReLU：
#       * ReLU 在 x<0 时梯度恒为 0，可能导致"神经元死亡"
#       * GELU 在 0 附近平滑过渡，保留少量负值信息
#   - 是 GPT/BERT 的标配激活函数


class GELU(nn.Module):
    """GELU 激活函数的 tanh 近似实现。"""

    def __init__(self):
        super().__init__()  # 没有参数

    def forward(self, x):
        # 逐元素计算：
        #   sqrt(2/π) ≈ 0.7978845608
        #   x^3 用 torch.pow(x, 3)
        #   tanh 用 torch.tanh
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# ----- 画图对比 GELU 与 ReLU -----
gelu, relu = GELU(), nn.ReLU()

# 在 [-3, 3] 生成 100 个等距点，用于画图
x = torch.linspace(-3, 3, 100)

y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)  # 1 行 2 列，第 i 个子图
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()  # 自动调整子图间距

print("\033[33m \n===== 输出：GELU 与 ReLU 激活函数曲线对比图 =====\n\033[0m")
plt.show()
sleep(2)
plt.close()


# ==============================================================================
# 第 8 部分：前馈网络 FeedForward
# ==============================================================================
# 前馈网络原理：
#   - Transformer 中每个 Block 有两个子层：
#       1) 注意力：混合不同 token 之间的信息
#       2) 前馈：对每个位置独立做非线性变换
#   - 标准做法：先升维（4 倍）再降维，形成"倒瓶颈"结构
#       * 升维增加表达能力，让 GELU 有更多空间做非线性
#       * 降维恢复到 emb_dim，方便残差相加
#   - 参数量：2 * emb_dim * 4 * emb_dim ≈ 8 * emb_dim^2


class FeedForward(nn.Module):
    """两层 MLP + GELU，是 Transformer Block 的第二个子层。"""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # 第一层：768 -> 3072（4 倍扩展）
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 非线性激活
            GELU(),
            # 第二层：3072 -> 768，恢复到原始维度
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
        print(f"\033[34m \n===== 输出：前馈网络参数量 =====\n\033[0m")
        total_params = sum(p.numel() for p in self.layers.parameters())
        print(f"\033[32m 总参数量: {total_params}\033[0m")
    def forward(self, x):
        # 逐层前向，形状不变：(..., emb_dim) -> (..., emb_dim)
        return self.layers(x)


# 测试
ffn = FeedForward(GPT_CONFIG_124M)

# 输入形状 (batch=2, seq_len=3, emb_dim=768)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("\033[32m \n===== 输出：前馈网络的输出形状 =====\n\033[0m")   
print(out.shape)  # 应为 torch.Size([2, 3, 768])


# ==============================================================================
# 第 9 部分：残差连接与梯度消失演示
# ==============================================================================
# 残差连接原理：
#   - 前向：y = x + f(x)，其中 f 是某个子层
#   - 反向：dy/dx = 1 + df/dx，梯度至少为 1，不会消失
#   - 作用：缓解深层网络的梯度消失，让模型可以堆到几十上百层
#   - GPT 每个 Block 有两处残差：注意力后、前馈后


class ExampleDeepNeuralNetwork(nn.Module):
    """
    用于演示残差连接效果的 5 层小网络。
    """

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut  # 是否启用残差

        # nn.ModuleList 是专门存子模块的列表，会被正确注册
        # 不能直接用 Python list，否则参数不会被 optimizer 看到
        self.layers = nn.ModuleList([
            # 5 层，每层是 Linear + GELU
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
        print(f"\033[34m \n===== 输出：小网络参数量 =====\n\033[0m")
        total_params = sum(p.numel() for p in self.layers.parameters())
        print(f"\033[32m 总参数量: {total_params}\033[0m")

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)  # 当前层输出

            # 若启用残差且形状一致，则 x = x + layer_output
            # 否则（第一层输入输出形状不同）直接替换
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    """
    打印每层权重的平均梯度绝对值，用于观察梯度消失现象。
    """
    # 前向传播
    output = model(x)

    # 目标值（假设真实标签是 0）
    target = torch.tensor([[0.]])

    # 均方误差损失
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    print(f"\033[32m 损失值: {loss.item()}\n\033[0m")
    print(f"\033[32m 输出: {output.item()}\n\033[0m")

    # 反向传播，计算梯度
    loss.backward()

    # 遍历所有参数，打印权重梯度的平均绝对值
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"\033[32m {name} has gradient mean of {param.grad.abs().mean().item()}\033[0m")


# 每层维度：3 -> 3 -> 3 -> 3 -> 3 -> 1
layer_sizes = [3, 3, 3, 3, 3, 1]

# 示例输入：1 个样本，3 个特征
sample_input = torch.tensor([[1., 0., -1.]])

# ---- 不使用残差 ----
torch.manual_seed(123)  # 固定权重初始化
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

print("\033[32m \n===== 输出：无 shortcut 时各层权重的平均梯度（会逐层变小，梯度消失）=====\n\033[0m")
print_gradients(model_without_shortcut, sample_input)

# ---- 使用残差 ----
torch.manual_seed(123)  # 同样种子，保证公平比较
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print("\033[32m \n===== 输出：有 shortcut 时各层权重的平均梯度（梯度更稳定）=====\n\033[0m")
print_gradients(model_with_shortcut, sample_input)


# ==============================================================================
# 第 10 部分：多头因果自注意力 MultiHeadAttention
# ==============================================================================
# 自注意力原理：
#   1. 把输入 x 线性投影得到 Q、K、V 三个矩阵
#        - Q (Query)：我在找什么
#        - K (Key)  ：我有什么
#        - V (Value)：实际内容
#   2. 计算注意力得分：scores = Q @ K^T / sqrt(d_k)
#        - 除以 sqrt(d_k) 是为了防止点积过大导致 softmax 梯度消失
#   3. 因果 mask：把未来位置的得分设为 -inf，softmax 后权重为 0
#        - GPT 是自回归模型，位置 i 只能看到 ≤i 的位置
#   4. softmax 归一化
#   5. 加权求和：output = attn_weights @ V
#   多头：
#     - 把 d_out=768 拆成 12 个头，每个头 64 维
#     - 不同头学习不同的注意力模式（比如有的关注语法、有的关注语义）
#     - 最后拼回去再做一次输出投影


class MultiHeadAttention(nn.Module):
    """多头因果自注意力。"""

    def __init__(self, d_in, d_out,
                context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # 断言：输出维度必须能被头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out                  # 输出总维度
        self.num_heads = num_heads          # 头数
        self.head_dim = d_out // num_heads  # 每个头的维度（768/12=64）

        # Q/K/V 的线性投影层：把 d_in 维映射到 d_out 维
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 输出投影：把多头拼接后的结果再映射一次
        self.out_proj = nn.Linear(d_out, d_out)

        # 对注意力权重做 dropout
        self.dropout = nn.Dropout(dropout)
        # print(f"\033[34m \n===== 输出：多头自注意力参数量Dropout =====\n\033[0m")
        # for i in self.dropout.parameters():
        #     print(f"\033[32m Dropout rate: {i}\033[0m")
        
        # 注册因果 mask：上三角矩阵（不含对角线）
        # torch.triu(..., diagonal=1)：取主对角线以上的元素
        # register_buffer：让 mask 跟随模型迁移到 GPU，但不是可学习参数
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        

    def forward(self, x):
        # 解包输入形状
        b, num_tokens, d_in = x.shape

        # 1) 线性投影得到 Q/K/V，形状都是 (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2) 拆成多头：
        #    把 d_out 拆成 (num_heads, head_dim)
        #    形状变为 (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3) 交换 num_tokens 和 num_heads 两个维度
        #    形状变为 (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4) 计算注意力得分：Q @ K^T
        #    keys.transpose(2, 3)：把 (b,h,T,d) 转成 (b,h,d,T)
        #    点积后形状 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 5) 取出当前序列长度对应的 mask 子矩阵
        #    因为 context_length=1024 可能比实际 seq_len 大
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 6) 把 mask 为 True 的位置填成负无穷，softmax 后权重为 0
        #    masked_fill_ 是原地操作，_ 后缀表示 in-place
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 7) 缩放 + softmax：除以 sqrt(head_dim) 防止点积过大
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # 8) 对注意力权重做 dropout
        attn_weights = self.dropout(attn_weights)

        # 9) 加权求和：attn_weights @ values
        #    形状 (b, num_heads, num_tokens, head_dim)
        #    再转置回 (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 10) 把多头拼回去：
        #     contiguous() 保证内存连续，view 才能用
        #     形状 (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 11) 输出投影，形状不变
        context_vec = self.out_proj(context_vec)
        return context_vec


# ==============================================================================
# 第 11 部分：Transformer Block
# ==============================================================================
# Transformer Block 原理：
#   - 由两个子层组成：多头自注意力 + 前馈网络
#   - 每个子层都采用 Pre-Norm + 残差结构：
#       x = x + sublayer(LayerNorm(x))
#   - Pre-Norm 比 Post-Norm 训练更稳定，GPT-2 采用
#   - 输入输出形状相同，这样才能堆叠 12 层


class TransformerBlock(nn.Module):
    """GPT 的 Transformer Block：注意力 + 前馈，各带残差和 LayerNorm。"""

    def __init__(self, cfg):
        super().__init__()

        # 多头自注意力子层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        # 前馈网络子层
        self.ff = FeedForward(cfg)

        # 两个 LayerNorm：分别用在注意力和前馈之前
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 残差分支上的 dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # ---- 第一个子层：注意力 ----
        shortcut = x              # 保存输入，用于残差
        x = self.norm1(x)         # 先归一化
        x = self.att(x)           # 多头注意力
        x = self.drop_shortcut(x) # dropout
        x = x + shortcut          # 残差相加

        # ---- 第二个子层：前馈 ----
        shortcut = x              # 更新残差基线
        x = self.norm2(x)         # 归一化
        x = self.ff(x)            # 前馈
        x = self.drop_shortcut(x) # dropout
        x = x + shortcut          # 残差相加

        return x


# 测试 Block
torch.manual_seed(123)
x = torch.rand(2, 4, 768)  # (batch=2, seq_len=4, emb_dim=768)

block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("\n===== 输出：TransformerBlock 的输入/输出形状 =====\n")
print("Input shape:", x.shape)
print("Output shape:", output.shape)  # 应仍是 (2, 4, 768)


# ==============================================================================
# 第 12 部分：完整 GPT 模型
# ==============================================================================
# 完整 GPT 结构：
#   1. token 嵌入 + 位置嵌入
#   2. dropout
#   3. N 个 TransformerBlock
#   4. 最终 LayerNorm
#   5. 输出头（映射到 vocab_size）
#
# 权重共享（weight tying）：
#   - 输出头的权重和 token 嵌入层共享
#   - 减少参数量（约 4 千万），且效果通常更好
#   - GPT-2 原版就是这么做的


class GPTModel(nn.Module):
    """完整的 GPT 模型。"""

    def __init__(self, cfg):
        super().__init__()

        # token 嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # 位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # 嵌入后的 dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠 n_layers 个真正的 TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终 LayerNorm
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 输出头：768 -> 50257，无 bias
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 【新增】权重共享：让输出层直接使用词嵌入的权重
        self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        # in_idx 形状：(batch, seq_len)
        batch_size, seq_len = in_idx.shape

        # 1) token 嵌入：(batch, seq_len) -> (batch, seq_len, 768)
        tok_embeds = self.tok_emb(in_idx)

        # 2) 位置嵌入：
        #    device=in_idx.device 自动与输入同设备（CPU/GPU）
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        # 3) 相加
        x = tok_embeds + pos_embeds

        # 4) dropout
        x = self.drop_emb(x)

        # 5) 12 个 TransformerBlock
        x = self.trf_blocks(x)

        # 6) 最终归一化
        x = self.final_norm(x)

        # 7) 输出 logits：(batch, seq_len, 50257)
        logits = self.out_head(x)
        return logits


# 实例化完整 GPT 模型
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

# 用前面构造的 batch 前向
out = model(batch)

print("\033[32m \n===== 输出：GPT 模型对输入 batch 的输出形状与内容 =====\n\033[0m")
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)  # (2, 5, 50257)
print(out)


# ==============================================================================
# 第 13 部分：统计参数量和模型大小
# ==============================================================================

# p.numel() 返回张量元素个数
# sum(...) 累加所有参数
total_params = sum(p.numel() for p in model.parameters())

print("\033[32m \n===== 输出：模型参数总量 =====\n\033[0m")
print(f"Total number of parameters: {total_params:,}")

# 查看 token 嵌入层和输出层的权重形状
print("\033[32m \n===== 输出：token 嵌入层与输出层的权重形状 =====\n\033[0m")
print("\033[32mToken embedding layer shape:", model.tok_emb.weight.shape)
print("\033[32mOutput layer shape:", model.out_head.weight.shape)

# 权重共享后：
#   如果输出头与 token 嵌入共享权重，则参数量要减去输出头
total_params_gpt2 = total_params - sum(
    p.numel() for p in model.out_head.parameters()
)

print("\033[32m \n===== 输出：考虑权重共享后的可训练参数量 =====\n\033[0m")
print("\033[32mNumber of trainable parameters considering weight tying: "
    f"{total_params_gpt2:,}")

# 假设每个参数是 float32（4 字节）
total_size_bytes = total_params * 4
# 转成 MB
total_size_mb = total_size_bytes / (1024 * 1024)

print("\n===== 输出：模型总大小（MB）=====\n")
print(f"Total size of the model: {total_size_mb:.2f} MB")


# ==============================================================================
# 第 14 部分：简单的文本生成函数
# ==============================================================================
# 生成原理（自回归 / 贪心解码）：
#   1. 把当前序列送进模型，得到每个位置的 logits
#   2. 只取最后一个位置的 logits（这是对"下一个 token"的预测）
#   3. softmax 得到概率分布
#   4. argmax 选概率最大的 token（贪心策略）
#   5. 把新 token 拼到序列末尾，循环
#
# 注意：
#   - 需要裁剪上下文到 context_size，防止位置嵌入越界
#   - 推理时要 model.eval() 关闭 dropout
#   - torch.no_grad() 关闭梯度计算，节省内存

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 【核心防复读机制】：如果某个词已经在句子里出现过了，就狠狠惩罚它的概率
        for i in range(idx.shape[1]):
            token_id = idx[0, i].item()
            logits[0, token_id] -= 5.0  # 惩罚力度加大

        # 取概率最高的前 20 个词，在里面随机抽一个
        top_k = 20
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
        
        # 用温度 0.7 做 softmax 并抽样
        probs = torch.softmax(top_logits / 0.7, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx_next = top_indices.gather(-1, idx_next)

        # 如果抽到了结束符，直接停止生成（防止它输出特别长）
        if idx_next.item() == 50256:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ---- 使用生成函数 ----
start_context = "Hello, I am a language model, and I can generate text."

# 编码起始文本
encoded = tokenizer.encode(start_context)

# 增加 batch 维度：(seq_len,) -> (1, seq_len)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print("\033[32m \n===== 输出：起始文本编码后的 token 列表与张量形状 =====\n\033[0m")
print("\033[32mencoded:", encoded)
print("\033[32mencoded_tensor.shape:", encoded_tensor.shape)

# 推理模式：关闭 dropout
model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,                     # 生成 6 个新 token
    context_size=GPT_CONFIG_124M["context_length"],
)

print("\033[31m \n===== 输出：生成后的 token 张量及序列长度 =====\n\033[0m")
print("\033[32mOutput:", out)
print("\033[33mOutput length:", len(out[0]))

# 把 token id 解码回文本
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print("\033[32m \n===== 输出：把生成的 token 解码回文本 =====\n\033[0m")
print("\033[32m", decoded_text)

# ==============================================================================
# 第 15 部分：手动加载 Hugging Face GPT-2 权重，并为特殊 Token 扩容
# ==============================================================================
from safetensors import safe_open

def load_hf_gpt2_weights(custom_model, safetensors_path):
    print(f"\n\033[34m正在从 {safetensors_path} 加载官方权重...\033[0m")
    hf_weights = {}
    
    # 1. 读取 safetensors 文件
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            hf_weights[k] = f.get_tensor(k)

    our_state_dict = custom_model.state_dict()

    # 2. 处理词嵌入层 (wte)
    # 原始权重是 (50257, 768)，我们的模型需要 (50260, 768)
    # 我们把前 50257 行替换为预训练权重，后 3 行（特殊 Token）保持随机初始化
    orig_wte = hf_weights["wte.weight"]
    our_state_dict["tok_emb.weight"][:50257, :] = orig_wte
    # 因为做了 weight tying，out_head 会自动跟着变，不用单独处理

    # 3. 处理位置嵌入
    our_state_dict["pos_emb.weight"] = hf_weights["wpe.weight"]

    # 4. 循环替换 12 层 Transformer
    for i in range(12):
        # 拆分合并的 QKV 权重
        # 【关键修复】先转置，把 (768, 2304) 变成 (2304, 768)，然后再按第 0 维切分
        qkv_weight = hf_weights[f"h.{i}.attn.c_attn.weight"].t()
        qkv_bias = hf_weights[f"h.{i}.attn.c_attn.bias"]
        
        # 按 dim=0 切分权重，每份 768
        q_w, k_w, v_w = torch.split(qkv_weight, 768, dim=0)
        # 偏置是一维的，直接用 chunk 均分成 3 份
        q_b, k_b, v_b = torch.chunk(qkv_bias, 3, dim=0)

        our_state_dict[f"trf_blocks.{i}.att.W_query.weight"] = q_w
        our_state_dict[f"trf_blocks.{i}.att.W_key.weight"] = k_w
        our_state_dict[f"trf_blocks.{i}.att.W_value.weight"] = v_w
        
        our_state_dict[f"trf_blocks.{i}.att.W_query.bias"] = q_b
        our_state_dict[f"trf_blocks.{i}.att.W_key.bias"] = k_b
        our_state_dict[f"trf_blocks.{i}.att.W_value.bias"] = v_b

        # 注意：官方权重有时是 Conv1D 格式，需要转置！
        # 检查形状，如果是 (768, 768) 就不转置；如果是 (768, 768) 但原本是 Conv1D，需要 .t()
        c_proj_w = hf_weights[f"h.{i}.attn.c_proj.weight"]
        if c_proj_w.shape == (768, 768):
            our_state_dict[f"trf_blocks.{i}.att.out_proj.weight"] = c_proj_w.t()
        our_state_dict[f"trf_blocks.{i}.att.out_proj.bias"] = hf_weights[f"h.{i}.attn.c_proj.bias"]

        # 前馈网络
        fc_w = hf_weights[f"h.{i}.mlp.c_fc.weight"]
        our_state_dict[f"trf_blocks.{i}.ff.layers.0.weight"] = fc_w.t()
        our_state_dict[f"trf_blocks.{i}.ff.layers.0.bias"] = hf_weights[f"h.{i}.mlp.c_fc.bias"]
        
        proj_w = hf_weights[f"h.{i}.mlp.c_proj.weight"]
        our_state_dict[f"trf_blocks.{i}.ff.layers.2.weight"] = proj_w.t()
        our_state_dict[f"trf_blocks.{i}.ff.layers.2.bias"] = hf_weights[f"h.{i}.mlp.c_proj.bias"]

        # LayerNorm
        our_state_dict[f"trf_blocks.{i}.norm1.scale"] = hf_weights[f"h.{i}.ln_1.weight"]
        our_state_dict[f"trf_blocks.{i}.norm1.shift"] = hf_weights[f"h.{i}.ln_1.bias"]
        our_state_dict[f"trf_blocks.{i}.norm2.scale"] = hf_weights[f"h.{i}.ln_2.weight"]
        our_state_dict[f"trf_blocks.{i}.norm2.shift"] = hf_weights[f"h.{i}.ln_2.bias"]

    # 最终 LayerNorm
    our_state_dict["final_norm.scale"] = hf_weights["ln_f.weight"]
    our_state_dict["final_norm.shift"] = hf_weights["ln_f.bias"]

    # 5. 装载回你的模型
    custom_model.load_state_dict(our_state_dict)
    print("\033[32m✅ 官方权重加载成功，且已为 3 个特殊 Token 留出空间！\033[0m")
    return custom_model

# # ---- 执行加载 ----
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device) # 在 load_hf_gpt2_weights 之后加上这个
# 请确保你已经把 model.safetensors 下载到了本地，并把路径填在这里
safetensors_path = model_path + "/model.safetensors"  # 例如 "./model.safetensors" 
model = load_hf_gpt2_weights(model, safetensors_path)

# 重新做一次推理测试
model.eval()
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
out = generate_text_simple(model, encoded_tensor, max_new_tokens=10, context_size=1024)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("\033[32m\n===== 加载权重后的生成测试 =====\033[0m")
print("\033[33m", decoded_text)


# ==============================================================================
# 第 16 部分：为纯手写模型注入 LoRA 并支持多权重挂载
# ==============================================================================
import glob

# 1. 定义带 LoRA 旁路的 Linear 层
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        # 冻结原权重
        for param in self.linear.parameters():
            param.requires_grad = False
            
        in_features = linear.in_features
        out_features = linear.out_features
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / r
        self.is_active = True  # 控制是否启用此 LoRA

    def forward(self, x):
        original_out = self.linear(x)
        if self.is_active:
            lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
            return original_out + lora_out
        return original_out

# 2. 注入 LoRA 到模型并管理多组权重
class MultiLoRAManager:
    def __init__(self, model, target_modules=["W_query", "W_key", "W_value"]):
        self.model = model
        self.lora_layers = [] # 记录所有被替换的 LoRA 层
        self.adapters = {}    # 存储加载的多个权重字典 { "adapter_name": { layer_idx: state_dict } }
        
        # 替换目标层
        for name, module in model.named_modules():
            # 检查是否是目标层（比如 W_query）
            layer_name = name.split('.')[-1]
            if isinstance(module, nn.Linear) and layer_name in target_modules:
                # 找到父模块，进行替换
                parent_name = '.'.join(name.split('.')[:-1])
                parent = model.get_submodule(parent_name)
                new_layer = LinearWithLoRA(module)
                setattr(parent, layer_name, new_layer)
                self.lora_layers.append(new_layer)
                
        print(f"✅ 已为 {len(self.lora_layers)} 个层注入 LoRA 旁路")

    def load_adapter(self, adapter_name, file_path):
        """加载外部训练好的 LoRA 权重"""
        print(f"📦 正在加载 LoRA 权重: {adapter_name} ({file_path})")
        state_dict = torch.load(file_path, map_location="cpu")
        
        # 如果只保存了 LoRA 权重，直接按顺序赋值（简化版演示）
        # 实际工程中建议用 state_dict 的 key 精确匹配
        lora_state = {}
        for k, v in state_dict.items():
            if "lora_A" in k or "lora_B" in k:
                lora_state[k] = v
                
        self.adapters[adapter_name] = lora_state
        print(f"✅ 权重 {adapter_name} 加载成功，包含 {len(lora_state)} 个 LoRA 张量")

    def switch_adapter(self, adapter_name):
        """动态切换使用哪个 LoRA 权重"""
        if adapter_name == "base":
            # 关闭所有 LoRA
            for layer in self.lora_layers:
                layer.is_active = False
            print("🔄 已切换到: 原始 GPT-2 (无 LoRA)")
            return
            
        if adapter_name not in self.adapters:
            print(f"❌ 找不到名为 {adapter_name} 的权重")
            return
            
        # 开启所有 LoRA
        for layer in self.lora_layers:
            layer.is_active = True
            
        # 加载指定权重 (这里是简化版，按顺序塞入，实际应用建议用 zip)
        state = self.adapters[adapter_name]
        # 在实际代码中，你需要根据 key 名精确匹配到对应的 layer
        # 这里假设你存的时候就是按层顺序存的
        for i, layer in enumerate(self.lora_layers):
            a_key = f"layer_{i}.lora_A"
            b_key = f"layer_{i}.lora_B"
            if a_key in state:
                layer.lora_A.data.copy_(state[a_key])
            if b_key in state:
                layer.lora_B.data.copy_(state[b_key])
                
        print(f"🔄 已切换到: {adapter_name} 的 LoRA 权重")
    def export_lora_weights(self, save_path):
        """把当前模型里的 LoRA 参数导出来存盘"""
        export_dict = {}
        for i, layer in enumerate(self.lora_layers):
            export_dict[f"layer_{i}.lora_A"] = layer.lora_A.data.cpu()
            export_dict[f"layer_{i}.lora_B"] = layer.lora_B.data.cpu()
        torch.save(export_dict, save_path)
        print(f"💾 权重已保存到 {save_path}")
# 初始化 LoRA 管理器
# 假设你有两个训练好的权重文件：lora_persona1.pt 和 lora_persona2.pt
# lora_manager = MultiLoRAManager(model)
# lora_manager.load_adapter("persona_1", "lora_persona1.pt")
# lora_manager.load_adapter("persona_2", "lora_persona2.pt")
# ==============================================================================
# 第 18 部分：微调（修正版）
# ==============================================================================
import random

file_path = "train_data.txt"
with open(file_path, "r", encoding="utf-8") as f:
    train_texts = [line.strip() for line in f if line.strip()]

print(f"📊 成功加载 {len(train_texts)} 条训练数据")

print("\n🔧 正在为模型注入 LoRA 旁路...")
lora_manager = MultiLoRAManager(model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 当前模型可训练参数总量: {trainable_params:,}")

print("\n🚀 开始微调...")
model.train()

# 【关键修正 1】只给 LoRA 参数，且学习率调大
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

for step in range(300): # 把步数增加到 300 步
    text = random.choice(train_texts)
    tokens = tokenizer.encode(text)[:128]
    input_ids = torch.tensor(tokens).unsqueeze(0)
    
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("✅ 微调完成！")
lora_manager.export_lora_weights("my_first_lora.pt")

# 激活你刚训好的权重
lora_manager.load_adapter("persona_1", "my_first_lora.pt")
lora_manager.switch_adapter("persona_1")

# ==============================================================================
# 第 17 部分：交互式多轮对话（修正版）
# ==============================================================================
def chat_loop(model, tokenizer, max_context=1024):
    history_tokens = []
    
    print("\n" + "="*50)
    print("🤖 GPT-2 纯手写模型对话已启动")
    print("输入 'quit' 退出")
    print("="*50 + "\n")
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 再见！")
            break
        if not user_input:
            continue
            
        # 【关键修正 2】用你训练数据里一模一样的格式！随便挑一条改
        prompt = f"<|persona|>i am a designer.<|user|>{user_input}<|bot|>"
        new_tokens = tokenizer.encode(prompt)
        
        input_ids_list = history_tokens + new_tokens
        
        if len(input_ids_list) > max_context:
            input_ids_list = input_ids_list[-max_context:]
            
        input_tensor = torch.tensor(input_ids_list).unsqueeze(0).to(next(model.parameters()).device)
        
        # 4. 生成回答
        with torch.no_grad():
            out_tensor = generate_text_simple(
                model=model, 
                idx=input_tensor, 
                max_new_tokens=30, 
                context_size=max_context
            )
            
        generated_ids = out_tensor[0][len(input_ids_list):].tolist()
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\nGPT-2: {response_text}")
        
        history_tokens = out_tensor[0].tolist()
        history_tokens.append(tokenizer.eos_token_id)

model.eval()
chat_loop(model, tokenizer)