# ============================================================
# mainGPT.py
# 来源: mainGPT.ipynb 合并导出
# 说明: 按原 notebook 单元格顺序排列，保留中文注释
# ============================================================

# ------------------------------------------------------------
# In[2] 初始化模型与配置
# ------------------------------------------------------------
from datetime import datetime, time
import importlib
import sys
sys.path.append('.')          # 把当前目录加入模块搜索路径，方便导入 mainCore.py
import mainCore
importlib.reload(mainCore)    # 重新加载 mainCore，避免修改后不生效
import torch
from mainCore import GPTModel
from typing import Any


# GPT-2 124M 的配置字典（为加快演示，这里把上下文长度缩短到 256）
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # 词表大小（GPT-2 的 BPE 词表）
    "context_length": 256,     # 上下文长度，从 1024 缩短到 256
    "emb_dim": 768,            # 嵌入维度
    "n_heads": 12,             # 多头注意力的头数
    "n_layers": 12,            # Transformer 层数
    "drop_rate": 0.1,          # dropout 比例
    "qkv_bias": False          # 注意力 QKV 是否使用偏置
}

torch.manual_seed(123)         # 固定随机种子，保证结果可复现
model = GPTModel(GPT_CONFIG_124M)
model.eval()                   # 切换到评估模式（关闭 dropout）


# ------------------------------------------------------------
# In[3] 文本 <-> token ID 转换工具函数
# ------------------------------------------------------------
import tiktoken
from mainCore import generate_text_simple

def text_to_token_ids(text, tokenizer):
    # 编码文本，允许 <|endoftext|> 作为特殊 token
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 增加 batch 维度
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 去掉 batch 维度
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

# 用简单的贪心生成函数做一次前向生成，看看未训练模型输出什么
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# ------------------------------------------------------------
# In[6] 手工构造一个 batch 的输入和目标
# ------------------------------------------------------------
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                    [40, 1107, 588]])      #  "I really like"]
# 目标：输入右移一位，即模型要预测的下一个 token
targets = torch.tensor([[3626, 6100, 345],    # [" effort moves you",
                        [1107, 588, 11311]])  #  " really like chocolate"]



#==================================================================================================

#==================================================================================================
# ------------------------------------------------------------
# In[7] 前向传播得到 logits 和概率
# ------------------------------------------------------------
with torch.no_grad():                          # 禁用梯度跟踪（当前不做训练）
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)         # 对最后一维做 softmax，得到每个 token 的概率
print(probas.shape)                            # 形状: (batch, seq_len, vocab_size)


# ------------------------------------------------------------
# In[8] 贪心取每个位置概率最大的 token
# ------------------------------------------------------------
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)


# ------------------------------------------------------------
# In[9] 对比目标文本与模型输出
# ------------------------------------------------------------
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# Targets batch 1: effort moves you
# Outputs batch 1: Armed heNetflix


# ------------------------------------------------------------
# In[10] 取出每个目标 token 对应的预测概率
# ------------------------------------------------------------
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)


# ------------------------------------------------------------
# In[11] 对目标概率取对数
# ------------------------------------------------------------
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)


# ------------------------------------------------------------
# In[12] 求平均对数概率
# ------------------------------------------------------------
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)


# ------------------------------------------------------------
# In[13] 取负号得到负对数似然损失
# ------------------------------------------------------------
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)


# ------------------------------------------------------------
# In[14] 查看 logits 和 targets 的形状
# ------------------------------------------------------------
print("Logits shape:", logits.shape)     # (batch, seq_len, vocab_size)
print("Targets shape:", targets.shape)   # (batch, seq_len)


# ------------------------------------------------------------
# In[15] 展平后计算交叉熵
# ------------------------------------------------------------
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)


# ------------------------------------------------------------
# In[16] 直接调用 PyTorch 交叉熵
# ------------------------------------------------------------
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)


# ------------------------------------------------------------
# In[18] 读取训练用文本《The Verdict》
# ------------------------------------------------------------
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


# ------------------------------------------------------------
# In[19] 统计字符数和 token 数
# ------------------------------------------------------------
total_characters = len(text_data)
# total_tokens = len(tokenizer.encode(text_data))
total_tokens = len(tokenizer.encode(text_data, allowed_special={'<|endoftext|>'}))

print("Characters:", total_characters)
print("Tokens:", total_tokens)


# ------------------------------------------------------------
# In[20] 按 90%/10% 划分训练集和验证集
# ------------------------------------------------------------
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


# ------------------------------------------------------------
# In[21] 构造训练/验证 DataLoader
# ------------------------------------------------------------
from chapter02 import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],   # 不重叠切分
    drop_last=True,                             # 丢掉最后一个不满的 batch
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# In[22] 打印一批数据的形状
# ------------------------------------------------------------
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)


# ------------------------------------------------------------
# In[23] 计算单个 batch 的损失
# ------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 数据搬到同一设备
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# ------------------------------------------------------------
# In[24] 计算 DataLoader 的平均损失
# ------------------------------------------------------------
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)              # 默认遍历所有 batch
    else:
        num_batches = min(num_batches, len(data_loader))  # 不能超过实际 batch 数
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()               # 累加每个 batch 的 loss
        else:
            break
    return total_loss / num_batches                 # 取平均


# ------------------------------------------------------------
# In[25] 选择设备并计算初始损失
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(device)

model.to(device)
with torch.no_grad():                               # 评估时不需要梯度
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])


# ------------------------------------------------------------
# In[29] 评估函数
# ------------------------------------------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                    # 评估模式：关闭 dropout
    with torch.no_grad():           # 关闭梯度，节省显存和计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()                   # 恢复训练模式
    return train_loss, val_loss


# ------------------------------------------------------------
# In[30] 生成一段样例文本
# ------------------------------------------------------------
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]    # 从位置嵌入推断上下文长度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))      # 单行紧凑打印
    model.train()


# ------------------------------------------------------------
# In[31] 主训练循环
# ------------------------------------------------------------
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                    eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []   # 记录损失和已见 token 数
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):                             # 外层：epoch 循环
        model.train()
        for input_batch, target_batch in train_loader:          # 内层：batch 循环
            optimizer.zero_grad()                               # 清空上一步梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                     # 反向传播求梯度
            optimizer.step()                                    # 更新参数
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:                    # 每隔 eval_freq 步评估一次
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(                              # 每个 epoch 结束生成一段样例
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


# ------------------------------------------------------------
# In[32] 正式训练
# ------------------------------------------------------------
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Every effort moves you", tokenizer=tokenizer
)


# ------------------------------------------------------------
# In[33] 绘制训练/验证损失曲线
# ------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()                               # 共享 y 轴的上方第二个 x 轴
    ax2.plot(tokens_seen, train_losses, alpha=0)    # 隐藏图形，仅用于对齐刻度
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# ------------------------------------------------------------
# In[34] 生成前切回 CPU
# ------------------------------------------------------------
model.to("cpu")
model.eval()


# ------------------------------------------------------------
# In[35] 用训练后的模型生成文本
# ------------------------------------------------------------
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# ------------------------------------------------------------
# In[36] 构造演示用小词表
# ------------------------------------------------------------
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}


# ------------------------------------------------------------
# In[37] 手工指定下一个 token 的 logits
# ------------------------------------------------------------
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)


# ------------------------------------------------------------
# In[38] 贪心选择
# ------------------------------------------------------------
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = int(torch.argmax(probas).item())
print(inverse_vocab[next_token_id])


# ------------------------------------------------------------
# In[39] 按概率随机采样
# ------------------------------------------------------------
torch.manual_seed(123)
next_token_id = int(torch.multinomial(probas, num_samples=1).item())
print(inverse_vocab[next_token_id])


# ------------------------------------------------------------
# In[40] 采样 1000 次统计频率
# ------------------------------------------------------------
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)


# ------------------------------------------------------------
# In[41] 温度缩放演示
# ------------------------------------------------------------
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]                                  # 原始 / 更尖锐 / 更平滑
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i],
                bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# In[42] top-k 过滤演示
# ------------------------------------------------------------
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)


# ------------------------------------------------------------
# In[43] 把 top-k 之外的 logits 置为 -inf
# ------------------------------------------------------------
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],   # 小于 top 3 最小值的 logits
    input=torch.tensor(float('-inf')),              # 置为负无穷
    other=next_token_logits                         # 其余保持不变
)
print(new_logits)


# ------------------------------------------------------------
# In[44] top-k 过滤后的概率
# ------------------------------------------------------------
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)


# ------------------------------------------------------------
# In[57] 带 temperature 和 top-k 的生成函数
# ------------------------------------------------------------
def generate(model, idx, max_new_tokens, context_size,
            temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]           # 只保留最近的 context_size 个 token
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                   # 只关心最后一个时间步
        if top_k is not None:                       # top-k 过滤
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:                       # 温度缩放 + 随机采样
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                       # 温度=0 时退化为贪心
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():   # 遇到 EOS 提前停止
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ------------------------------------------------------------
# In[58] 用 generate 生成文本
# ------------------------------------------------------------
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# ------------------------------------------------------------
# In[59] 只保存模型权重
# ------------------------------------------------------------
torch.save(model.state_dict(), "model.pth")


# ------------------------------------------------------------
# In[60] 加载模型权重
# ------------------------------------------------------------
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()


# ------------------------------------------------------------
# In[61] 同时保存模型和优化器
# ------------------------------------------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)


# ------------------------------------------------------------
# In[62] 加载模型和优化器
# ------------------------------------------------------------
checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()


# ------------------------------------------------------------
# In[63] 下载 gpt_download.py
# ------------------------------------------------------------
import urllib.request
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)
filename = url.split('/')[-1]
urllib.request.urlretrieve(url, filename)


# ------------------------------------------------------------
# In[65] 下载并加载 GPT-2 预训练权重
# ------------------------------------------------------------
from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"          # 下载到 ./gpt2/124M/ 目录
)
params: dict[str, Any] = params


print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())


# ------------------------------------------------------------
# In[66] 查看 settings 和 params 的键
# ------------------------------------------------------------
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())


# ------------------------------------------------------------
# In[67] 查看 token 嵌入权重
# ------------------------------------------------------------
print("Token embedding weight tensor:", params["wte"])

print("Token embedding weight tensor dimensions:", params["wte"].shape)


# ------------------------------------------------------------
# In[68] 定义不同 GPT-2 尺寸的配置
# ------------------------------------------------------------
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])


# ------------------------------------------------------------
# In[69] 更新上下文长度
# ------------------------------------------------------------
NEW_CONFIG.update({"context_length": 1024})


# ------------------------------------------------------------
# In[70] 更新 qkv_bias 并初始化 GPT 模型
# ------------------------------------------------------------
NEW_CONFIG.update({"qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()


# ------------------------------------------------------------
# In[71] 权重赋值辅助函数
# ------------------------------------------------------------
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# ------------------------------------------------------------
# In[72] 将 OpenAI 权重加载进我们的 GPT 模型
# ------------------------------------------------------------
import numpy as np
from typing import Dict, Any

def load_weights_into_gpt(gpt, params: Dict[str, Any]):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])   # 位置嵌入
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])   # token 嵌入
    for b in range(len(params["blocks"])):                           # 遍历每一层
        q_w, k_w, v_w = np.split(                                    # 拆分 QKV 权重
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # 权重共享


# ------------------------------------------------------------
# In[73] 加载权重到 gpt 并移到设备
# ------------------------------------------------------------
load_weights_into_gpt(gpt, params)
gpt.to(device)


# ------------------------------------------------------------
# In[74] 用 GPT-2 预训练权重生成文本
# ------------------------------------------------------------
torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))