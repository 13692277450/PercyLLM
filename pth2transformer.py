from transformers import GPT2LMHeadModel, GPT2Config

# 1. 用和你的模型一致的配置，创建 HF 的 GPT-2 模型
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,
    n_head=12,
    n_layer=12,
)
hf_model = GPT2LMHeadModel(config)

# 2. 加载你自己训练的 .pth 权重
import torch
state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)

# 关键一步：把你自己模型的参数名，映射到 HF 模型的参数名
# 如果你训练时就是按 HF 的命名保存的，可以直接 load
hf_model.load_state_dict(state_dict, strict=False)

# 3. 保存成标准 HF 格式
hf_model.save_pretrained("my-gpt2-hf")