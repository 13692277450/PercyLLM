#doubao split
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# ===================== 1. GPU检测 =====================
print("="*70)
print("GPU信息检测")
gpu_count = torch.cuda.device_count()
print(f"GPU数量: {gpu_count}")
for i in range(gpu_count):
    print(f"GPU{i}: {torch.cuda.get_device_name(i)}")
print("="*70)

# ===================== 2. Tokenizer 加载 + 注册自定义特殊Token【核心修复】 =====================
model_name_or_path = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# 注册人设对话标记
special_tokens_dict = {
    "additional_special_tokens": ["<|persona|>", "<|user|>", "<|bot|>", "<|endoftext|>"]
}
tokenizer.add_special_tokens(special_tokens_dict)

# ✅ 校验：打印特殊token编号，确认是独立token！
print("\n✅ 特殊Token ID校验：")
print("<|persona|> id =", tokenizer.convert_tokens_to_ids("<|persona|>"))
print("<|user|>    id =", tokenizer.convert_tokens_to_ids("<|user|>"))
print("<|bot|>     id =", tokenizer.convert_tokens_to_ids("<|bot|>"))
print("<|endoftext|> id =", tokenizer.convert_tokens_to_ids("<|endoftext|>"))

# ===================== 3. 加载基础模型，扩容词表 =====================
print("\n===== 原始未加载权重生成 =====")
test_raw_prompt = "<|persona|>I like to ski<|user|>how are you?<|bot|>"
inputs_raw = tokenizer(test_raw_prompt, return_tensors="pt").to("cuda")

print("正在从 /kaggle/input/datasets/mangzhang/hfgpt2traindata/hfgpt2traindata//model.safetensors 加载官方权重...")
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/datasets/mangzhang/hfgpt2traindata/hfgpt2traindata/",
    torch_dtype=torch.float16
).to("cuda")

# 必须扩容embedding，匹配新增token
model.resize_token_embeddings(len(tokenizer))
print("✅ 官方权重加载成功！")

# 加载权重后简单生成测试
print("\n===== 加载权重后的生成测试 =====")
# 开启推理模式，告诉 PyTorch 不需要计算和存储梯度（省显存、加速）
with torch.no_grad():
    # 调用 Hugging Face 封装好的 generate 方法生成文本
    out_raw = model.generate(
        **inputs_raw,               # 输入是之前 tokenize 过的 prompt 字典 (input_ids, attention_mask)
        max_new_tokens=15,          # 限制模型最多只生成 15 个新 Token
        do_sample=False,            # 关闭随机采样，使用贪心解码（每次只选概率最高的词，输出最稳定）
        pad_token_id=tokenizer.eos_token_id  # 告诉模型在遇到 batch 大小不一致时，用结束符 ID 作为填充符
    )

# 把模型吐出的 token ID 列表解码回人类能看懂的文本
# skip_special_tokens=False 表示保留特殊 Token（如 <|persona|>, <|bot|>），方便我们观察格式是否被破坏
print(tokenizer.decode(out_raw[0], skip_special_tokens=False))

# ===================== 4. LoRA配置【修复！冻结主干，只训练LoRA】 =====================
# 从 peft 库导入 LoraConfig（配置类）和 TaskType（任务类型枚举）
# ===================== 4. LoRA配置【修复！冻结主干，只训练LoRA】 =====================
lora_config = LoraConfig(
    r=8,                        # LoRA 的秩（Rank）。决定了旁路矩阵的大小。r 越小参数越少，r=8 是微调小模型的黄金标准
    lora_alpha=16,              # 缩放系数。决定了 LoRA 旁路对原始输出的影响程度。通常设为 r 的 2 倍
    target_modules=["c_attn"],  # 指定要把 LoRA 加在哪些层上。GPT-2 把 Q、K、V 合并成了一个 c_attn 层，所以只填它
    lora_dropout=0.05,          # 在 LoRA 旁路上做 5% 的随机丢弃，防止模型只死记硬背这几个词，增强泛化
    bias="none",                # 不训练偏置项（bias），进一步减少参数量
    task_type=TaskType.CAUSAL_LM, # 明确告诉 peft：这是一个因果语言模型（预测下一个词），它内部会自动调整适配逻辑
    inference_mode=False        # False 表示这是训练模式（旁路矩阵需要更新）；如果是推理，设为 True 可以融合权重
)
model = get_peft_model(model, lora_config)
print("\n🔧 正在为模型注入 LoRA 旁路...")
model.print_trainable_parameters() # 这里正常应该是 ~0.3% 参数量（几十万参数，不是1亿！）
print("✅ LoRA注入完成")

# ===================== 5. 加载你的5000条训练数据【沿用你原有数据集路径】 =====================
# 如果你是文件夹内有txt，改用load_dataset("text", data_files="xxx.txt")
# 这里适配你原有：5000条样本，格式：<|persona|>...<|user|>...<|bot|>...<|endoftext|>
dataset_raw = load_dataset(
    "text",
    data_files="train_data.txt",
    split="train"
)
print(f"\n📊 成功加载 {len(dataset_raw)} 条训练数据")

def tokenize_function(examples):
    # examples 是一个字典，因为使用了 batched=True，examples["text"] 是一个包含多条文本的列表
    return tokenizer(
        examples["text"],      # 输入文本列表
        truncation=True,       # 开启截断。如果某句话超过了 max_length，直接砍掉多余部分
        max_length=256         # 设定单条数据的最大长度。超过 256 个 token 就去掉。
        # 移除 padding="max_length" 是极其英明的决定！
    )

# 用 map 函数把分词逻辑应用到整个数据集
# batched=True 表示一次喂给 tokenize_function 一批数据（比如 1000 条），速度比一条一条处理快几十倍
tokenized_dataset = dataset_raw.map(tokenize_function, batched=True)

# 自回归LM数据收集器，自动动态padding
# 从 transformers 库中导入专门处理语言模型数据的整理器
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling( #Hugging Face Trainer 能够高效运转的核心枢纽。
            #DataCollatorForLanguageModeling 的字面意思是“用于语言建模的数据整理器”。在微调 GPT-2 的过程中，它主要扮演“打包员”和“出题人”两个角色。
    tokenizer=tokenizer, # 传入你的分词器，它需要用到里面的 pad_token_id
    mlm=False            # 极其关键：mlm=False 代表“因果语言模型”（Causal LM），如果是 True 代表“掩码语言模型”（如 BERT）
)
# ===================== 6. 训练参数【修复：降低学习率，DDP双卡支持】 =====================
training_args = TrainingArguments(
    output_dir="./persona_gpt2_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,        # 原2e-4 → 降到1e-4，防止loss爆炸
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="epoch",
    fp16=True,
    ddp_find_unused_parameters=False,
    report_to="none",
)

# ===================== 7. Trainer 训练 =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

if __name__ == "__main__":
    print("\n🚀 开始微调...")
    trainer.train()
    print("\n✅ 微调完成！")
    # 保存LoRA权重
    trainer.save_model("my_first_lora0.pt")
    tokenizer.save_pretrained("my_first_lora_tokenizer0")
    print("💾 LoRA权重保存至 my_first_lora0.pt，tokenizer保存在 my_first_lora_tokenizer0")
