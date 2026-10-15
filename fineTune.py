
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(os.environ.get('HF_ENDPOINT')) # 确认输出是镜像地址
# 然后再导入相关库
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from datasets import load_dataset
MODEL_DIR = "./my-gpt2-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载模型和 tokenizer
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 没有 pad_token

# 2. 添加对话专用的特殊 token
special_tokens = {"additional_special_tokens": ["<|persona|>", "<|user|>", "<|bot|>"]}
tokenizer.add_special_tokens(special_tokens)

# 3. 调整模型嵌入层大小，容纳新 token
model.resize_token_embeddings(len(tokenizer))

# 4. 移到设备
model.to(device) # pyright: ignore[reportArgumentType]

#=======================Loading dataset==================================


# 1. 加载 PersonaChat
print("Loading dataset...")
dataset = load_dataset("AlekseyKorshuk/persona-chat")
print("Loaded dataset: ",dataset)  # 看结构

# 2. 预处理函数：把对话转成 GPT-2 能训练的格式
def preprocess(example):
    personality = " ".join(example["personality"])
    utterances = example["utterances"]
    
    # 构造训练文本
    text = f"<|persona|>{personality}\n"
    for i, utt in enumerate(utterances):
        if i % 2 == 0:
            text += f"<|user|>{utt}\n"
        else:
            text += f"<|bot|>{utt}\n"
    text += tokenizer.eos_token   # 结尾加 EOS
    
    return {"text": text}

# 3. 应用预处理
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# 4. Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,          # 根据你的 context_length 调整
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True)

# 5. 设置 labels（GPT-2 是自回归，labels = input_ids）
def set_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example

tokenized = tokenized.map(set_labels)

#=======================Training==================================
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./persona-gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=4,      # 显存小就调小
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # 等效 batch_size=16
    learning_rate=5e-5,                 # 微调学习率要小
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),     # GPU 上开混合精度
    report_to="none",                   # 不上报 wandb
)

# 数据整理器（自动处理 padding 和 labels）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                          # GPT-2 是自回归，不是 MLM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized.get("validation", None),
    data_collator=data_collator,
)
#=======================Saving==================================

# 1. 开始训练
trainer.train()

# 2. 保存最终模型
trainer.save_model("./persona-gpt2-final")
tokenizer.save_pretrained("./persona-gpt2-final")

print("✅ 微调完成，模型保存到 ./persona-gpt2-final/")

#=======================Evaluation==================================
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./persona-gpt2-final")
tokenizer = GPT2Tokenizer.from_pretrained("./persona-gpt2-final")
model.eval()

# 构造输入
persona = "i like to ski. my wife is a teacher."
user_input = "hi, how are you?"
prompt = f"<|persona|>{persona}\n<|user|>{user_input}\n<|bot|>"

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate( # pyright: ignore[reportAttributeAccessIssue]
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
print("Generated response:", tokenizer.decode(outputs[0], skip_special_tokens=False))

#=======================Draw Conclusion==================================
'''

1. 加载本地权重 ./my-gpt2-hf/
2. 添加特殊 token <|persona|> <|user|> <|bot|>
3. resize_token_embeddings
4. 加载 PersonaChat 数据集
5. 预处理：把对话拼成带特殊 token 的文本
6. Tokenize + 设置 labels
7. 配置 TrainingArguments（lr=5e-5, epochs=3）
8. 用 Trainer 训练
9. 保存到 ./persona-gpt2-final/
10. 加载微调后模型测试对话


'''
