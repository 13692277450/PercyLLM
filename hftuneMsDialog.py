from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. 加载模型和 Tokenizer
print("\033[31mLoading model and tokenizer...\033[0m")
model_name = "./MicrosfotGPT2DialogptSmall"  # 或 gpt2
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
print("\33[32mModel and tokenizer loaded successfully.\033[0m")

# 2. 【关键】添加你自定义的特殊 Token
print("\033[31mAdding special tokens...\033[0m")
special_tokens = {"additional_special_tokens": ["<|persona|>", "<|user|>", "<|bot|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))  # 必须同步调整模型嵌入层大小
print("\033[33mSpecial tokens added successfully.\033[0m")

# 3. 准备数据（示例）
data = Dataset.from_dict({
    "text": [
        "<|persona|>i am a engineer.<|user|>how are you<|bot|>I'm doing well, thanks!<|endoftext|>",
        "<|persona|>i am a engineer.<|user|>what's your job?<|bot|>I design and build software.<|endoftext|>",
    ]
})


def tokenize(example):
    output = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    output["labels"] = output["input_ids"].copy()  # 因果语言建模，标签就是输入本身
    print("\033[34mTokenizer output:\033[0m", output)
    return output

data = data.map(tokenize)

# 4. 配置训练参数并开始
training_args = TrainingArguments(
    output_dir="./hfFinetunedChatbot",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=50,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
)

trainer.train()
print("\033[31mTraining completed.\033[0m")

trainer.save_model("./hfFinetunedChatbot")
tokenizer.save_pretrained("./hfFinetunedChatbot")
print("\033[32mTrainer saved model.\033[0m")
