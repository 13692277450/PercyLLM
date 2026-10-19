import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# 【修改 1】删掉 from peft import PeftModel

device = "cpu" # "cuda"
base_model_path = "./hfgpt2traindata/"
# 【修改 2】删掉 lora_path 变量，因为不需要了
tokenizer_path = "my_first_lora_tokenizer0"

# 加载保存好的tokenizer（依然需要它，因为它认识你加的 4 个特殊 Token）
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16
).to(device)

# 【修改 3】扩容词表！这是必须的，因为 tokenizer 有 50260 个词，
# 但原始模型只有 50257 个词的位置。如果不扩容，遇到特殊 Token 会报错。
base_model.resize_token_embeddings(len(tokenizer))

# 【修改 4】去掉 PeftModel 的包装，直接使用 base_model
# model = PeftModel.from_pretrained(base_model, lora_path).to(device) 
model = base_model
model.eval()

print("✅ 原厂基础模型加载成功（无 LoRA 权重）")

# 对话循环
print("="*50)
print("🤖 GPT-2 纯原厂对话已启动，输入 'quit' 退出")
print("="*50)

PERSONA = "i like to ski. my wife is a teacher."

while True:
    user_input = input("\n你: ").strip()
    if user_input.lower() == "quit":
        print("👋 退出对话")
        break
    if not user_input:
        continue

    # Prompt格式保持不变
    prompt = f"<|persona|>{PERSONA}<|user|>{user_input}<|bot|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.6,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")
        )
        
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
    bot_reply = full_text.split("<|bot|>")[-1].split("<|endoftext|>")[0].strip()
    print(f"GPT-2: {bot_reply}")