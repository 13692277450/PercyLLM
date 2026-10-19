#Doubao
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "cpu" #"cuda"
base_model_path = "./hfgpt2traindata/"
lora_path = "my_first_lora0.pt"
tokenizer_path = "my_first_lora_tokenizer0"

# 加载保存好的tokenizer（重点！！）
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型+LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16
).to(device)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, lora_path).to(device)
model.eval()

print("✅ 权重 persona_1 加载成功")

# 对话循环
print("="*50)
print("🤖 GPT-2 人设对话已启动，输入 'quit' 退出")
print("="*50)

# 你的人设，和训练数据集保持风格一致
PERSONA = "i like to ski. my wife is a teacher."

while True:
    user_input = input("\n你: ").strip()
    if user_input.lower() == "quit":
        print("👋 退出对话")
        break
    if not user_input:
        continue

    # ✅ Prompt格式【和训练完全一模一样！】
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
    # 截取bot回答
    bot_reply = full_text.split("<|bot|>")[-1].split("<|endoftext|>")[0].strip()
    print(f"GPT-2: {bot_reply}")
