# chatclient.py
import requests
API_URL = "http://localhost:8000/generate"

# 在这里修改人设
PERSONA = "i am a engineer."

def chat():
    print("=" * 60)
    print("GPT-2 Persona 聊天客户端 (输入 'quit' 退出)")
    print("=" * 60)
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue
        
        payload = {
            "persona": PERSONA,
            "user_text": user_input,
            "max_new_tokens": 50,
            "temperature": 0.8,
            "top_k": 50,
        }
        
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            print(f"\nGPT-2: {data['generated_text']}")
        except requests.exceptions.RequestException as e:
            print(f"\n❌ 请求失败: {e}")
            # 打印服务返回的详细错误信息，方便调试422
            try:
                print("服务返回详情：", resp.text)
            except:
                pass

if __name__ == "__main__":
    chat()
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# tokenizer = AutoTokenizer.from_pretrained("./MicrosfotGPT2DialogptSmall")
# model = AutoModelForCausalLM.from_pretrained("./MicrosfotGPT2DialogptSmall")
# model.eval()

# # 注意：DialoGPT 训练时用 <|endoftext|> 作为分隔符
# # 这里我们把 persona 和 user 的话拼成一句自然语言
# prompt = "good morning. what's your function?" + tokenizer.eos_token

# inputs = tokenizer(prompt, return_tensors="pt")

# with torch.no_grad():
#     output_ids = model.generate(
#         inputs["input_ids"],
#         max_new_tokens=50,
#         do_sample=True,
#         temperature=0.8,
#         top_k=50,
#         pad_token_id=tokenizer.eos_token_id,
#     )

# # 只解码新生成的部分
# reply = tokenizer.decode(
#     output_ids[0][inputs["input_ids"].shape[-1]:],
#     skip_special_tokens=True
# )
# print("Bot:", reply)