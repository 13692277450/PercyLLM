# chat_client.py
import requests
API_URL = "http://localhost:8000/generate"

# 固定人设，可以在这里修改
PERSONA = "i like to ski."

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
        "text": f"<|persona|>{PERSONA}<|user|>{user_input}<|bot|>",
        "max_length": 100,
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


# # chat_client.py
# import requests

# API_URL = "http://localhost:8000/generate"

# def chat():
#     print("=" * 60)
#     print("GPT-2 聊天客户端 (输入 'quit' 退出)")
#     print("=" * 60)
    
#     while True:
#         user_input = input("\n你: ").strip()
#         if user_input.lower() in ("quit", "exit", "q"):
#             print("再见！")
#             break
#         if not user_input:
#             continue
        
#         payload = {
#             "text": user_input,
#             "max_length": 100,
#             "temperature": 0.8,
#             "top_k": 50,
#         }
        
#         try:
#             resp = requests.post(API_URL, json=payload, timeout=60)
#             resp.raise_for_status()
#             data = resp.json()
#             print(f"\nGPT-2: {data['generated_text']}")
#         except requests.exceptions.RequestException as e:
#             print(f"\n❌ 请求失败: {e}")

# if __name__ == "__main__":
#     chat()