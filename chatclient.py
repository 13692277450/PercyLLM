# chat_client.py
import requests

API_URL = "http://localhost:8000/generate"

def chat():
    print("=" * 60)
    print("GPT-2 聊天客户端 (输入 'quit' 退出)")
    print("=" * 60)
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not user_input:
            continue
        
        payload = {
            "text": user_input,
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

if __name__ == "__main__":
    chat()