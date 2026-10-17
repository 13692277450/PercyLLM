from dotenv import load_dotenv
load_dotenv()   # 默认加载当前目录的 .env

import os
import requests

url = "https://mp-staging.kaggle.net/models/openapi/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {os.environ['MODEL_PROXY_API_KEY']}",
    "Content-Type": "application/json",
}
body = {
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
}

resp = requests.post(url, headers=headers, json=body)
print("状态码:", resp.status_code)
print("响应:", resp.text)