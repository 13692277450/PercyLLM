import tiktoken
from transformers import GPT2Tokenizer

enc = tiktoken.get_encoding("gpt2")
# hf_tok = GPT2Tokenizer.from_pretrained("gpt2")   # 从 Hub 下载
hf_tok = GPT2Tokenizer.from_pretrained("./my-gpt2-hf") # pyright: ignore[reportArgumentType]


texts = ["Hello world", "who are u", "Every effort moves you"]
for t in texts:
    print(f"'{t}'")
    print(f"  tiktoken: {enc.encode(t)}")
    print(f"  HF:       {hf_tok.encode(t)}")
    print(f"  一致: {enc.encode(t) == hf_tok.encode(t)}")