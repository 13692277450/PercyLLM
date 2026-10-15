

from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("./personagpt2final")
print("len(tokenizer):", len(tok))
print(tok.encode("<|persona|>"))
print(tok.encode("<|user|>"))
print(tok.encode("<|bot|>"))

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("./personagpt2final")
print("model.config.vocab_size =", model.config.vocab_size)
print("wte行数 =", model.transformer.wte.weight.shape[0])


#==============================DOUBAO VERSION==============================

# import tiktoken
# from transformers import GPT2Tokenizer

# enc = tiktoken.get_encoding("gpt2")
# # 加载你微调项目的tokenizer
# hf_tok = GPT2Tokenizer.from_pretrained("./personagpt2final") 

# texts = ["Hello world", "who are u", "Every effort moves you", "<|persona|>"]
# for t in texts:
#     print(f"'{t}'")
#     print(f"  tiktoken: {enc.encode(t)}")
#     print(f"  HF:       {hf_tok.encode(t)}")
#     print(f"  一致: {enc.encode(t) == hf_tok.encode(t)}")


#==============================DEEPSEEK VERSION==============================

# import tiktoken
# from transformers import GPT2Tokenizer

# enc = tiktoken.get_encoding("gpt2")
# # hf_tok = GPT2Tokenizer.from_pretrained("gpt2")   # 从 Hub 下载
# hf_tok = GPT2Tokenizer.from_pretrained("./persona-gpt-final") # pyright: ignore[reportArgumentType]


# texts = ["Hello world", "who are u", "Every effort moves you"]
# for t in texts:
#     print(f"'{t}'")
#     print(f"  tiktoken: {enc.encode(t)}")
#     print(f"  HF:       {hf_tok.encode(t)}")
#     print(f"  一致: {enc.encode(t) == hf_tok.encode(t)}")