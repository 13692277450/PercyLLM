

#==============================DEEPSEEK VERSION==============================

# serve_gpt2.py - GPT-2 服务（HuggingFace Tokenizer 版本 + 权重检查）
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json

app = FastAPI(title="GPT-2 Service (HF Tokenizer Edition)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_DIR = "./my-gpt2-hf"
MODEL_DIR = "./MicrosoftGPT2DialogptSmall"

print("=" * 70)
print("🚀 启动 GPT-2 服务 (HuggingFace Tokenizer Version)")
print("=" * 70)


# ============================================================
# 权重检查函数
# ============================================================
def check_model_weights(model_dir, model):
    """加载后检查模型权重是否正常"""
    print("\n🔍 权重检查开始...")
    issues = []
    

    # ---------- 1. 检查目录内容 ----------
    print(f"\n[1/5] 检查目录: {model_dir}")
    if not os.path.exists(model_dir):
        issues.append(f"❌ 目录不存在: {model_dir}")
        print(f"   ❌ 目录不存在")
        return issues
    files = os.listdir(model_dir)
    print(f"   目录内容: {files}")

    has_config = "config.json" in files
    has_weights = any(f in files for f in ["pytorch_model.bin", "model.safetensors"])
    print(f"   config.json: {'✅' if has_config else '❌'}")
    print(f"   权重文件: {'✅' if has_weights else '❌'}")
    if not has_config:
        issues.append("缺少 config.json")
    if not has_weights:
        issues.append("缺少 pytorch_model.bin 或 model.safetensors")

    # ---------- 2. 检查权重文件大小 ----------
    print(f"\n[2/5] 检查权重文件大小")
    for f in ["pytorch_model.bin", "model.safetensors"]:
        fp = os.path.join(model_dir, f)
        if os.path.exists(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"   {f}: {size_mb:.2f} MB")
            if size_mb < 400:
                issues.append(f"{f} 大小异常 ({size_mb:.1f}MB, 正常应约 500MB)")
                print(f"   ⚠️ 大小异常！GPT-2 124M 应约 500MB")

    # ---------- 3. 检查权重统计量 ----------
    print(f"\n[3/5] 检查权重统计量")
    try:
        # ============================================================
        # 提取模型三大核心权重张量（.data 表示只取数值，不包含计算图）
        # ============================================================

        # 1. 提取 Token 嵌入矩阵 (Word Token Embedding)
        # 形状通常是 (vocab_size, emb_dim)，比如 (50257, 768)
        # 作用：把输入的文字 ID 转换成模型能理解的 768 维稠密向量。
        wte = model.transformer.wte.weight.data

        # 2. 提取位置嵌入矩阵 (Word Position Embedding)
        # 形状通常是 (context_length, emb_dim)，比如 (1024, 768)
        # 作用：告诉模型每个词在句子中的相对或绝对位置（因为自注意力本身没有顺序概念）。
        wpe = model.transformer.wpe.weight.data

        # 3. 提取语言模型输出头矩阵 (Language Model Head)
        # 形状通常是 (vocab_size, emb_dim)，比如 (50257, 768)
        # 作用：把经过 12 层 Transformer 处理后的 768 维隐藏向量，重新映射回词表大小 (50257 维)，
        # 从而算出预测下一个词的概率（Logits）。
        lm_head = model.lm_head.weight.data

        print(f"   wte (token emb): shape={tuple(wte.shape)}, "
            f"mean={wte.mean().item():.6f}, std={wte.std().item():.6f}")
        print(f"   wpe (pos emb):   shape={tuple(wpe.shape)}, "
            f"mean={wpe.mean().item():.6f}, std={wpe.std().item():.6f}")
        print(f"   lm_head:         shape={tuple(lm_head.shape)}, "
            f"mean={lm_head.mean().item():.6f}, std={lm_head.std().item():.6f}")

        if wte.std().item() < 0.001:
            issues.append("wte 标准差过小，权重可能未正确加载")
            print(f"   ⚠️ wte std 过小，可能未加载")
        if wte.std().item() > 0.1:
            issues.append("wte 标准差过大，可能是随机初始化")
            print(f"   ⚠️ wte std 过大，可能随机初始化")

        if torch.allclose(lm_head, wte):
            print(f"   ✅ lm_head 与 wte 权重共享 (GPT-2 特性)")
        else:
            print(f"   ℹ️ lm_head 与 wte 不共享（部分实现不共享）")

    except Exception as e:
        issues.append(f"权重统计量检查失败: {e}")
        print(f"   ❌ 检查失败: {e}")

    # ---------- 4. 检查 config.json ----------
    print(f"\n[4/5] 检查 config.json")
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"   vocab_size: {cfg.get('vocab_size')}")
            print(f"   n_embd:     {cfg.get('n_embd')}")
            print(f"   n_layer:    {cfg.get('n_layer')}")
            print(f"   n_head:     {cfg.get('n_head')}")
            print(f"   n_positions:{cfg.get('n_positions')}")

            actual_vocab = model.transformer.wte.weight.shape[0]
            actual_embd = model.transformer.wte.weight.shape[1]
            actual_layers = len(model.transformer.h)
            if cfg.get("vocab_size") != actual_vocab:
                issues.append(f"vocab_size 不匹配: config={cfg.get('vocab_size')}, actual={actual_vocab}")
            if cfg.get("n_embd") != actual_embd:
                issues.append(f"n_embd 不匹配: config={cfg.get('n_embd')}, actual={actual_embd}")
            if cfg.get("n_layer") != actual_layers:
                issues.append(f"n_layer 不匹配: config={cfg.get('n_layer')}, actual={actual_layers}")
        except Exception as e:
            issues.append(f"config.json 读取失败: {e}")

    # ---------- 5. 检查 HF tokenizer 是否可用 加载一个模型后需要进行编码解码测试以确保tokenizer正常工作 ----------
    print(f"\n[5/5] 检查 HuggingFace tokenizer")
    try:
        tok = GPT2Tokenizer.from_pretrained(model_dir)
        test_texts = ["Hello world", "who are u?", "Every effort moves you"]
        for t in test_texts:
            ids = tok.encode(t)
            decoded = tok.decode(ids, skip_special_tokens=True)
            ok = "✅" if ids else "❌"  #一切皆可布尔化
            print(f"   {ok} '{t}': ids={ids}, decoded='{decoded}'")
            if not ids:
                issues.append(f"tokenizer 编码失败: '{t}' 返回空")
    except Exception as e:
        issues.append(f"tokenizer 加载失败: {e}")
        print(f"   ⚠️ 无法加载: {e}")

    # ---------- 总结 ----------
    print(f"\n{'=' * 50}")
    if issues:
        print(f"⚠️ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("✅ 权重检查全部通过！")
    print(f"{'=' * 50}")
    return issues


# ============================================================
# 1. 加载模型
# ============================================================
print("\n📦 步骤 1/3: 加载模型权重...")
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"❌ 模型目录不存在: {MODEL_DIR}\n"
        f"   请确认路径，或先用 pth2transformer.py 转换权重。"
    )

model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)  # pyright: ignore[reportArgumentType]
print("   ✅ 模型加载完成")

# 运行权重检查
weight_issues = check_model_weights(MODEL_DIR, model)


# ============================================================
# 2. 加载 HuggingFace tokenizer
# ============================================================
print("\n🔤 步骤 2/3: 加载 HuggingFace tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

# GPT-2 没有 pad_token，用 eos_token 代替
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"   ✅ Tokenizer 加载成功!")
print(f"      词汇表大小: {tokenizer.vocab_size}")
print(f"      pad_token_id: {tokenizer.pad_token_id}")
print(f"      eos_token_id: {tokenizer.eos_token_id}")

# 测试 tokenizer
print("\n🧪 测试 HF tokenizer...")
test_texts = ["Hello world", "Every effort moves you"]
for test_text in test_texts:
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"   '{test_text}' → ids={ids} → decoded='{decoded}'")
    if not ids:
        raise ValueError(f"❌ tokenizer 编码失败: '{test_text}' 返回空!")
print("   ✅ tokenizer 测试通过")


# ============================================================
# 3. 模型切到评估模式
# ============================================================
print("\n🎯 步骤 3/3: 模型切到评估模式...")
model.eval()
print("   ✅ 完成")

print(f"\n{'=' * 70}")
print("✅ 服务启动完成!")
print(f"   设备: {device}")
print(f"   模型: GPT-2 (124M)")
print(f"   Tokenizer: HuggingFace GPT2Tokenizer")
print(f"{'=' * 70}\n")


# ============================================================
# API 定义
# ============================================================
class GenerateRequest(BaseModel):
    text: str
    max_length: int = 50
    temperature: float = 1.0
    top_k: int = 50


class GenerateResponse(BaseModel):
    generated_text: str


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        print(f"\n{'─' * 70}")
        print(f"📝 收到生成请求")
        print(f"   输入: '{request.text}'")
        print(f"   参数: max_length={request.max_length}, temp={request.temperature}, top_k={request.top_k}")

        if not request.text or not request.text.strip():
            return _error_response("输入文本不能为空!")

        # Tokenization（HF tokenizer）
        print(f"   🔤 Tokenization (HF GPT2Tokenizer)...")
        inputs = tokenizer(request.text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        ids = inputs['input_ids']
        print(f"   ✅ Token IDs: {ids.tolist()}")
        print(f"   Shape: {ids.shape}, 元素数: {ids.numel()}")

        if ids.numel() == 0:
            return _error_response("Tokenization 返回空结果!")

        # 生成
        print(f"   🚀 生成中...")
        with torch.no_grad():
            outputs = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"   ✅ 生成完成!")
        print(f"   📄 输出: '{generated_text[:120]}...'")
        print(f"{'─' * 70}\n")
        return GenerateResponse(generated_text=generated_text)

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'─' * 70}\n")
        return _error_response(str(e))


def _error_response(message: str):
    from fastapi.responses import JSONResponse
    import traceback
    return JSONResponse(
        status_code=500,
        content={"error": message, "traceback": traceback.format_exc()}
    )


@app.get("/")
async def root():
    return {
        "message": "✅ GPT-2 Service (HF Tokenizer Edition) is running!",
        "status": "healthy",
        "endpoints": ["/generate", "/health", "/test-tokenizer"],
        "model": "GPT-2 (124M Custom)",
        "tokenizer": "HuggingFace GPT2Tokenizer",
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "weight_issues": weight_issues,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "tokenizer_ready": True,
        "tokenizer_type": "HuggingFace GPT2Tokenizer",
        "device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "weight_issues": weight_issues,
    }


@app.get("/test-tokenizer")
async def test_tokenizer_endpoint(text: str = "Hello world"):
    try:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        return {
            "success": True,
            "input_text": text,
            "token_ids": ids,
            "num_elements": len(ids),
            "decoded_back": decoded,
            "match": text == decoded,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("🌐 启动 HTTP 服务器...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)