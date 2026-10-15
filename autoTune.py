# auto_tune.py
# 自动测试 /generate 接口，遍历参数组合，打分并输出最优参数
import itertools
import json
import time
import re
from collections import Counter

import requests
import pandas as pd

API_URL = "http://localhost:8000/generate"

# ---------------- 测试用 prompt 集 ----------------
# 覆盖不同类型的问题：问候、追问、指令、开放问题
TEST_PROMPTS = [
    ("i am a engineer.", "how are you"),
    ("i am a engineer.", "what's your job ?"),
    ("i am a engineer.", "how old are you?"),
    ("i am a engineer.", "what can you do ?"),
    ("i am a engineer.", "tell me a joke"),
    ("i am a engineer.", "what's the weather today ?"),
    ("i am a engineer.", "do you like music ?"),
    ("i am a engineer.", "explain what is python"),
]

# ---------------- 参数搜索空间 ----------------
TEMPERATURES = [0.5, 0.7, 0.8, 1.0, 1.2]
TOP_KS = [10, 30, 50, 100]
TOP_PS = [0.9, 0.95, 1.0]
REPETITION_PENALTIES = [1.0, 1.1, 1.2]

MAX_NEW_TOKENS = 50


# ---------------- 打分函数 ----------------
def tokenize_words(text: str):
    """简单按空白和标点切词"""
    return re.findall(r"[a-zA-Z']+", text.lower())


def ngram_repeat_ratio(tokens, n=3):
    """计算 n-gram 重复率：1 - 唯一 n-gram 数 / 总 n-gram 数"""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total


def score_one_answer(question: str, answer: str) -> dict:
    """
    对单条回答打分，返回各项指标和综合分。
    综合分越高越好，范围大致 0~1。
    """
    q_tokens = tokenize_words(question)
    a_tokens = tokenize_words(answer)

    # 1. 非空
    if not answer or not a_tokens:
        return {
            "length": 0,
            "repeat_ratio": 1.0,
            "overlap": 0.0,
            "ends_well": 0.0,
            "score": 0.0,
        }

    # 2. 长度分：5~40 词最好
    length = len(a_tokens)
    if length < 5:
        length_score = length / 5 * 0.5
    elif length <= 40:
        length_score = 1.0
    else:
        length_score = max(0.0, 1.0 - (length - 40) / 60)

    # 3. 重复率分：重复越低越好
    repeat = ngram_repeat_ratio(a_tokens, n=3)
    repeat_score = max(0.0, 1.0 - repeat * 2)  # 重复 50% 以上直接 0 分

    # 4. 与问题重叠：回答里是否包含问题关键词
    q_set = set(q_tokens)
    a_set = set(a_tokens)
    overlap = len(q_set & a_set) / max(1, len(q_set))
    overlap_score = min(1.0, overlap * 2)

    # 5. 结束合理性
    ends_well = 1.0 if answer.strip().endswith((".", "?", "!")) else 0.5

    # 6. 综合分（权重可调）
    score = (
        0.25 * length_score
        + 0.35 * repeat_score
        + 0.20 * overlap_score
        + 0.20 * ends_well
    )

    return {
        "length": length,
        "repeat_ratio": round(repeat, 3),
        "overlap": round(overlap, 3),
        "ends_well": ends_well,
        "score": round(score, 4),
    }


# ---------------- 调用一次接口 ----------------
def call_api(persona, user_text, temperature, top_k, top_p, rep_penalty):
    payload = {
        "persona": persona,
        "user_text": user_text,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": rep_penalty,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("generated_text", ""), None
    except Exception as e:
        return "", str(e)


# ---------------- 主流程 ----------------
def main():
    results = []

    combos = list(itertools.product(
        TEMPERATURES, TOP_KS, TOP_PS, REPETITION_PENALTIES
    ))
    print(f"共 {len(combos)} 组参数，每组 {len(TEST_PROMPTS)} 条 prompt，"
          f"总计 {len(combos) * len(TEST_PROMPTS)} 次请求\n")

    for idx, (temp, top_k, top_p, rep_pen) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] temp={temp}, top_k={top_k}, "
              f"top_p={top_p}, rep_pen={rep_pen}")

        scores = []
        lengths = []
        repeats = []
        errors = 0

        for persona, user_text in TEST_PROMPTS:
            answer, err = call_api(persona, user_text, temp, top_k, top_p, rep_pen)
            if err:
                errors += 1
                continue

            m = score_one_answer(user_text, answer)
            scores.append(m["score"])
            lengths.append(m["length"])
            repeats.append(m["repeat_ratio"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        avg_rep = sum(repeats) / len(repeats) if repeats else 1.0

        results.append({
            "temperature": temp,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": rep_pen,
            "avg_score": round(avg_score, 4),
            "avg_length": round(avg_len, 2),
            "avg_repeat_ratio": round(avg_rep, 3),
            "errors": errors,
        })
        print(f"    avg_score={avg_score:.4f}, avg_len={avg_len:.1f}, "
              f"avg_repeat={avg_rep:.3f}, errors={errors}")

        time.sleep(0.1)  # 稍微歇一下，避免压垮服务

    # ---------------- 输出结果 ----------------
    df = pd.DataFrame(results)
    df = df.sort_values("avg_score", ascending=False).reset_index(drop=True)
    df.to_csv("auto_tune_results.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("🏆 最优参数 Top 10")
    print("=" * 70)
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print("\n" + "=" * 70)
    print("✅ 最终推荐参数")
    print("=" * 70)
    print(f"  temperature         = {best['temperature']}")
    print(f"  top_k               = {int(best['top_k'])}")
    print(f"  top_p               = {best['top_p']}")
    print(f"  repetition_penalty  = {best['repetition_penalty']}")
    print(f"  综合得分            = {best['avg_score']}")
    print(f"  平均长度            = {best['avg_length']} 词")
    print(f"  平均重复率          = {best['avg_repeat_ratio']}")

    # 同时写一份 json，方便程序读取
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "temperature": float(best["temperature"]),
            "top_k": int(best["top_k"]),
            "top_p": float(best["top_p"]),
            "repetition_penalty": float(best["repetition_penalty"]),
            "avg_score": float(best["avg_score"]),
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到 auto_tune_results.csv 和 best_params.json")


if __name__ == "__main__":
    main()