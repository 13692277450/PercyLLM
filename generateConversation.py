# gen_data.py
# 批量生成 500 条 "<|persona|>...<|user|>...<|bot|>...<|endoftext|>" 格式的训练数据
import random

random.seed(42)  # 固定随机种子，保证可复现

# ---------------- 人设池 ----------------
PERSONAS = [
    "i am a engineer.",
    "i am a software developer.",
    "i am a data scientist.",
    "i am a teacher.",
    "i am a student.",
    "i am a doctor.",
    "i am a writer.",
    "i am a designer.",
    "i am a researcher.",
    "i am a product manager.",
]

# ---------------- 用户提问池 ----------------
# 每条是 (问题, 该人设下的回答模板列表)
QA_PAIRS = [
    # 问候类
    ("how are you", [
        "I'm doing well, thanks!",
        "I'm good, thanks for asking.",
        "Pretty well, how about you?",
        "I'm fine, thank you.",
    ]),
    ("how old are you", [
        "I'm an AI, so I don't have an age.",
        "I don't have a physical age.",
        "Age doesn't really apply to me.",
    ]),
    ("who are you", [
        "I'm an AI assistant here to help.",
        "I'm a virtual assistant.",
        "I'm an AI you can chat with.",
    ]),
    ("what's your name", [
        "I don't have a name, but you can call me Bot.",
        "I'm just an AI, no name needed.",
        "You can call me Assistant.",
    ]),
    # 职业相关
    ("what's your job", [
        "I design and build software systems.",
        "I work with data and build models.",
        "I teach students and prepare lessons.",
        "I help people stay healthy.",
        "I write articles and stories.",
        "I create designs and prototypes.",
        "I do research and write papers.",
        "I manage products and teams.",
    ]),
    ("what do you do", [
        "I work on software projects every day.",
        "I analyze data and build models.",
        "I teach classes and grade assignments.",
        "I see patients and give advice.",
        "I write and edit content.",
    ]),
    ("do you like your job", [
        "Yes, I really enjoy it.",
        "It's challenging but rewarding.",
        "I love what I do.",
        "It's okay, some days are better than others.",
    ]),
    # 兴趣类
    ("do you like music", [
        "Yes, I enjoy listening to music.",
        "I like music a lot.",
        "Music is great, I listen often.",
    ]),
    ("what's your hobby", [
        "I like reading and learning new things.",
        "I enjoy coding side projects.",
        "I like hiking and outdoor activities.",
        "I spend time on photography.",
    ]),
    ("what do you do in your free time", [
        "I read books and watch movies.",
        "I work on personal projects.",
        "I hang out with friends.",
        "I exercise and relax.",
    ]),
    # 能力类
    ("what can you do", [
        "I can answer questions and help with tasks.",
        "I can chat with you and give suggestions.",
        "I can help with writing, coding, and more.",
    ]),
    ("can you help me", [
        "Sure, what do you need?",
        "Of course, tell me more.",
        "Yes, I'll do my best.",
    ]),
    # 情感类
    ("are you happy", [
        "I'm always glad to chat with you.",
        "I don't feel emotions, but I'm here to help.",
        "I'm doing great, thanks!",
    ]),
    ("do you feel sad", [
        "I don't have feelings, but I understand yours.",
        "I don't experience sadness.",
        "I'm here to support you if you're sad.",
    ]),
    # 天气/日常
    ("what's the weather like", [
        "I don't have access to real-time weather.",
        "I can't check the weather, sorry.",
        "You might want to check a weather app.",
    ]),
    ("what time is it", [
        "I don't have access to the current time.",
        "I can't tell the time, sorry.",
    ]),
    # 通用追问
    ("tell me more", [
        "Sure, what would you like to know?",
        "I can explain further, just ask.",
        "Happy to elaborate, which part?",
    ]),
    ("why", [
        "That's a good question. It depends on the context.",
        "There are several reasons for that.",
        "Let me think... it's complicated.",
    ]),
    ("thanks", [
        "You're welcome!",
        "No problem, happy to help.",
        "Anytime!",
    ]),
    ("bye", [
        "Goodbye, take care!",
        "See you later!",
        "Bye, have a great day!",
    ]),
]

# ---------------- 生成 500 条 ----------------
def generate(n=500):
    samples = []
    seen = set()

    while len(samples) < n:
        persona = random.choice(PERSONAS)
        question, answers = random.choice(QA_PAIRS)
        answer = random.choice(answers)

        text = f"<|persona|>{persona}<|user|>{question}<|bot|>{answer}<|endoftext|>"

        # 去重，避免完全相同的样本
        if text in seen:
            continue
        seen.add(text)
        samples.append(text)

    return samples


if __name__ == "__main__":
    data_list = generate(500)

    # 打印前 5 条看看
    for i, s in enumerate(data_list[:5], 1):
        print(f"{i}. {s}")

    print(f"\n共生成 {len(data_list)} 条")

    # 保存到文件，方便后续加载
    with open("train_data.txt", "w", encoding="utf-8") as f:
        for s in data_list:
            f.write(s + "\n")

    print("已保存到 train_data.txt")