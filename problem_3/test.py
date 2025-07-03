import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === 加载模型 ===
model_path = "../../.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()

# === 选择一个新闻进行测试 ===
# 选择数据中的一条新闻
test_df = pd.read_csv("./fake-and-real/True.csv").dropna(subset=["text"]).sample(n=1, random_state=42)  # 只选择一条真新闻
test_text = test_df['text'].iloc[0]  # 获取新闻文本

# === 预测函数 ===
def classify_news(text):
    
    prompt = f"你是一个语义情感识别专家，请分析下面这段新闻的语义情感：\n\n{text.strip()}\n\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        # 增加生成的token数量，允许生成更多文本
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,  # 允许更多生成的内容
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 输出模型的回答
    answer = output_text.replace(prompt, "").strip()

    print("=" * 60)
    print("📌 Prompt:")
    print(prompt)
    print("🤖 模型回答:")
    print(answer)

    if "真" in answer and "假" not in answer:
        return 1
    elif "假" in answer:
        return 0
    else:
        return -1

# === 测试模型 ===
pred = classify_news(test_text)
print(pred)