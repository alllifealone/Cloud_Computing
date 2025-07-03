import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import numpy as np

# === 加载模型 ===
model_path = "../../.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1__5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()

# === 读取并采样数据（每类各100条）===
true_df = pd.read_csv("./fake-and-real/True.csv").dropna(subset=["text"]).sample(n=500, random_state=42)
fake_df = pd.read_csv("./fake-and-real/Fake.csv").dropna(subset=["text"]).sample(n=500, random_state=42)
true_df['label'] = 1
fake_df['label'] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df[['text', 'label']].dropna()

# === 预测函数 ===
def classify_news(text): 
    prompt = f"你是一个虚假新闻识别专家，请先对新闻内容进行情感分析，再判断下面这段新闻是真新闻还是假新闻，只能用“真”或“假”回答，不要回答多余内容：\n\n{text.strip()}\n\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        # 增加生成的token数量，允许生成更多文本
        outputs = model.generate(
            **inputs,
            max_new_tokens=2,  # 允许更多生成的内容
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = output_text.replace(prompt, "").strip()

    # print("=" * 60)
    # print("📌 Prompt:")
    # print(prompt)
    # print("🤖 模型回答:")
    # print(answer)

    if "真" in answer and "假" not in answer:
        return 1
    elif "假" in answer and "真" not in answer:
        return 0
    else:
        return -1

# === 执行预测 ===
predictions = []
for text in df['text']:
    pred = classify_news(text)
    # print(pred)
    predictions.append(pred)
df['prediction'] = predictions

# === 过滤掉无法判断的数据 ===
df = df[df['prediction'] != -1]

# === 指标计算 ===
y_true = df['label'].values
y_pred = df['prediction'].values

accuracy = (y_true == y_pred).mean()
f1 = f1_score(y_true, y_pred)

# confusion_matrix 结构：[ [TN, FP], [FN, TP] ]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

try:
    auc = roc_auc_score(y_true, y_pred)
except ValueError:
    auc = float('nan')

# === 输出结果 ===
print("\n" + "=" * 60)
print(f"✅ 总体准确率 Accuracy: {accuracy:.2%}")
print(f"✅ F1-score: {f1:.2%}")
print(f"✅ G-mean: {g_mean:.2%}")
print(f"✅ AUC: {auc:.2%}")

acc_true = df[df['label'] == 1].apply(lambda row: row['label'] == row['prediction'], axis=1).mean()
acc_fake = df[df['label'] == 0].apply(lambda row: row['label'] == row['prediction'], axis=1).mean()
print(f"📘 真新闻准确率 Accuracy_true: {acc_true:.2%}")
print(f"📕 假新闻准确率 Accuracy_fake: {acc_fake:.2%}")
