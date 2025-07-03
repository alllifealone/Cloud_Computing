import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import numpy as np

# === 加载模型 ===
model_path = "../../.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()

# === 读取并采样数据（每类各500条）===
true_df = pd.read_csv("./fake-and-real/True.csv").dropna(subset=["text"]).sample(n=500, random_state=42)
fake_df = pd.read_csv("./fake-and-real/Fake.csv").dropna(subset=["text"]).sample(n=500, random_state=42)
true_df['label'] = 1
fake_df['label'] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df[['text', 'label']].dropna()

# === 预测函数 ===
def classify_news(text):
    prompt = (f"你是一个专注通过分析新闻情感进行识别虚假新闻的专家。请阅读以下新闻内容，并严格按照以下格式回答：\n"
        f"只能回答“真”或“假”，不能包含其他任何文字或解释。\n\n"
        f"新闻内容：{text.strip()}\n\n"
        f"请回答："
        )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1  # 允许生成最多2个token
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text.replace(prompt, "").strip()
    
    # print(answer)  # 可选打印模型输出

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
    predictions.append(pred)

df['prediction'] = predictions

# === 过滤掉无法判断的数据 ===
df = df[df['prediction'] != -1]

# === 指标计算 ===
y_true = df['label'].values
y_pred = df['prediction'].values
accuracy = (y_true == y_pred).mean()
f1 = f1_score(y_true, y_pred)
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
