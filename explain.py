from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import os

# 设置模型路径
model_path = "../../.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()

# 构造文本生成pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)  # 默认使用 CPU


# 读取主题提示词CSV
df_prompt = pd.read_csv("lda_topic_keywords.csv")

# 创建输出目录
os.makedirs("topic_explanations", exist_ok=True)

# 循环处理每个主题
for index, row in df_prompt.iterrows():
    topic_id = row["topic_id"]
    prompt = row["prompt"]

    # 调用模型生成解释
    response = generator(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]

    # 截取从 prompt 后开始的部分作为纯解释
    explanation = response.replace(prompt, "").strip()

    # 将解释绘制成图像（每张图一个主题）
    plt.figure(figsize=(10, 6))
    wrapped_text = "\n".join(textwrap.wrap(explanation, width=80))
    plt.text(0.01, 0.99, f"Topic #{topic_id}\n\n{wrapped_text}", fontsize=12, va='top')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"topic_explanations/topic_{topic_id}_explanation.png")
    plt.close()

print("✅ 所有主题的解释已生成并保存为图片 (topic_explanations/*.png)")
