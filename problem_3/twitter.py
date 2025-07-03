import pandas as pd
from langdetect import detect
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.data.path.clear()
nltk.data.path.append("/home/mq/kevin/nltk_data")
# 初始化
tqdm.pandas()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 1. 读取CSV文件
file_path = "twitter_data.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 2. 定义语言检测函数
def is_english(text):
    try:
        return detect(str(text)) == 'en'
    except:
        return False

# 3. 筛选为英语的行
df["is_english"] = df["post_text"].progress_apply(is_english)
df_en = df[df["is_english"]].copy()

# 4. 计算字符与词数统计
df_en["char_length"] = df_en["post_text"].apply(lambda x: len(str(x)))
df_en["word_count"] = df_en["post_text"].apply(lambda x: len(str(x).split()))

# 输出统计信息
dataset_size = len(df_en)
avg_length = df_en["char_length"].mean()
avg_words = df_en["word_count"].mean()
max_words = df_en["word_count"].max()
min_words = df_en["word_count"].min()

print("===== 英文推文数据统计 =====")
print(f"数据大小（行数）：{dataset_size}")
print(f"平均文本长度（字符数）：{avg_length:.2f}")
print(f"平均词数：{avg_words:.2f}")
print(f"最大词数：{max_words}")
print(f"最小词数：{min_words}")

# 5. 添加三个文本处理函数
def tokenize_and_clean(text):
    # 保留空格分隔的英文单词，转小写
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    tokens = text.lower().split()
    return tokens

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]

# 6. 应用文本处理流程
df_en["tokens"] = df_en["post_text"].progress_apply(tokenize_and_clean)
df_en["clean_tokens"] = df_en["tokens"].apply(remove_stopwords)
df_en["lemmatized_tokens"] = df_en["clean_tokens"].apply(lemmatize_tokens)

# 示例：输出前5条处理结果
print("\n===== 示例词元处理结果 =====")
print(df_en[["post_text", "lemmatized_tokens"]].head())

from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# 7. 构建词典与语料库
dictionary = corpora.Dictionary(df_en["lemmatized_tokens"])

corpus = [dictionary.doc2bow(text) for text in df_en["lemmatized_tokens"]]

# 8. 训练 LDA 模型（假设主题数为10）
lda_model = models.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=10,
                            random_state=42,
                            passes=10,
                            alpha='auto',
                            per_word_topics=True)

# 9. pyLDAvis 可视化
print("\n>>> 生成交互式主题图（pyLDAvis）...")
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_vis, "lda_vis.html")
print("已保存为 lda_vis.html（请用浏览器打开）")

# 10. 每个主题的词云图
print("\n>>> 绘制主题词云...")
for t in range(lda_model.num_topics):
    plt.figure()
    word_freqs = dict(lda_model.show_topic(t, topn=30))
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic #{t}")
    plt.tight_layout()
    plt.savefig(f"topic_{t}_wordcloud.png")  # 可保存为图片
    plt.close()
print("词云图已保存（每个主题一个 PNG 文件）")

# 11. （可选）绘制热力图（文档-主题概率分布）
print("\n>>> 绘制主题分布热力图（前50个文档）...")
topic_matrix = []
for bow in corpus[:50]:
    topic_dist = lda_model.get_document_topics(bow)
    row = [0] * lda_model.num_topics
    for topic_id, prob in topic_dist:
        row[topic_id] = prob
    topic_matrix.append(row)

plt.figure(figsize=(12, 6))
sns.heatmap(topic_matrix, cmap="YlGnBu", cbar=True)
plt.xlabel("Topic")
plt.ylabel("Document")
plt.title("Document-Topic Heatmap (Top 50)")
plt.tight_layout()
plt.savefig("topic_heatmap.png")
plt.close()
print("热力图已保存为 topic_heatmap.png")

# 12. （重点）主题内容解释：打印每个主题的关键词 + 解释提示词（建议用 ChatMindAi 分析）
print("\n===== 每个主题的关键词 + 可输入大模型解释的提示词 =====")
for i in range(lda_model.num_topics):
    top_words = [word for word, _ in lda_model.show_topic(i, topn=10)]
    prompt = f"请解释以下词构成的主题含义：{', '.join(top_words)}"
    print(f"\n🧠 Topic #{i}: {' '.join(top_words)}\n📝 ChatMindAi提示：{prompt}")
import csv

# 13. 输出每个主题关键词到CSV
output_csv = "lda_topic_keywords.csv"
topic_keywords = []

for i in range(lda_model.num_topics):
    top_words = [word for word, _ in lda_model.show_topic(i, topn=10)]
    prompt = f"请解释以下词构成的主题含义：{', '.join(top_words)}"
    topic_keywords.append({
        "topic_id": i,
        "keywords": ", ".join(top_words),
        "prompt": prompt
    })

# 写入CSV文件
with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["topic_id", "keywords", "prompt"])
    writer.writeheader()
    writer.writerows(topic_keywords)

print(f"\n>>> 已将每个主题的关键词及提示词保存至 {output_csv}")
