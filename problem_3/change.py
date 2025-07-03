import pandas as pd

# 读取原始的txt文件，假设是 tab 分隔
df = pd.read_csv("posts_groundtruth.txt", sep="\t", encoding="utf-8")

# 写入为CSV文件
df.to_csv("twitter_data.csv", index=False, encoding="utf-8")
