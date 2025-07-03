import nltk

# 明确使用自定义路径，避免再次加载系统路径
nltk_data_path = "/home/mq/kevin/nltk_data"
nltk.data.path.clear()  # 先清空默认路径
nltk.data.path.append(nltk_data_path)

# 下载资源到该路径
for resource in ['stopwords']:
    nltk.download(resource, download_dir=nltk_data_path)
