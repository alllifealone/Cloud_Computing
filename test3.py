from nltk.stem import WordNetLemmatizer
import nltk
# 强制使用自定义路径
nltk.data.path.clear()
nltk.data.path.append("/home/mq/kevin/nltk_data")

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos='v'))
