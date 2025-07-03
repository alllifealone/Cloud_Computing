from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd

data = pd.read_csv("../data/yelpdata7.csv")
# 提取特征
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['user_cotent'])
# 准备标签
y = data['label']

# 创建并训练模型
model = MultinomialNB()
model.fit(X, y)

# 进行预测和评估
y_pred = model.predict(X)
print(len(y_pred))
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y_pred, y)
recall = recall_score(y_pred, y)
f1 = f1_score(y_pred, y)
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f"Accuracy: {accuracy:.4f}")