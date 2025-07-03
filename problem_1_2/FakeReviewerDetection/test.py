# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from transformers import LongformerTokenizer
# from YelpDataset import ReviewDataset
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
# import pandas as pd
# from model import Net
#
# pretrained_model = '../hub/swift/longformer-base-4096'
# model_path = '../model/checkpoint_2_last.pt'
# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# # 读取测试集数据
# valid_data = pd.read_csv('../data/yelp_testSelect.csv')
# valid_ratings = valid_data['rating'].tolist()
# valid_labels = valid_data['label'].tolist()
# valid_comment_counts = valid_data['user_comment_count'].tolist()
# valid_time_gaps = valid_data['time_gap'].tolist()
# valid_rating_entropy = valid_data['rating_entropy'].tolist()
# valid_rating_deviations = valid_data['rating_deviation'].tolist()
# valid_review_times = valid_data['review_time'].tolist()
# valid_user_tenures = valid_data['user_tenure'].tolist()
# valid_texts = valid_data['text'].tolist()
# # valid_texts = valid_data['user_cotent'].tolist()
# # valid_content_len = valid_data['user_cotent_length'].tolist()
#
# tokenizer = LongformerTokenizer.from_pretrained(pretrained_model)
# max_length = 5000
# valid_dataset = ReviewDataset(valid_ratings, valid_labels, valid_comment_counts, valid_time_gaps, valid_rating_entropy,valid_rating_deviations,valid_review_times, valid_user_tenures, valid_texts, tokenizer, max_length)
# valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
#
# net = Net()
# net.load_state_dict(torch.load(model_path, map_location=device))
# net = net.to(device)
# print(model_path + "模型加载完成")
#
# net.eval()
# valid_preds = []
# valid_true_labels = []
# with torch.no_grad():
#     for i, valid_input in enumerate(valid_loader):
#         valid_behave_encode = valid_input['behavior_feature'].to(device)
#         valid_text_encode = valid_input['input_ids'].to(device)
#         valid_label = valid_input['labels'].to(device)
#         valid_behave_encode.unsqueeze_(-1)
#         valid_behave_encode = valid_behave_encode.permute(0, 2, 1)
#         valid_output = net(valid_behave_encode, valid_text_encode)
#         valid_pred = valid_output.argmax(dim=1)
#         valid_preds.append(valid_pred.cpu().numpy())
#         valid_true_labels.append(valid_label.cpu().numpy())
#         print(valid_output, valid_pred, valid_label)
#         print(f"predict {i} / {len(valid_loader)}done")
#
# valid_preds = np.concatenate(valid_preds)
# valid_true_labels = np.concatenate(valid_true_labels)
#
# precision = precision_score(valid_true_labels, valid_preds)
# recall = recall_score(valid_true_labels, valid_preds)
# f1 = f1_score(valid_true_labels, valid_preds)
#
# # 计算准确率
# correct_predictions = sum(valid_preds == valid_true_labels)
# accuracy = correct_predictions / len(valid_true_labels)
#
# print(f"Testing model at path: {model_path}")
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1:.4f}')
# print(f"Accuracy: {accuracy:.4f}")
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer
from YelpDataset import ReviewDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
from model import Net


pretrained_model = '../hub/swift/longformer-base-4096'
model_path = '../model/TE_checkpoint_1_last.pt'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# 读取测试集数据
valid_data = pd.read_csv('../data/res_yelpChi_test_downsample.csv')
valid_ratings = valid_data['rating'].tolist()
valid_labels = valid_data['label'].tolist()
valid_comment_counts = valid_data['user_comment_count'].tolist()
valid_time_gaps = valid_data['time_gap'].tolist()
valid_rating_entropy = valid_data['rating_entropy'].tolist()
valid_rating_deviations = valid_data['rating_deviation'].tolist()
valid_review_times = valid_data['review_time'].tolist()
valid_user_tenures = valid_data['user_tenure'].tolist()
valid_texts = valid_data['text'].tolist()
# valid_texts = valid_data['user_cotent'].tolist()
# valid_content_len = valid_data['user_cotent_length'].tolist()


tokenizer = LongformerTokenizer.from_pretrained(pretrained_model)
max_length = 5000
valid_dataset = ReviewDataset(valid_ratings, valid_labels, valid_comment_counts, valid_time_gaps, valid_rating_entropy,
                              valid_rating_deviations, valid_review_times, valid_user_tenures, valid_texts, tokenizer,
                              max_length)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)


net = Net()
net.load_state_dict(torch.load(model_path, map_location=device))
net = net.to(device)
print(model_path + "模型加载完成")


net.eval()
valid_preds = []
valid_true_labels = []
with torch.no_grad():
    for i, valid_input in enumerate(valid_loader):
        valid_behave_encode = valid_input['behavior_feature'].to(device)
        valid_text_encode = valid_input['input_ids'].to(device)
        valid_label = valid_input['labels'].to(device)
        valid_behave_encode.unsqueeze_(-1)
        valid_behave_encode = valid_behave_encode.permute(0, 2, 1)
        valid_output = net(valid_behave_encode, valid_text_encode)
        valid_pred = valid_output.argmax(dim=1)
        valid_preds.append(valid_pred.cpu().numpy())
        valid_true_labels.append(valid_label.cpu().numpy())
        print(valid_output, valid_pred, valid_label)
        print(f"predict {i} / {len(valid_loader)}done")


valid_preds = np.concatenate(valid_preds)
valid_true_labels = np.concatenate(valid_true_labels)

precision = precision_score(valid_true_labels, valid_preds)
recall = recall_score(valid_true_labels, valid_preds)
f1 = f1_score(valid_true_labels, valid_preds)
# 计算准确率
correct_predictions = sum(valid_preds == valid_true_labels)
accuracy = correct_predictions / len(valid_true_labels)

print(f"Testing model at path: {model_path}")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f"Accuracy: {accuracy:.4f}")

