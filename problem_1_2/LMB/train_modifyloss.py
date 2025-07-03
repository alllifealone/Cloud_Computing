import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import BigBirdTokenizer
from YelpDataset import ReviewDataset
import matplotlib.pyplot as plt
from model import Net
import pandas as pd
import os
from sklearn.utils.class_weight import compute_class_weight

# 路径配置
pretrained_model = 'hub/google/bigbird-roberta-base'
save_path = 'model'
plt.switch_backend('agg')

def draw(x_axis, y_axis):
    plt.figure('loss image')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.plot(x_axis, y_axis)
    plt.legend(['loss'])
    plt.savefig('loss.jpg')

def save_checkpoint(state_dict, epoch, iter):
    checkpoint_path = os.path.join(save_path, f"checkpoint_{epoch}_{iter}.pt")
    os.makedirs(save_path, exist_ok=True)
    torch.save(state_dict, checkpoint_path)

def train_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Net()
    net = net.to(device)
    learning_rate = 1e-5
    max_length = 4096
    batch_size = 4
    num_epochs = 5
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 读取训练集数据
    train_data = pd.read_csv('data/yelpChi_train.csv')

    # 提取各特征
    train_ratings = train_data['rating'].tolist()
    train_labels = train_data['label'].tolist()
    train_comment_counts = train_data['user_comment_count'].tolist()
    train_time_gaps = train_data['time_gap'].tolist()
    train_rating_entropy = train_data['rating_entropy'].tolist()
    train_rating_deviations = train_data['rating_deviation'].tolist()
    train_review_times = train_data['review_time'].tolist()
    train_user_tenures = train_data['user_tenure'].tolist()
    train_texts = train_data['text'].tolist()

    # 加载 Tokenizer
    tokenizer = BigBirdTokenizer.from_pretrained(pretrained_model)

    # 创建数据集和 DataLoader
    train_dataset = ReviewDataset(
        train_ratings, train_labels, train_comment_counts,
        train_time_gaps, train_rating_entropy, train_rating_deviations,
        train_review_times, train_user_tenures, train_texts, tokenizer, max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_group = len(train_loader)

    # 类别权重 + 标签平滑
    class_weights = compute_class_weight(class_weight='balanced', classes=list(set(train_labels)), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 记录最小损失
    min_loss = float('inf')

    net.train()
    x_axis = []
    y_axis = []

    for epoch in range(num_epochs):
        for iter, input_batch in enumerate(train_loader, 0):
            optimizer.zero_grad()

            behave_encode = input_batch['behavior_feature'].to(device)
            text_encode = input_batch['input_ids'].to(device)
            label = input_batch['labels'].to(device)

            behave_encode = behave_encode.unsqueeze(-1).permute(0, 2, 1)

            output = net(behave_encode, text_encode)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # 打印与绘制 loss
            if iter % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{iter}/{len(train_loader)}], loss={loss.item():.4f}')
                x_axis.append(iter + data_group * epoch)
                y_axis.append(loss.item())
                draw(x_axis, y_axis)

            # 保存最优模型
            if loss.item() <= min_loss:
                min_loss = loss.item()
                best_checkpoint_path = os.path.join(save_path, f"best_checkpoint.pt")
                torch.save(net.state_dict(), best_checkpoint_path)
                print(f'Best checkpoint saved at epoch {epoch+1}, iter {iter}')

            # 每1000步保存一次
            if iter % 1000 == 0:
                save_checkpoint(net.state_dict(), epoch + 1, iter)

        # 每个 epoch 最后保存一次
        save_checkpoint(net.state_dict(), epoch + 1, 'last')

if __name__ == '__main__':
    train_model()
