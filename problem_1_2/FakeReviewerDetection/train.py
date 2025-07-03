import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import LongformerTokenizer
from YelpDataset import ReviewDataset
import matplotlib.pyplot as plt
from model import Net
import pandas as pd
import os
from fvcore.nn import FlopCountAnalysis

pretrained_model = '../hub/swift/longformer-base-4096'
save_path = '../model'

plt.switch_backend('agg')


def draw(x_axis, y_axis):
    plt.figure('loss image')
    plt.xlabel('iter')
    plt.ylabel('loss')
    # 提取损失张量中的值
    plt.plot(x_axis, y_axis)
    plt.legend(['loss'])
    plt.savefig('loss.jpg')

def save_checkpoint(state_dict, epoch, iter):
    checkpoint_path = os.path.join(save_path, f"checkpoint_{epoch}_{iter}.pt")
    # 创建保存路径（如果不存在）
    os.makedirs(save_path, exist_ok=True)
    torch.save(state_dict, checkpoint_path)

def train_model():
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    net = Net()
    learning_rate = 1e-5
    max_length = 5000
    batch_size = 4
    num_epochs = 5
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)

    # 新增代码：用于记录最小损失值以及对应的轮次和迭代次数
    min_loss = float('inf')


    # checkpoint = torch.load("../model/checkpoint_1_3000.pt")
    # net.load_state_dict(checkpoint)
    # print('参数恢复完成')

    # 读取训练集数据
    train_data = pd.read_csv('../data/yelp_trainSelect_downsample.csv')
    # 提取文本数据和标签数据
    train_ratings = train_data['rating'].tolist()
    train_labels = train_data['label'].tolist()
    train_comment_counts = train_data['user_comment_count'].tolist()
    train_time_gaps = train_data['time_gap'].tolist()
    train_rating_entropy = train_data['rating_entropy'].tolist()
    train_rating_deviations = train_data['rating_deviation'].tolist()
    train_review_times = train_data['review_time'].tolist()
    train_user_tenures = train_data['user_tenure'].tolist()
    train_texts = train_data['text'].tolist()
    # train_contentlen = train_data['user_cotent_length'].tolist()

    tokenizer = LongformerTokenizer.from_pretrained(pretrained_model)
    train_dataset = ReviewDataset(train_ratings,train_labels,train_comment_counts, train_time_gaps, train_rating_entropy, train_rating_deviations, train_review_times, train_user_tenures, train_texts, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_group = len(train_loader)

    net.train()
    x_axis = []
    y_axis = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for iter, input in enumerate(train_loader, 0):
            optimizer.zero_grad()
            behave_encode = input['behavior_feature'].to(device)
            text_encode = input['input_ids'].to(device)
            # 修正标签键的名称
            label = input['labels'].to(device)
            behave_encode.unsqueeze_(-1)
            behave_encode = behave_encode.permute(0, 2, 1)
            # print(text_encode.size())
            # print(behave_encode.size())
            output = net(behave_encode, text_encode)
            # print(output)
            # print(output.size())
            # flops = FlopCountAnalysis(net, (behave_encode, text_encode))
            # print(f'FLOPS: {flops.total() / 1e9} GFLOPS')
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{iter}/{len(train_loader)}], loss={loss.item():.4f}')
                x_axis.append(iter + data_group * epoch)
                y_axis.append(loss.item())
                draw(x_axis, y_axis)

            # 新增代码：比较当前损失值与最小损失值
            if loss.item() <= min_loss:
                min_loss = loss.item()
                # 保存当前损失最小的模型参数
                best_checkpoint_path = os.path.join(save_path,
                                                    f"best_checkpoint.pt")
                torch.save(net.state_dict(), best_checkpoint_path)
                print(f'Best checkpoint saved')


            if iter % 1000 == 0 :
                save_checkpoint(net.state_dict(), epoch + 1, iter)
        save_checkpoint(net.state_dict(), epoch + 1, 'last')




if __name__ == '__main__':
    train_model()