# import torch
# import torch.nn as nn
# from transformers import BigBirdModel, BigBirdTokenizer
#
# pretrained_model = '../hub/google/bigbird-roberta-base'
#
# # 定义 GLUSEBlock1D
# class GLUSEBlock1D(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(GLUSEBlock1D, self).__init__()
#         # Squeeze-and-Excitation部分
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#         # Gated Linear Unit部分
#         self.gate_linear = nn.Conv1d(in_channels, in_channels, kernel_size=1)
#         self.gate_sigmoid = nn.Conv1d(in_channels, in_channels, kernel_size=1)
#
#     def forward(self, x):
#         # SE机制
#         se = self.global_avg_pool(x)
#         se = self.fc1(se)
#         se = self.relu(se)
#         se = self.fc2(se)
#         se = self.sigmoid(se)
#
#         # GLU机制
#         gate_linear_out = self.gate_linear(x)
#         gate_sigmoid_out = torch.sigmoid(self.gate_sigmoid(x))
#
#         # GLU风格门控
#         gated_se = gate_linear_out * gate_sigmoid_out
#
#         # 应用SE权重
#         out = x * se
#         out = out + gated_se
#         return out
#
# # 修改后的ContextAwareAttention
# class ContextAwareAttention(nn.Module):
#     def __init__(self):
#         super(ContextAwareAttention, self).__init__()
#         # BigBird
#         self.bigbird = BigBirdModel.from_pretrained(pretrained_model)
#         for param in self.bigbird.parameters():
#             param.requires_grad = False
#
#         inchannel = self.bigbird.config.hidden_size  # 768
#
#         # 替换成GLUSEBlock1D
#         self.gluse_block1 = GLUSEBlock1D(in_channels=inchannel)
#         self.gluse_block2 = GLUSEBlock1D(in_channels=inchannel)
#         self.gluse_block3 = GLUSEBlock1D(in_channels=inchannel)
#
#         # Dropout
#         self.dropout = nn.Dropout(p=0.5)
#
#         # 后续LSTM和注意力
#         self.lstm = nn.LSTM(input_size=inchannel * 3, hidden_size=64, bidirectional=True)
#         self.attention_linear_1 = nn.Linear(in_features=128, out_features=64, bias=True)
#         self.attention_linear_2 = nn.Linear(in_features=64, out_features=1, bias=True)
#
#     def forward(self, x):
#         x = self.bigbird(x)[0]  # (batch_size, seq_len, hidden_size)
#         # print("BigBird输出：", x.shape)
#
#         x = x.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)
#
#         # 分支
#         x1 = self.gluse_block1(x)
#         x1 = self.dropout(x1)
#         # print("GLUSEBlock1输出:", x1.shape)
#
#         x2 = self.gluse_block2(x)
#         x2 = self.dropout(x2)
#         # print("GLUSEBlock2输出:", x2.shape)
#
#         x3 = self.gluse_block3(x)
#         x3 = self.dropout(x3)
#         # print("GLUSEBlock3输出:", x3.shape)
#
#         # 拼接
#         concatenated_result = torch.cat((x1, x2, x3), dim=1)  # 通道维度拼接
#         # print("拼接结果:", concatenated_result.shape)
#
#         concatenated_result = concatenated_result.permute(2, 0, 1)  # (seq_len, batch_size, feature_size)
#
#         concatenated_result, _ = self.lstm(concatenated_result)
#         concatenated_result = concatenated_result.permute(1, 0, 2)  # (batch_size, seq_len, 2*hidden_size)
#         # print("LSTM输出结果:", concatenated_result.shape)
#
#         # 注意力
#         attention_scores = self.attention_linear_2(torch.tanh(self.attention_linear_1(concatenated_result)))
#         attention_weights = torch.softmax(attention_scores, dim=1)
#         concatenated_result = torch.sum(concatenated_result * attention_weights, dim=1)
#         # print("注意力机制输出：", concatenated_result.shape)
#
#         return concatenated_result


import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BigBirdModel, BigBirdTokenizer
pretrained_model = 'hub/google/bigbird-roberta-base'
# 定义 GLUSEBlock1D
class GLUSEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GLUSEBlock1D, self).__init__()
        # Squeeze-and-Excitation部分
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Gated Linear Unit部分
        self.gate_linear = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gate_sigmoid = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # SE机制
        se = self.global_avg_pool(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)

        # GLU机制
        gate_linear_out = self.gate_linear(x)
        gate_sigmoid_out = torch.sigmoid(self.gate_sigmoid(x))

        # GLU风格门控
        gated_se = gate_linear_out * gate_sigmoid_out

        # 应用SE权重
        out = x * se
        out = out + gated_se
        return out
class ContextAwareAttention(nn.Module):
    def __init__(self):
        super(ContextAwareAttention, self).__init__()
        # 初始化BigBird模型实例，加载参数以及预训练权重
        self.bigbird = BigBirdModel.from_pretrained(pretrained_model)
        for param in self.bigbird.parameters():
            # 训练过程中不更新BigBird参数
            param.requires_grad = False

        inchannel = self.bigbird.config.hidden_size  # hidden_size通常是768

        self.conv1d_1 = nn.Conv1d(in_channels=inchannel, out_channels=64, kernel_size=3, stride=1)
        self.max_pooling_1 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.dropout_1 = nn.Dropout(p=0.5)

        self.conv1d_2 = nn.Conv1d(in_channels=inchannel, out_channels=64, kernel_size=4, stride=1)
        self.max_pooling_2 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.dropout_2 = nn.Dropout(p=0.5)

        self.conv1d_3 = nn.Conv1d(in_channels=inchannel, out_channels=64, kernel_size=5, stride=1)
        self.max_pooling_3 = nn.MaxPool1d(kernel_size=4, stride=1)
        self.dropout_3 = nn.Dropout(p=0.5)

        self.bn1_1 = nn.BatchNorm1d(num_features=64)
        self.bn1_2 = nn.BatchNorm1d(num_features=64)
        self.bn1_3 = nn.BatchNorm1d(num_features=64)

        self.glu_se_resblock = GLUSEBlock1D(in_channels=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True)
        self.attention_linear_1 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.attention_linear_2 = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        x = self.bigbird(x)[0]  # 获取最后一个隐状态输出

        # print("BigBird输出：", x.shape)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)

        x1 = self.conv1d_1(x)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)
        x1 = self.glu_se_resblock(x1)
        x1 = self.max_pooling_1(x1)
        x1 = self.dropout_1(x1)
        # print("conv1d_1输出:", x1.shape)

        x2 = self.conv1d_2(x)
        x2 = self.bn1_2(x2)
        x2 = F.relu(x2)
        x2 = self.glu_se_resblock(x2)
        x2 = self.max_pooling_2(x2)
        x2 = self.dropout_2(x2)
        # print("conv1d_2输出:", x2.shape)

        x3 = self.conv1d_3(x)
        x3 = self.bn1_3(x3)
        x3 = F.relu(x3)
        x3 = self.glu_se_resblock(x3)
        x3 = self.max_pooling_3(x3)
        x3 = self.dropout_3(x3)
        # print("conv1d_3输出:", x3.shape)

        # 在通道维度拼接
        concatenated_result = torch.cat((x1, x2, x3), dim=2)
        # print("卷积拼接结果:", concatenated_result.shape)

        concatenated_result = self.glu_se_resblock(concatenated_result)
        # print("gluse输出:", concatenated_result.shape)

        concatenated_result = concatenated_result.permute(2, 0, 1)  # (seq_len, batch_size, feature_size)

        concatenated_result, _ = self.lstm(concatenated_result)
        concatenated_result = concatenated_result.permute(1, 0, 2)  # (batch_size, seq_len, 2*hidden_size)
        # print("lstm输出结果：", concatenated_result.shape)

        # 注意力机制
        attention_scores = self.attention_linear_2(torch.tanh(self.attention_linear_1(concatenated_result)))
        attention_weights = torch.softmax(attention_scores, dim=1)
        concatenated_result = torch.sum(concatenated_result * attention_weights, dim=1)
        # print("注意力机制输出：", concatenated_result.shape)
        return concatenated_result

# # 测试代码
# text = r"Although the streets of South Philly do not have the glamour typically associated with Rome, the ethnic neighborhood definitely boasts some of the best Italian food in America. ..."
# tokenizer = BigBirdTokenizer.from_pretrained(pretrained_model)
#
# tokens = tokenizer(text, add_special_tokens=True, return_tensors='pt')
# input_ids = tokens['input_ids']
# print("输入张量:", input_ids.shape)
# print(input_ids)
#
# contextAMM = ContextAwareAttention()
# x = contextAMM.forward(input_ids)
# print(x.size())
# print(tokenizer.decode(input_ids.tolist()[0]))
