import torch
import torch.nn as nn

class FakeReviewerDetection(nn.Module):
    def __init__(self):
        super(FakeReviewerDetection, self).__init__()
        self.dense_1 = nn.Linear(in_features=448, out_features=128)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.dense_2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, behavior_features, text_features):
        # print(behavior_features.size())
        # print(text_features.size())
        x = torch.cat((behavior_features, text_features), dim=1)
        x = self.dense_1(x)
        x = self.dropout_4(x)
        x = self.dense_2(x)
        x = self.softmax(x)
        return x

