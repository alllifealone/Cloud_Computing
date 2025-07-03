import torch.nn as nn
from BFExtractor import BehaviorSensitiveFeatureExtractor
from BRCFE import ContextAwareAttention
# from CNN_Transformer import Model
from FakeReviewDetector import FakeReviewerDetection

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.behavior_extractor = BehaviorSensitiveFeatureExtractor()
        # self.text_extractor =  Model()
        self.context_aware_attention = ContextAwareAttention()
        self.fake_reviewer_detection = FakeReviewerDetection()

    def forward(self, behavior_data, text_data):
        behavior_features = self.behavior_extractor(behavior_data)
        behavior_features = behavior_features.view(behavior_features.size(0), -1)
        text_features = self.context_aware_attention(text_data)
        # text_features = self.text_extractor(text_data)
        return self.fake_reviewer_detection(behavior_features, text_features)


def count_parameters_in_mb(model):
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    total_bytes = total_params * 4  # 假设每个参数是 float32 类型，每个占 4 字节
    total_mb = total_bytes / (1024 * 1024)  # 转换为 MB
    return total_mb

# # 初始化模型
# model = Net()
#
# # 统计参数量并转换为 MB
# params_in_mb = count_parameters_in_mb(model)
# print(f'Total parameters in MB: {params_in_mb:.2f} MB')