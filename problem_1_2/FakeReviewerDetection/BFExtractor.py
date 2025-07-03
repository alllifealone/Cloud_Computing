import torch
import torch.nn as nn

class BehaviorSensitiveFeatureExtractor(nn.Module):
    def __init__(self):
        super(BehaviorSensitiveFeatureExtractor, self).__init__()
        self.conv1d = nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        feature_maps = self.conv1d(x)
        feature_maps = self.bn1(feature_maps)
        feature_maps = self.relu(feature_maps)
        feature_maps = feature_maps.reshape(feature_maps.size(0), -1)
        return feature_maps


