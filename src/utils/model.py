import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class EMNISTDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features.astype(np.float32).reshape(-1, 1, 28, 28)
        self.y = labels.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        # saída do conv+pool: 32 x 13 x 13 = 5408
        self.fc1   = nn.Linear(32 * 13 * 13, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# def eval_metrics(sets, metric):

#     outputs = []
    
#     with torch.no_grad():
#         for batch in dataloader:
            