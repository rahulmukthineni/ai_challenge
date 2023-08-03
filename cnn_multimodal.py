import torch
import torch.nn as nn

import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, num_channels):
        super(BinaryClassifier, self).__init__()
        self.num_channels = num_channels
        
        self.conv1 = nn.Conv1d(num_channels, 24, kernel_size=10, stride=2)
        self.bn1 = nn.BatchNorm1d(24)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
  
        self.dropout1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(kernel_size=10, stride=2)
        
        self.conv2 = nn.Conv1d(24, 32, kernel_size=10, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=2)
        
        self.conv3 = nn.Conv1d(32, 48, kernel_size=10, stride=2)
        self.bn3 = nn.BatchNorm1d(48)
        self.dropout3 = nn.Dropout(0.2)
        self.pool3 = nn.MaxPool1d(kernel_size=10, stride=2)
        
        self.conv4 = nn.Conv1d(48, 64, kernel_size=10, stride=2)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        self.pool4 = nn.MaxPool1d(kernel_size=10, stride=2)

        self.lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)

        self.fc = nn.Linear(64, 32)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.pool4(x)
        
        x = x.permute(0, 2, 1)  
        
        x, _ = self.lstm1(x)
     
        x, _ = self.lstm2(x)

        
        x = x[:, -1, :] 
        
        x = self.fc(x)
        
        return x