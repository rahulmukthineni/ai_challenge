import torch
import torch.nn as nn

from cnn_multimodal import BinaryClassifier

import torch.nn as nn

class MultimodalClassifier(nn.Module):
    def __init__(self, num_eeg_channels, num_ecg_channels, num_clinical_features):
        super(MultimodalClassifier, self).__init__()
        self.eeg_model = BinaryClassifier(num_channels=num_eeg_channels)
        self.ecg_model = BinaryClassifier(num_channels=num_ecg_channels)
        self.num_clinical_features = num_clinical_features
        self.fc_clinical = nn.Linear(num_clinical_features, 32)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc_final1 = nn.Linear(32 + 32 + 32, 128)
        self.bn_final1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc_final2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, eeg_input, ecg_input, clinical_input):
        eeg_output = self.eeg_model(eeg_input)
        ecg_output = self.ecg_model(ecg_input)
        clinical_output = self.fc_clinical(clinical_input)
        clinical_output = self.dropout(clinical_output)
        clinical_output = self.relu(clinical_output)
        
        concatenated_input = torch.cat((eeg_output, ecg_output, clinical_output), dim=1)
        

        final_output = self.fc_final1(concatenated_input)
        final_output = self.bn_final1(final_output)
        final_output = self.relu(final_output)
        final_output = self.dropout(final_output)
        final_output = self.fc_final2(final_output)
        final_output = self.sigmoid(final_output)
        return final_output
  