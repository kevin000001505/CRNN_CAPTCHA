import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ResNet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
ResNet_model = nn.Sequential(*list(ResNet_model.children())[:-2])  # Keep up to layer4


class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=11):
        super(CRNN, self).__init__()
        self.resnet = ResNet_model
        self.lstm1 = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            hidden_size * 2, hidden_size, bidirectional=True, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.resnet(x)  # [B, C=512, H=1, W]
        x = x.squeeze(2)  # Remove height dim → [B, 512, W]
        x = x.permute(0, 2, 1)  # [B, W, 512] → (batch, seq_len, features)
        x, _ = self.lstm1(x)  # Extract only the output, ignore hidden states
        x, _ = self.lstm2(x)  # Extract only the output, ignore hidden states
        x = self.classifier(x)  # Apply the fully connected layer
        return x


class CRNNModel:
    def __init__(self, input_size=2048, hidden_size=256, num_classes=11, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN(input_size, hidden_size, num_classes).to(self.device)
        self.num_classes = num_classes
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def decode_predictions(self, logits):
        """Decode CTC predictions using greedy decoding"""
        # Get the most likely class at each timestep
        _, preds = torch.max(logits, dim=2)  # [B, T] - batch_size, time_steps

        decoded_sequences = []
        for pred in preds:
            # Remove only blanks, keep all other predictions
            non_blank_chars = []
            for char in pred:
                char = char.item()
                # non_blank_chars.append(char)
                if char != 0:  # Keep all non-blank characters
                    actual_digit = char - 1
                    non_blank_chars.append(actual_digit)
                    # non_blank_chars.append(char)

            # Take first 4 for CAPTCHA
            final_sequence = (
                non_blank_chars[:4] if len(non_blank_chars) >= 4 else non_blank_chars
            )
            decoded_sequences.append(final_sequence)

        return decoded_sequences

    def predictions_to_strings(self, predictions):
        """Convert decoded predictions to readable strings"""
        strings = []
        for pred in predictions:
            # Convert digits to string
            string_pred = "".join(map(str, pred))
            strings.append(string_pred)
        return strings
