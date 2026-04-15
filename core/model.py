"""
model.py — ResNet-CRNN + Attention architecture for handwriting recognition.

The network has four stages:
  1. ResNet-18 backbone — pretrained CNN that extracts visual features from a
                          grayscale word image, with modified strides to preserve
                          temporal (width) resolution for CTC decoding.
  2. BiLSTM layers      — reads the feature sequence bidirectionally to capture
                          context in both directions (3 layers, 512 total hidden).
  3. Attention layer    — learns to focus on the most relevant timesteps,
                          improving character-level predictions.
  4. Linear head        — projects each timestep to a distribution over the
                          character vocabulary (+ CTC blank).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Attention(nn.Module):
    """Single-head additive attention over LSTM output timesteps.

    For each timestep, computes an attention-weighted context vector
    over all timesteps, then concatenates it with the original hidden
    state. This helps the model focus on relevant spatial positions.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_out):
        # rnn_out: (B, T, H)
        energy = torch.tanh(self.attn(rnn_out))     # (B, T, H)
        scores = self.v(energy).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=-1)          # (B, T)

        # Context: weighted sum over all timesteps
        context = torch.bmm(weights.unsqueeze(1), rnn_out)  # (B, 1, H)
        context = context.expand_as(rnn_out)                 # (B, T, H)

        # Concatenate context with original, project back to H
        return rnn_out + context  # residual connection


class ResNetCRNN(nn.Module):
    """ResNet-18 backbone + BiLSTM + Attention for handwriting recognition.

    Replaces the original hand-crafted CNN with a pretrained ResNet-18,
    modified to preserve width resolution for CTC sequence decoding.

    Feature map progression (input: 1×32×128):
      conv1  (stride 1)  : 64×32×128
      layer1             : 64×32×128
      layer2 (stride 2)  : 128×16×64
      layer3 (stride 2,1): 256×8×64   ← height reduced, width preserved
      layer4 (stride 2,1): 512×4×64   ← height reduced, width preserved
      adaptive_pool      : 512×1×64   → sequence length 64
    """

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Stage 1 — Adapted ResNet-18 backbone
        # Replace 7×7 stride-2 conv with 3×3 stride-1 for small 32px-height input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)
        # Skip resnet.maxpool — 32px height is too small for aggressive downsampling

        self.layer1 = resnet.layer1   # 64→64,   no spatial change
        self.layer2 = resnet.layer2   # 64→128,  stride 2 in both dims

        # Modify layer3 & layer4: stride (2,1) to reduce height only, preserve width
        self.layer3 = resnet.layer3
        self._modify_stride(self.layer3, (2, 1))

        self.layer4 = resnet.layer4
        self._modify_stride(self.layer4, (2, 1))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse height to 1
        self.cnn_dropout = nn.Dropout2d(dropout)

        # Stage 2 — BiLSTM (3 layers, 256 per direction = 512 total)
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True,
                           dropout=dropout, num_layers=3)
        self.rnn_dropout = nn.Dropout(dropout)

        # Stage 3 — Attention over LSTM timesteps
        self.attention = Attention(512)

        # Stage 4 — CTC classifier for each timestep
        self.fc = nn.Linear(512, num_classes)

        # Initialize the new conv1 (can't transfer 7×7 RGB weights to 3×3 grayscale)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _modify_stride(layer, new_stride):
        """Replace stride in the first block's downsampling convolutions
        to preserve width while still reducing height."""
        block = layer[0]
        old_conv = block.conv1
        new_conv = nn.Conv2d(old_conv.in_channels, old_conv.out_channels,
                             old_conv.kernel_size, stride=new_stride,
                             padding=old_conv.padding, bias=False)
        new_conv.weight.data.copy_(old_conv.weight.data)
        block.conv1 = new_conv

        if block.downsample is not None:
            old_ds = block.downsample[0]
            new_ds = nn.Conv2d(old_ds.in_channels, old_ds.out_channels,
                               old_ds.kernel_size, stride=new_stride, bias=False)
            new_ds.weight.data.copy_(old_ds.weight.data)
            block.downsample[0] = new_ds

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 64, 32, 128)
        x = self.layer1(x)                        # (B, 64, 32, 128)
        x = self.layer2(x)                        # (B, 128, 16, 64)
        x = self.layer3(x)                        # (B, 256, 8, 64)
        x = self.layer4(x)                        # (B, 512, 4, 64)
        x = self.adaptive_pool(x)                 # (B, 512, 1, 64)
        x = self.cnn_dropout(x)

        x = x.squeeze(2)                          # (B, 512, 64)
        x = x.permute(0, 2, 1)                   # (B, 64, 512)

        x, _ = self.rnn(x)                       # (B, 64, 512)
        x = self.rnn_dropout(x)
        x = self.attention(x)                     # (B, 64, 512)
        x = self.fc(x)                            # (B, 64, num_classes)
        x = x.permute(1, 0, 2)                  # (64, B, num_classes) — CTC time-first
        return x