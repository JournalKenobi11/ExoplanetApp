import torch
import torch.nn as nn


class ConvBranch(nn.Module):

    def __init__(self):

        super().__init__()
        self.branch = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.10),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.10),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):

        return self.branch(x)


class TessPrecisionRecallNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.raw_branch = ConvBranch()
        self.fft_branch = ConvBranch()
        self.fold_branch = ConvBranch()

        self.stats_branch = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, raw_x, fft_x, fold_x, stats_x):

        features = torch.cat([
            self.raw_branch(raw_x),
            self.fft_branch(fft_x),
            self.fold_branch(fold_x),
            self.stats_branch(stats_x),
        ], dim=1)

        return self.classifier(features).squeeze(1)