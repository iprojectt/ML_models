import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self, num_classes, cnn_out_features=256, rnn_hidden_size=128):
        super(OCRModel, self).__init__()

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, cnn_out_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # RNN part
        self.rnn = nn.LSTM(cnn_out_features, rnn_hidden_size, batch_first=True)

        # Fully connected layer to output character probabilities
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        # CNN part
        x = self.cnn(x)

        # Reshape for RNN input
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)

        # RNN part
        x, _ = self.rnn(x)

        # Fully connected layer for character probabilities
        x = self.fc(x)
        return x
