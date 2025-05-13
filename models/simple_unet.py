import torch.nn as nn


class SimpleUnet(nn.Module):
    """Simple Unet model"""

    def __init__(self):
        super(SimpleUnet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """Forward method"""

        # x ( batch_size, c=1, h, w)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
