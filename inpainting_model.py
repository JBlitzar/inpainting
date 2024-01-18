import torch.nn as nn
import torch
def black_out_random_rectangle(tensor):
    batch_size, num_channels, height, width = list(tensor.shape)

    for i in range(batch_size):
        # Randomly select the position and size of the rectangle
        top = torch.randint(0, height-2, (1,)).item()
        left = torch.randint(0, width-2, (1,)).item()
        rect_height = torch.randint(1, int(height/4), (1,)).item()
        rect_width = torch.randint(1, int(width/4), (1,)).item()
        # Black out the selected rectangle in all channels for the current image
        tensor[i, :, top:max(top+rect_height, height), left:max(left+rect_width,width)] = 0

class Autoencoder_CAE(nn.Module):
    def __init__(self):
        super(Autoencoder_CAE, self).__init__()

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Define forward pass
        x = self.encoder(x)
        x = self.decoder(x)
        x = x * 255
        return x