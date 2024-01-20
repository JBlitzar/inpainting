import torch.nn as nn
import torch
def black_out_random_rectangle(tensor):
    batch_size, num_channels, height, width = list(tensor.shape)

    for i in range(batch_size):
        # Randomly select the position and size of the rectangle
        top = torch.randint(0, height-2, (1,)).item()
        left = torch.randint(0, width-2, (1,)).item()
        rect_height = torch.randint(1, int(height/2), (1,)).item()
        rect_width = torch.randint(1, int(width/2), (1,)).item()
        # Black out the selected rectangle in all channels for the current image
        tensor[i, :, top:min(top+rect_height, height), left:min(left+rect_width,width)] = 0

def black_out_random_rectangle_legacy(tensor):
    batch_size, num_channels, height, width = list(tensor.shape)

    for i in range(batch_size):
        # Randomly select the position and size of the rectangle
        top = torch.randint(0, height-2, (1,)).item()
        left = torch.randint(0, width-2, (1,)).item()
        rect_height = torch.randint(1, height-top, (1,)).item()
        rect_width = torch.randint(1, width-left, (1,)).item()
        # Black out the selected rectangle in all channels for the current image
        tensor[i, :, top:min(top+rect_height, height), left:min(left+rect_width,width)] = 0

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
class Autoencoder_CAEv2(nn.Module):
    def __init__(self):
        super(Autoencoder_CAEv2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # bilinear, nearest
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Define forward pass
        x = self.encoder(x)
        x = self.decoder(x)
        x = x * 255
        return x
class Autoencoder_CAEv3(nn.Module):
    def __init__(self):
        super(Autoencoder_CAEv3, self).__init__()
        reduced_width = 16
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten layer
        )

        # Decoder layers
        self.decoder = nn.Sequential(

            nn.Unflatten(dim=1, unflattened_size=(16, reduced_width, reduced_width)),  # Reshape layer
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x / 255
        # Define forward pass
        x = self.encoder(x)
        x = self.decoder(x)
        x = x * 255
        return x