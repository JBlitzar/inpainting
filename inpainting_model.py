import torch.nn as nn
import torch
import os
from collections import OrderedDict
def alert(msg):
    os.system(
    f"osascript -e 'display alert \"{msg}\"' &")

def box_out(tensor, top, left, rwidth, rheight):
    batch_size, num_channels, height, width = list(tensor.shape)
    for i in range(batch_size):
        tensor[i, :, top:min(top+rheight, height),
               left:min(left+rwidth, width)] = 0


def black_out_random_rectangle_centered(tensor):
    batch_size, num_channels, height, width = list(tensor.shape)

    for i in range(batch_size):
        # Randomly select the position and size of the rectangle
        top = torch.randint(0, int(height/2)-2, (1,)).item()
        left = torch.randint(0, int(width/2)-2, (1,)).item()
        rect_height = torch.randint(int(height/4), int(height/2), (1,)).item()
        rect_width = torch.randint(int(height/4), int(width/2), (1,)).item()
        # Black out the selected rectangle in all channels for the current image
        tensor[i, :, top:min(top+rect_height, height),
               left:min(left+rect_width, width)] = 0


def black_out_random_rectangle(tensor):
    batch_size, num_channels, height, width = list(tensor.shape)

    for i in range(batch_size):
        # Randomly select the position and size of the rectangle
        top = torch.randint(0, height-2, (1,)).item()
        left = torch.randint(0, width-2, (1,)).item()
        rect_height = torch.randint(1, int(height/2), (1,)).item()
        rect_width = torch.randint(1, int(width/2), (1,)).item()
        # Black out the selected rectangle in all channels for the current image
        tensor[i, :, top:min(top+rect_height, height),
               left:min(left+rect_width, width)] = 0


def black_out_random_rectangle_legacy(tensor):
    batch_size, num_channels, height, width = list(tensor.shape)

    for i in range(batch_size):
        # Randomly select the position and size of the rectangle
        top = torch.randint(0, height-2, (1,)).item()
        left = torch.randint(0, width-2, (1,)).item()
        rect_height = torch.randint(1, height-top, (1,)).item()
        rect_width = torch.randint(1, width-left, (1,)).item()
        # Black out the selected rectangle in all channels for the current image
        tensor[i, :, top:min(top+rect_height, height),
               left:min(left+rect_width, width)] = 0


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
            nn.Upsample(scale_factor=2, mode='nearest'),  # bilinear, nearest

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

            nn.Unflatten(dim=1, unflattened_size=(
                16, reduced_width, reduced_width)),  # Reshape layer

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


class CelebACAE(nn.Module):
    def __init__(self):
        super(CelebACAE, self).__init__()

        # works on 128x128
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(


            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x / 255
        # Define forward pass
        x = self.encoder(x)
        x = self.decoder(x)
        # x = x * 255
        return x
    
class CelebACAEv2(nn.Module):
    def __init__(self):
        self.quiet = False
        super(CelebACAEv2, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1 ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        print("hi")
        # x = x / 255
        # Define forward pass
        if(not self.quiet):
            print("Input Size:", x.size())

        # Encoder forward pass with printing
        for layer in self.encoder:
            x = layer(x)
            if(not self.quiet):
                print(f"{layer.__class__.__name__} Output Size:", x.size())
        if(not self.quiet):
            print("\n=====Begin decoding=====\n")
        # Decoder forward pass with printing
        for layer in self.decoder:
            x = layer(x)
            if(not self.quiet):
                print(f"{layer.__class__.__name__} Output Size:", x.size())

        # x = x * 255
        return x
class CelebAUnet(nn.Module):
    def __init__(self):
        self.quiet = True
        super(CelebAUnet, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Define forward pass
        if not self.quiet:
            print("Input Size:", x.size())

        skip_connections = []

        # Encoder forward pass with saving skip connections
        for layer in self.encoder:
            x = layer(x)
            if "pool" in layer.__class__.__name__.lower():
                skip_connections.append(x.clone())  # Save skip connection
            if not self.quiet:
                print(f"{layer.__class__.__name__} Output Size:", x.size())
           
        if not self.quiet:
            print("\n=====Begin decoding=====\n")
        skipnum = 1
        # Decoder forward pass with using skip connections
        for i, layer in enumerate(self.decoder):
            if "Upsample" in layer.__class__.__name__:
                x = torch.cat((x, skip_connections[-(skipnum)]), dim=1)  # Concatenate with skip connection
                x = layer(x)
                
                skipnum += 1
            else:
                x = layer(x)

            if not self.quiet:
                print(f"{layer.__class__.__name__} Output Size:", x.size())

        return x
class CelebAUnetv2(nn.Module):
    def __init__(self):
        self.quiet = True
        super(CelebAUnetv2, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
    ('relu2', nn.ReLU()),
    ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

    ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
    ('relu3', nn.ReLU()),
    ('conv4', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
    ('relu4', nn.ReLU()),
    ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

    ('conv5', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
    ('relu5', nn.ReLU()),
    ('conv6', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
    ('relu6', nn.ReLU()),
    ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
]))

        # Decoder layers
        self.decoder = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
    ('relu1', nn.ReLU()),
    ('upsample1', nn.Upsample(scale_factor=2, mode='nearest')),

    ('conv2', nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
    ('relu2', nn.ReLU()),
    ('conv3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
    ('relu3', nn.ReLU()),
    ('upsample2', nn.Upsample(scale_factor=2, mode='nearest')),

    ('conv4', nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1)),
    ('relu4', nn.ReLU()),
    ('conv5', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
    ('relu5', nn.ReLU()),
    ('upsample3', nn.Upsample(scale_factor=2, mode='nearest')),
    ('conv6', nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
    ("relu6",nn.ReLU()),
    ('conv7', nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)),
    ('sigmoid', nn.Sigmoid())
]))

    def forward(self, x):
        # Define forward pass
        if not self.quiet:
            print("Input Size:", x.size())

        skip_connections = []

        # Encoder forward pass with saving skip connections
        for layer_name, layer in self.encoder.named_children():
            x = layer(x)
            if "pool" in layer.__class__.__name__.lower():
                skip_connections.append(x.clone())  # Save skip connection
            if not self.quiet:
                print(f"{layer.__class__.__name__} Output Size:", x.size())
           
        if not self.quiet:
            print("\n=====Begin decoding=====\n")
        skipnum = 1
        # Decoder forward pass with using skip connections
        for layer_name, layer in self.decoder.named_children():
            if "Upsample" in layer.__class__.__name__:
                if(layer_name != "upsample3"):
                    x = torch.cat((x, skip_connections[-(skipnum)]), dim=1)  # Concatenate with skip connection
                    x = layer(x)
                    
                    skipnum += 1
                else:
                    x = layer(x)
            else:
                x = layer(x)

            if not self.quiet:
                print(f"{layer.__class__.__name__} Output Size:", x.size())

        return x

# class CelebACAEv2(nn.Module):
#     def __init__(self):
#         dropout_rate = 0.2
#         super(CelebACAEv2, self).__init__()
#         # removed one downsample/upsample layer for a larger bottleneck

#         # works on 128x128
#         # Encoder layers
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout_rate),

#         )

#         # Decoder layers
#         self.decoder = nn.Sequential(




#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x / 255
#         # Define forward pass
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = x * 255
#         return x


# class CelebACAEv3(nn.Module):
#     def __init__(self):
#         dropout_rate = 0.0
#         super(CelebACAEv3, self).__init__()
#         # removed one downsample/upsample layer for a larger bottleneck

#         # works on 128x128
#         # Encoder layers
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(dropout_rate),
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#             nn.Linear(4096, 2048)

#         )

#         # Decoder layers
#         self.decoder = nn.Sequential(
#             nn.Linear(2048, 4096),
#             nn.Unflatten(dim=1, unflattened_size=(16, 16, 16)),

#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),  # Bilinear or nearest

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x / 255
#         # Define forward pass
#         x = self.encoder(x)
#         x = self.decoder(x)
#         x = x * 255
#         return x
