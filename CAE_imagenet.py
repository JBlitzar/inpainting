import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import pickle
print("imported")
# Set random seed for reproducibility
learning_rate = 0.001
batch_size = 64
num_epochs = 10
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

ImageNet_dataset = datasets.ImageNet(
    root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=ImageNet_dataset,
                         batch_size=batch_size, shuffle=True)
print("data loaded")


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
            nn.Sigmoid()  # Use Sigmoid for the output layer if input is normalized between 0 and 1
        )

    def forward(self, x):
        # Define forward pass
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Instantiate model, define loss function, and optimizer
model = Autoencoder_CAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("model initialized")
# Training loop
for epoch in tqdm.trange(num_epochs):
    for data, _ in tqdm.tqdm(data_loader):
        #print(data.shape)
        #inputs = data.view(data.size(0), -1)
        inputs = data
        optimizer.zero_grad()
        print("zeroed")
        outputs = model(inputs)
        print("evalled")
        loss = criterion(outputs, inputs)
        print("lossed")
        loss.backward()
        print("backwarded")
        optimizer.step()
        print("stepped")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'unsplash_autoencoder.pth')

# Save the trained model if needed
torch.save(model.state_dict(), 'unsplash_autoencoder.pth')
