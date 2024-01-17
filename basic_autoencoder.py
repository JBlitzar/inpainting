import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)
learning_rate = 0.001
batch_size = 64
num_epochs = 10
# MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset,
                         batch_size=batch_size, shuffle=True)


class Autoencoder_v1(nn.Module):
    def __init__(self):
        input_size = 784  # 28x28 images flattened
        hidden_size = 256
        encoding_size = 64
        super(Autoencoder_v1, self).__init__()

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, encoding_size)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Use Sigmoid for the output layer if input is normalized between 0 and 1
        )

    def forward(self, x):
        # Define forward pass
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Instantiate model, define loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for data, _ in data_loader:
        inputs = data.view(data.size(0), -1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model if needed
torch.save(model.state_dict(), 'mnist_autoencoder.pth')
