import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import pickle
print("imported")

USE_MPS = False
device = "cpu"
if torch.backends.mps.is_available():
    print("MPS available")
    USE_MPS = True
    device = "mps"
learning_rate = 0.001
batch_size = 64
num_epochs = 10
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("imgnet_train.pickle")
print(data.shape)
print(data[0].shape)
print(data[0][0].shape)
dataset = torch.Tensor(data)
sections_size = 100
splitted_data = tuple([t.to(device) for t in torch.split(dataset, sections_size)])
print(type(splitted_data[0]))

print("Data loaded.")
print("Shapes:")
print(dataset.size())
print("data loaded")

def black_out_random_rectangle(tensor):
    num_channels, height, width = tensor.shape

    # Randomly select the position and size of the rectangle
    top = torch.randint(0, height - 10, (1,)).item()
    left = torch.randint(0, width - 10, (1,)).item()
    rect_height = torch.randint(1, 10, (1,)).item()
    rect_width = torch.randint(1, 10, (1,)).item()

    # Black out the selected rectangle in all channels
    tensor[:, top:top+rect_height, left:left+rect_width] = 0

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
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("model initialized")
# Training loop
for epoch in tqdm.trange(num_epochs):
    pbar = tqdm.tqdm(splitted_data)
    current_loss = 0
    for data in pbar:
        #print(data.shape)
        #inputs = data.view(data.size(0), -1)
        inputs = data
        black_out_random_rectangle(inputs)
        optimizer.zero_grad()
        pbar.set_description(f"Loss: {current_loss} | eval...")
        outputs = model(inputs)
        pbar.set_description(f"Loss: {current_loss} | loss...")
        loss = criterion(outputs, data) # changed from  criterion(outputs, inputs)  because we want reconstructed to equal output
        current_loss = loss
        pbar.set_description(f"Loss: {current_loss} | backprop...")
        loss.backward()
        pbar.set_description(f"Loss: {current_loss} | Done!")
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'CAEimgenet.pth')

# Save the trained model if needed
torch.save(model.state_dict(), 'CAEimgnet.pth')
