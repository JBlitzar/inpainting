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

# Instantiate model, define loss function, and optimizer
PATH = 'Inpainting_CAEimgnet.pth'
model = Autoencoder_CAE()
try:
    model.load_state_dict(torch.load(PATH))
except Exception as e:
    print(e)
    print("Cancelled model loading")
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("model initialized")
# Training loop
for epoch in tqdm.trange(num_epochs):
    pbar = tqdm.tqdm(splitted_data)
    current_loss = 0
    for idx, data in enumerate(pbar):
        #print(data.shape)
        #inputs = data.view(data.size(0), -1)
        inputs =data
        #print(torch.all(inputs.eq(data)))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs) # changed from  criterion(outputs, inputs)  because we want reconstructed to equal output
        current_loss = loss
        pbar.set_description(f"Loss: {current_loss}")
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            torch.save(model.state_dict(), PATH)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), PATH)

# Save the trained model if needed
torch.save(model.state_dict(), PATH)
