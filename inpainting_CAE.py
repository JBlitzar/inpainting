import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import pickle
import numpy as np
from inpainting_model import Autoencoder_CAE, black_out_random_rectangle,Autoencoder_CAEv2, Autoencoder_CAEv3
print("imported")

USE_MPS = False
device = "cpu"
if torch.backends.mps.is_available():
    print("MPS available")
    USE_MPS = True
    device = "mps"
learning_rate = 0.005
batch_size = 64
num_epochs = 20
print("Hyperparameters: ")
print(learning_rate, batch_size, num_epochs)
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("imgnet_train.pickle")
np.random.shuffle(data)
# print(data.shape)
# print(data[0].shape)
# print(data[0][0].shape)
dataset = torch.Tensor(data)
sections_size = 100
splitted_data = tuple([t.to(device) for t in torch.split(dataset, sections_size)])

print("Data loaded.")
print("Shapes:")
print(dataset.size())



# Instantiate model, define loss function, and optimizer
PATH = 'v3Inpainting_CAEimgnet.pth'
model = Autoencoder_CAEv3()
try:
    model.load_state_dict(torch.load(PATH))
    print("Model loaded", PATH)
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
        inputs = data.detach().clone() # deepcopy
        black_out_random_rectangle(inputs)
        #print(torch.all(inputs.eq(data)))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, data) # changed from  criterion(outputs, inputs)  because we want reconstructed to equal output
        current_loss = loss
        pbar.set_description(f"Loss: {round(current_loss.item()*100)/100}")
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            torch.save(model.state_dict(), PATH)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), PATH)

# Save the trained model if needed
torch.save(model.state_dict(), PATH)
