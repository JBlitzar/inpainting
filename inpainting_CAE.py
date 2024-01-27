from inpainting_model import Autoencoder_CAE, black_out_random_rectangle, Autoencoder_CAEv2, Autoencoder_CAEv3, CelebACAE,CelebACAEv2, black_out_random_rectangle_centered, CelebACAEv3
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import numpy as np
import pickle
import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import warnings
warnings.filterwarnings("ignore")

writer = SummaryWriter()
warnings.filterwarnings("default")
print("imported")
np.random.seed()

USE_MPS = False
device = "cpu"
if torch.backends.mps.is_available():
    print("MPS available")
    USE_MPS = True
    device = "mps"

# hyperparameters
learning_rate = 0.005
batch_size = 64
num_epochs = 128
rectangle_fn = black_out_random_rectangle_centered
print("Hyperparameters: ")
print(learning_rate, batch_size, num_epochs, rectangle_fn)

# load data


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data = unpickle("celeba.pickle")
np.random.shuffle(data)
# print(data.shape)
# print(data[0].shape)
# print(data[0][0].shape)
dataset = torch.Tensor(data)
sections_size = batch_size
splitted_data = tuple([t.to(device)
                      for t in torch.split(dataset, sections_size)])

print("Data loaded.")
print("Shapes:")
print(dataset.size())


# Instantiate model, define loss function, and optimizer
PATH = 'celebaCAEv3.pth'
model = CelebACAEv3()
# v1 for loading up just the model, not the optimizer and stuff
model_loading_format = "v2"
model_saving_format = "v2"
print(model_loading_format, model_saving_format)
try:
    if model_loading_format == "v2":
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        model.load_state_dict(torch.load(PATH))
    print("Model loaded", PATH)
except Exception as e:
    print(e)
    print("Cancelled model loading")
model.train()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("model initialized")
# Training loop
for epoch in tqdm.trange(num_epochs):
    pbar = tqdm.tqdm(splitted_data, leave=False)
    current_loss = 0
    running_sum = 0
    i = 0
    for idx, data in enumerate(pbar):
        desc = f"Loss: {round(current_loss*100)/100}"
        i = idx
        # print(data.shape)
        # inputs = data.view(data.size(0), -1)
        inputs = data.detach().clone()  # deepcopy
        rectangle_fn(inputs)
        # print(torch.all(inputs.eq(data)))
        optimizer.zero_grad()
        pbar.set_description(f"eval  | {desc}")
        outputs = model(inputs)
        # changed from  criterion(outputs, inputs)  because we want reconstructed to equal output
        pbar.set_description(f"loss  | {desc}")
        loss = criterion(outputs, data)
        current_loss = loss.item()
        running_sum += loss.item()
        desc = f"Loss: {round(current_loss*100)/100}"
        pbar.set_description(f"back  | {desc}")
        loss.backward()
        optimizer.step()
        pbar.set_description(f"clean | {desc}")
        if idx % 50 == 0:
            if model_saving_format == "v2":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, PATH)
            else:
                torch.save(model.state_dict(), PATH)
    writer.add_scalar("Loss/train", running_sum/(i+1), epoch)
    print("=======================================================")
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("=======================================================")
    if model_saving_format == "v2":

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)
    else:
        torch.save(model.state_dict(), PATH)

# Save the trained model if needed
# torch.save(model.state_dict(), PATH)
if model_saving_format == "v2":
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, PATH)
else:
    torch.save(model.state_dict(), PATH)
writer.flush()
print("Done")
os.system("say 'All done'")
currenttime = datetime.now().strftime("%I:%M:%S %p")
print(f"Finished at {currenttime}")
os.system(
    f"osascript -e 'display alert \"Finished training at {currenttime} \"' &")
