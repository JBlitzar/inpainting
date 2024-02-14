import warnings
warnings.filterwarnings("ignore")  # libressl thing
from inpainting_model import CelebAUnetv2,CelebAUnet,Autoencoder_CAE, black_out_random_rectangle, Autoencoder_CAEv2, Autoencoder_CAEv3, CelebACAE, CelebACAEv2, black_out_random_rectangle_centered
from inpainting_CAE_setup import CelebADataset,ValidationLossEarlyStopping
from losses import PSNR
from colorama import Fore, Back, Style
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
import pickle
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torch.utils.data import random_split

import matplotlib.pyplot as plt

QUIET = True

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
learning_rate = 0.001
batch_size = 128
num_epochs = 128
rectangle_fn = black_out_random_rectangle_centered
print("Hyperparameters: ")
print(learning_rate, batch_size, num_epochs, rectangle_fn)

# load data


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def savepickle(filename, obj):
    with open(filename, 'wb+') as file:
        pickle.dump(obj, file)


CACHE = False
"""if os.path.exists("cached_data.pickle") and CACHE:
    print("===============================")
    print(Fore.YELLOW+"IMPORTANT: DATA LOADED FROM CACHE"+Style.RESET_ALL)
    print("===============================")
    splitted_data = unpickle("cached_data.pickle")[0]
else:
    print("no cache found/cache disabled, applying transforms")
    data = unpickle("celeba.pickle")
    print("data unpickled")
    np.random.shuffle(data)
    # print(data.shape)
    # print(data[0].shape)
    # print(data[0][0].shape)
    dataset = torch.Tensor(data)
    print("data converted")
    print("transforming...")
    transforms = v2.Compose([  # epic data augmentation
        v2.RandomRotation(20),
        v2.RandomResizedCrop(
            size=(128, 128), antialias=True, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    # dataset = transforms(dataset)
    print(dataset.size())
    # cast to numpy, cast to tensor
    dataset = np.array([transforms(d).numpy() for d in tqdm.tqdm(dataset)])
    dataset = torch.Tensor(dataset)  # Cast to tensor takes forever
    print(dataset.size())
    print("data transformed")
    sections_size = batch_size
    splitted_data = tuple([t.to(device)
                           for t in torch.split(dataset, sections_size)])
    print(f"Moved to {device}")
    train_size = int(0.8 * len(splitted_data))
    test_size = len(splitted_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        splitted_data, [train_size, test_size])
    print("splitted")
    savepickle("cached_data.pickle", [train_dataset, test_dataset])
"""
transforms = v2.Compose([  # epic data augmentation
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),#v2.ToTensor(), # toTensor is deprecated
    v2.RandomRotation(20),
    v2.RandomResizedCrop(
        size=(128, 128), antialias=True, scale=(0.8, 1.0)), # augmentations are only supposed to happen on the training set.
    v2.RandomHorizontalFlip(p=0.5),
])
dataset = CelebADataset()#transform=transforms)
# augmentations only on train
validation_split = 0.2

# Calculate the sizes of training and validation sets
dataset_size = len(dataset)
validation_size = int(validation_split * dataset_size)
train_size = dataset_size - validation_size

# Use random_split to get the indices for training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

train_dataset._add_transform(transforms) # augmentations only on train

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Data loaded.")


# Instantiate model, define loss function, and optimizer
PATH = 'celebaUnetv2.pth'
model = CelebAUnetv2()
model.train()
model.to(device)
# v1 for loading up just the model, not the optimizer and stuff
model_loading_format = "v2"
model_saving_format = "v2"
print(model_loading_format, model_saving_format)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
try:
    #raise IndentationError
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
    print("=========IMPORTANT=========")
    print(e)
    print(Fore.YELLOW+"Cancelled model loading"+Style.RESET_ALL)
    print()
    print("=========IMPORTANT=========")

print("model initialized")
def validate(model, criterion, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            inputs = data.detach().clone().to(device)
            rectangle_fn(inputs)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, data).to(device)
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    return average_loss
writer = None
prev_loss = 1000
earlyStopping = ValidationLossEarlyStopping()
# Training loop
for epoch in tqdm.trange(num_epochs):
    pbar = tqdm.tqdm(train_loader, leave=False)
    current_loss = 0
    running_sum = 0
    i = 0
    last_input = None
    last_output = None
    model.train()
    for idx, data in enumerate(pbar):
        desc = f"Loss: {round(current_loss,4)}"
        if QUIET:
            pbar.set_description(desc)
        #pbar.set_description(f"2dev  | {desc}")
        data = data.to(device)

        i = idx
        # print(data.shape)
        # inputs = data.view(data.size(0), -1)
        inputs = data.detach().clone().to(device)  # deepcopy
        rectangle_fn(inputs)
        last_input = inputs
        # print(torch.all(inputs.eq(data)))

        optimizer.zero_grad()
        if not QUIET:
            pbar.set_description(f"eval  | {desc}")
        outputs = model(inputs)
        last_output = outputs

        # changed from  criterion(outputs, inputs)  because we want reconstructed to equal output
        if not QUIET:
            pbar.set_description(f"loss  | {desc}")
        loss = criterion(outputs, data).to(device)
        current_loss = loss.item()
        running_sum += loss.item()
        desc = f"Loss: {round(current_loss,4)}"

        if not QUIET:
            pbar.set_description(f"back  | {desc}")
        loss.backward()

        optimizer.step()
        if not QUIET:
            pbar.set_description(f"clean | {desc}")
        """if idx % 50 == 0:
            if model_saving_format == "v2":
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, PATH)
            else:
                torch.save(model.state_dict(), PATH)"""
    
    avg_val_loss = validate(model, criterion, val_loader, device)

    if not writer:
        writer = SummaryWriter()
    writer.add_scalar("Loss/train", running_sum/(i+1), epoch)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    
    # if(earlyStopping.early_stop_check(avg_val_loss)):
    #     print("=======================================================")
    #     print(
    #         Fore.CYAN+'Early Stop triggered'+Style.RESET_ALL)
    #     print("=======================================================")

    #     break

    print("=======================================================")
    print(
        Fore.CYAN+f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}'+Style.RESET_ALL)
    print("=======================================================")

    comparisoninp = inputs[0]
    comparisonout = outputs[0]
    comparisoninp = torch.transpose(comparisoninp, 0, 2)
    comparisonout = torch.transpose(comparisonout, 0, 2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Displaying the first image
    axes[0].imshow(comparisoninp.detach().cpu())
    axes[0].set_title('Original')
    axes[0].axis('off')  # Hide the axes ticks

    # Displaying the second image
    axes[1].imshow(comparisonout.detach().cpu())
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')  # Hide the axes ticks
    plt.savefig(f"train_imgs/image_{epoch}.png")
    plt.close()


    if  (running_sum/(i+1)) - prev_loss > 0.1:

        try:
            #raise IndentationError
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
            print("=========IMPORTANT=========")
            print(e)
            print(Fore.YELLOW+"Cancelled model loading"+Style.RESET_ALL)
            print()
            print("=========IMPORTANT=========")

    else:

        prev_loss = (running_sum/(i+1))

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
