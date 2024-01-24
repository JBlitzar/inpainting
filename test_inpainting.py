import matplotlib.pyplot as plt
import torch
from inpainting_model import Autoencoder_CAE, black_out_random_rectangle, Autoencoder_CAEv2, Autoencoder_CAEv3, CelebACAE, black_out_random_rectangle_centered
import pickle
import numpy as np
import random
from matplotlib.widgets import Button
import time
import torch.nn as nn
import tqdm


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


criterion = nn.MSELoss()
data = unpickle("celeba.pickle")
print(data.shape)
print(data[0].shape)
print(data[0][0].shape)
rectangle_fn = black_out_random_rectangle_centered
print(rectangle_fn)
# PATH = 'inpaintingv1/BACKUP_2Inpainting_CAEimgnet.pth'
# net = Autoencoder_CAE()
# PATH = 'inpaintingv2/BACKUP2_v2Inpainting_CAEimgnet.pth'
# PATH = "v2Inpainting_CAEimgnet.pth"
# net = Autoencoder_CAEv2()
net = None


def reload_model(_=None):
    global net
    PATH = 'celebaCAE.pth'  # v1
    net = CelebACAE()
    # v1 for loading up just the model, not the optimizer and stuff
    model_saving_format = "v2"
    PATH = 'celeba/BACKUP_4celebaCAE.pth'
    model_saving_format = "v1"
    try:
        if model_saving_format == "v2":
            checkpoint = torch.load(PATH)
            print(type(checkpoint))
            print(checkpoint.keys())
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(torch.load(PATH))
        print("Model loaded", PATH)
    except Exception as e:
        print(e)
        print("Cancelled model loading")
    # PATH = 'v3Inpainting_CAEimgnet.pth'
    # net = Autoencoder_CAEv3()
    # PATH = 'inpaintingv1/BACKUP_2Inpainting_CAEimgnet.pth'
    # PATH = "Inpainting_CAEimgnet.pth"
    # net = Autoencoder_CAE()
    # PATH = 'inpaintingv2/BACKUP2_v2Inpainting_CAEimgnet.pth'
    # PATH = "v2Inpainting_CAEimgnet.pth"
    # net = Autoencoder_CAEv2()
    net.eval()


reload_model(_=1)


def get_avg_loss(_=None):
    with torch.no_grad():
        losses = []
        for item in tqdm.tqdm(data):
            # item = data[random.randint(0,len(data)-1)]
            input_ = np.array([item])
            input_ = torch.Tensor(input_)
            rectangle_fn(input_)
            result = net(input_)
            loss = criterion(result, torch.Tensor(np.array([item])))
            losses.append(loss.item())
        avg_loss = sum(losses)/len(data)
        print("Loss:", avg_loss)
        plt.text(0.5, 2, f"Loss: {avg_loss}")


def test(_=None):
    with torch.no_grad():
        item = data[random.randint(0, len(data)-1)]
        input_ = np.array([item])
        input_ = torch.Tensor(input_)
        rectangle_fn(input_)
        result = net(input_)
        loss = criterion(result, torch.Tensor(np.array([item])))
        result = result.detach().numpy()
        result = result[0].astype(int)
        # print(result.shape)
        input_ = input_.numpy()[0].astype(int)

        # print(input_.shape)
        # Display the first image in the first subplot
        # Transpose to (64, 64, 3) for RGB format
        axs[0].imshow(input_.transpose(1, 2, 0))
        axs[0].axis('off')  # Turn off axis labels
        print(loss.item())
        # Display the second image in the second subplot
        axs[1].imshow(result.transpose(1, 2, 0))
        axs[1].axis('off')
        plt.draw()


fig, axs = plt.subplots(1, 2)
buttonax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
btest = Button(buttonax, 'Test')
btest.on_clicked(test)
test()
rbutonax = fig.add_axes([0.5, 0.05, 0.1, 0.075])
breset = Button(rbutonax, 'Reload')
breset.on_clicked(reload_model)

lbutonax = fig.add_axes([0.3, 0.05, 0.1, 0.075])
bloss = Button(lbutonax, 'Loss')
bloss.on_clicked(get_avg_loss)


plt.show()
