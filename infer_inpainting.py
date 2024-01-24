import matplotlib.pyplot as plt
import torch
from inpainting_model import box_out,Autoencoder_CAE, black_out_random_rectangle, Autoencoder_CAEv2, Autoencoder_CAEv3, CelebACAE, black_out_random_rectangle_centered
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

def run_model(image, top, left, rwidth, rheight):
    with torch.no_grad():
        image = torch.Tensor(np.array([image]))
        box_out(image, top, left, rwidth, rheight)
        result = net(image)
        loss = criterion(result, torch.Tensor(np.array([image])))
        result = result.detach().numpy()
        result = result[0].astype(int)
        image = np.array([image])[0].astype(int)
        return result, loss, image
