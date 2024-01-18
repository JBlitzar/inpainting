import matplotlib.pyplot as plt
import torch
from inpainting_model import Autoencoder_CAE, black_out_random_rectangle
import pickle
import numpy as np
import random
from matplotlib.widgets import Button
import time

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("imgnet_test.pickle")
print(data.shape)
print(data[0].shape)
print(data[0][0].shape)
def test(_=None):
    PATH = 'Inpainting_CAEimgnet.pth'
    net = Autoencoder_CAE()
    try:
        net.load_state_dict(torch.load(PATH))
        print(net.state_dict)
        print(f"Loaded from: {PATH}")
    except Exception as e:
        print(e)
        print("Cancelled model loading")
        exit()
    item = data[random.randint(0,len(data)-1)]
    input_ = np.array([item])
    input_ = torch.Tensor(input_)
    black_out_random_rectangle(input_)
    result = net(input_)
    result = result.detach().numpy()
    result = result[0].astype(int)
    print(result.shape)
    input_ = input_.numpy()[0].astype(int)

    print(input_.shape)
    # Display the first image in the first subplot
    axs[0].imshow(input_.transpose(1, 2, 0))  # Transpose to (64, 64, 3) for RGB format
    axs[0].axis('off')  # Turn off axis labels

    # Display the second image in the second subplot
    axs[1].imshow(result.transpose(1, 2, 0))
    axs[1].axis('off')

fig, axs = plt.subplots(1, 2)

test()

plt.show()
