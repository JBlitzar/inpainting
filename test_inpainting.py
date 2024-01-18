import matplotlib.pyplot as plt
import torch
from inpainting_model import Autoencoder_CAE, black_out_random_rectangle
import pickle
import numpy as np
import random

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle("imgnet_train.pickle")
print(data.shape)
print(data[0].shape)
print(data[0][0].shape)

PATH = 'Inpainting_CAEimgnet.pth'
net = Autoencoder_CAE()
try:
    net.load_state_dict(torch.load(PATH))
    print(net.state_dict)
except Exception as e:
    print(e)
    print("Cancelled model loading")
item = data[random.randint(0,len(data)-1)]
input_ = np.array([item])
input_ = torch.Tensor(input_)
black_out_random_rectangle(input_)
result = net(input_)
result = result.detach().numpy()
result = result[0]
print(result.shape)
print(item.shape)
print(item)
fig, axs = plt.subplots(1, 2)

# Display the first image in the first subplot
axs[0].imshow(item.transpose(1, 2, 0))  # Transpose to (64, 64, 3) for RGB format
axs[0].axis('off')  # Turn off axis labels

# Display the second image in the second subplot
axs[1].imshow(result.transpose(1, 2, 0))
axs[1].axis('off')

plt.show()