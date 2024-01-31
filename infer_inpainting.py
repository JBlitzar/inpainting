import matplotlib.pyplot as plt
import torch
from inpainting_model import box_out, Autoencoder_CAE, black_out_random_rectangle, Autoencoder_CAEv2, Autoencoder_CAEv3, CelebACAE, CelebACAEv2, black_out_random_rectangle_centered, CelebACAEv3
import pickle
from matplotlib.widgets import RectangleSelector
import numpy as np
import random
from matplotlib.widgets import Button
import time
import torch.nn as nn
import tqdm
from PIL import Image


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
    PATH = 'celebaCAEv3.pth'  # celebaCAE.pth'  # v1
    net = CelebACAEv3()
    # v1 for loading up just the model, not the optimizer and stuff
    model_saving_format = "v2"
    # PATH = 'celeba/BACKUP_4celebaCAE.pth'
    # model_saving_format = "v1"
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
    # net.eval()


reload_model(_=1)


def run_model(image, top, left, rwidth, rheight):
    with torch.no_grad():
        image_ = torch.Tensor(
            np.array([np.transpose(np.array(image), (2, 0, 1))]))
        box_out(image_, top, left, rwidth, rheight)
        print(image_.size())
        result = net(image_)
        loss = None
        # loss = criterion(result, torch.Tensor(np.array([np.array(image)])))
        result = result.detach().numpy()
        result = result[0].astype(int)
        image = np.array([np.array(image)])[0].astype(int)
        return result, loss, image


def take_and_process_image():
    # Take a picture (assuming the image is saved as 'input_image.jpg')
    input_image_path = 'test2.jpg'
    input_image = Image.open(input_image_path)

    # Crop to square and resize to 128x128
    size = min(input_image.size)
    left = (input_image.width - size) // 2
    top = (input_image.height - size) // 2
    right = (input_image.width + size) // 2
    bottom = (input_image.height + size) // 2

    cropped_image = input_image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((128, 128))

    # Create a plot with the image
    fig, ax = plt.subplots()
    ax.imshow(resized_image)
    plt.title("Draw a Rectangle for Inpainting")

    # Use RectangleSelector for interactive rectangle drawing
    def onselect(eclick, erelease):
        top = int(min(eclick.ydata, erelease.ydata))
        left = int(min(eclick.xdata, erelease.xdata))
        rwidth = int(abs(erelease.xdata - eclick.xdata))
        rheight = int(abs(erelease.ydata - eclick.ydata))
        print(top, left, rwidth, rheight)

        # Run the model
        result, loss, input_image_np = run_model(
            resized_image, top, left, rwidth, rheight)

        # Display the original and inpainted images
        display_images(input_image_np, result)
        plt.close()

    selector = RectangleSelector(ax, onselect, useblit=True, button=[
                                 1], minspanx=5, minspany=5)

    plt.show()


def display_images(original, inpainted):
    original = np.array(original).astype(np.uint8)
    inpainted = np.transpose(np.array(inpainted), (1, 2, 0)).astype(np.uint8)
    print(original.shape)
    print(inpainted.shape)
    original_image = Image.fromarray(original)
    inpainted_image = Image.fromarray(inpainted)

    # Create a side-by-side display
    combined_image = Image.new(
        'RGB', (original_image.width * 2, original_image.height))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(inpainted_image, (original_image.width, 0))

    # Display the images
    combined_image.show()


take_and_process_image()
