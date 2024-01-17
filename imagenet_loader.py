import pickle
from PIL import Image
import glob
import shutil
from tqdm import tqdm
import numpy as np
import pickle


def loadnsave(fileglob, output):
    imgs = []
    for img in tqdm(fileglob):
        myimg = Image.open(img)
        resized_img = myimg.resize((200, 133))
        arr = np.array(resized_img)
        print(arr.shape)
        imgs.append(arr)
    imgs = np.array(imgs)
    #transposed_data = np.transpose(imgs, (0, 3, 1, 2))
    print(imgs.shape)
    with open(output, 'wb+') as file:
        pickle.dump(imgs, file)
loadnsave(glob.glob("tiny-imagenet-200/train/*/images/*.JPEG"), "imgnet_train.pickle")
loadnsave(glob.glob("tiny-imagenet-200/test/images/*.JPEG"), "imgnet_test.pickle")