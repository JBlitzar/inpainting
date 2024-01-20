import pickle
from PIL import Image
import glob
import shutil
from tqdm import tqdm
import numpy as np
import pickle


def loadnsave(fileglob, output):
    imgs = []
    first_shape = None
    for img in tqdm(fileglob):
        myimg = Image.open(img)
        resized_img = myimg.resize((128, 128))
        arr = np.array(resized_img)
        #print("arr.shape", arr.shape)
        if first_shape == None:
            first_shape = arr.shape
        if arr.shape != first_shape:
            e = np.expand_dims(arr, axis=-1)
            arr = np.concatenate([e, e, e], axis=-1)
            assert arr.shape == first_shape
        imgs.append(np.transpose(arr, (2, 0, 1)))

    
    imgs = np.array(imgs)
    #transposed_data = np.transpose(imgs, (0, 3, 1, 2))
    print(imgs.shape)
    with open(output, 'wb+') as file:
        pickle.dump(imgs, file)
loadnsave(glob.glob("celeba_hq_256/*.jpg"), "celeba.pickle")