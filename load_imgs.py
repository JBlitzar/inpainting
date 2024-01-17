import pickle
from PIL import Image
import glob
import shutil
from tqdm import tqdm
import numpy as np
import pickle


for idx, file in enumerate(tqdm(glob.glob("unsplash/low res/*_6.jpg"))):
    img = Image.open(file)

    # Get the original width and height
    original_width, original_height = img.size

    # Define the new size (1/6 of the original)
    new_width = original_width // 6
    new_height = original_height // 6

    # Resize the image
    resized_img = img.resize((new_width, new_height))
    resized_img.save(f"images/{idx}.jpg")
imgs = []
for img in tqdm(glob.glob("images/*")):
    myimg = Image.open(img)
    resized_img = myimg.resize((200, 133))
    arr = np.array(resized_img)
    print(arr.shape)
    imgs.append(arr)
imgs = np.array(imgs)
transposed_data = np.transpose(imgs, (0, 3, 1, 2))
print(transposed_data.shape)
with open("images.pickle", 'wb+') as file:
    pickle.dump(transposed_data, file)
