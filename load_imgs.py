import pickle
from PIL import Image
import glob
import shutil
from tqdm import tqdm


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
