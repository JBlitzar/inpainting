import pickle
from PIL import Image
import glob
import shutil
from tqdm import tqdm
import os
print(glob.glob("test_samples_1000/test/*/*_mask000.png"))
output_directory = "masks/"

# Set the target size for scaling down
scaled_size = (200, 200)

# Set the final size after cropping
final_size = (200, 133)
for idx, file in enumerate(tqdm(glob.glob("test_samples_1000/test/*/*_mask000.png"))):
    # Open the PNG mask
    img = Image.open(file)

    # Scale down the image to the specified size
    img.thumbnail(scaled_size)

    # Calculate the cropping box to center-crop the image
    left_margin = (img.width - final_size[0]) // 2
    top_margin = (img.height - final_size[1]) // 2
    right_margin = left_margin + final_size[0]
    bottom_margin = top_margin + final_size[1]

    # Crop the image to the specified final size
    img = img.crop((left_margin, top_margin, right_margin, bottom_margin))

    # Save the resulting image
    output_path = os.path.join(output_directory, f"{idx}.jpg")
    img.save(output_path, format="JPEG", quality=95)
