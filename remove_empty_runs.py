import os
import shutil

def remove_small_subfolders(root_folder, threshold_bytes=100):
    # Walk through the directory tree
    for folder_path, _, _ in os.walk(root_folder):
        # Calculate the size of the current folder
        folder_size = sum(os.path.getsize(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)))

        # Check if the folder size is below the threshold
        if folder_size < threshold_bytes:
            print(f"Removing {folder_path}")
            # Uncomment the line below to actually remove the folder
            shutil.rmtree(folder_path)
        else:
            print(f"not removing {folder_path}")

runs_folder_path = 'runs'
remove_small_subfolders(runs_folder_path)
