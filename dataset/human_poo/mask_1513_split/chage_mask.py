import os
from PIL import Image
import numpy as np

# Define the directory containing the mask images
mask_dir = '/root/ai/dataset/mask_1513_split/mask'

# Iterate through each file in the directory
for filename in os.listdir(mask_dir):
    if filename.endswith("_mask.png"):
        mask_path = os.path.join(mask_dir, filename)

        # Load the mask image
        mask = Image.open(mask_path).convert("L")

        # Convert mask to numpy array
        mask_np = np.array(mask)

        # Set all non-background values (1, 2, 3, etc.) to 255
        mask_np[mask_np > 0] = 255

        # Convert numpy array back to PIL image
        new_mask = Image.fromarray(mask_np)
        print(mask_path)
        # Save the modified mask, overwriting the original
        new_mask.save(mask_path)

print("All masks have been updated.")
