import os
import cv2
from PIL import Image
import numpy as np

from imSlim import imSlim
from utils import to8U


def process_hdr_images(input_folder, output_folder):
    """
    Iterates through .hdr images in the input_folder,
    calls imSlim on them, and saves the results as .png in the output_folder.

    Args:
        input_folder (str): Path to the folder containing .hdr images.
        output_folder (str): Path to the folder where .png images will be saved.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".hdr"):
            hdr_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            png_path = os.path.join(output_folder, f"{base_name}.png")

            print(f"Processing '{filename}'...")
            try:
                hdr_image = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR_RGB)
                img = hdr_image.astype(np.float32)
                
                if img is None:
                    print(f"Error: Could not read HDR image '{hdr_path}'. Skipping.")
                    continue

                processed_image = imSlim(img, 0.5)
                
                if processed_image.dtype != np.uint8:
                    processed_image = to8U(processed_image)

                pil_image = Image.fromarray(processed_image)
                pil_image.save(png_path)
                print(f"Saved '{base_name}.png' to '{output_folder}'")

            except Exception as e:
                print(f"An error occurred while processing '{filename}': {e}")

                
if __name__ == "__main__":
    input_images_folder = "input_hdr/"  
    output_png_folder = "output_hdr/"

    process_hdr_images(input_images_folder, output_png_folder)
    print("\nProcessing complete.")
