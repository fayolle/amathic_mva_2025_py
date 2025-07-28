import os
import sys

import cv2
from PIL import Image
import numpy as np

from imSlim import imSlim
from utils import to8U, to64F


if __name__ == "__main__":
    # HDR
    hdr_path = "memorial.hdr"
    base_name = os.path.splitext(hdr_path)[0]
    png_path = base_name + "_imSlim.png"

    try:
        print(f"Processing {hdr_path}")
        hdr_image = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
        img = hdr_image.astype(np.float32)
        
        if img is None:
            print(f"Error: Could not read HDR image '{hdr_path}'. Skipping.")
            sys.exit(1)

        processed_image = imSlim(img, 0.5)
        
        if processed_image.dtype != np.uint8:
            processed_image = to8U(processed_image)

        pil_image = Image.fromarray(processed_image)
        pil_image.save(png_path)
        print(f"Saved '{base_name}.png'")

    except Exception as e:
        print(f"An error occurred while processing '{hdr_path}': {e}")


    # LLIE
    llie_path = "540.png"
    base_name = os.path.splitext(llie_path)[0]
    png_path = base_name + "_imSlim.png"

    try:
        print(f"Processing {llie_path}")
        llie_image = cv2.imread(llie_path, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
        llie_image = cv2.cvtColor(llie_image, cv2.COLOR_BGR2RGB)
        img = to64F(llie_image)

        if img is None:
            print(f"Error: Could not read image '{llie_path}'. Skipping.")
            sys.exit(1)

        processed_image = imSlim(img, 0.0)
        
        if processed_image.dtype != np.uint8:
            processed_image = to8U(processed_image)

        pil_image = Image.fromarray(processed_image)
        pil_image.save(png_path)
        print(f"Saved '{base_name}.png'")
        
    except Exception as e:
        print(f"An error occurred while processing '{llie_path}': {e}")
