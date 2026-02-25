import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from unet import UnetInference # Import the simplified class

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Execution Mode:
    # 'predict':     Predict a single image from input path.
    # 'fps':         Test inference speed (FPS).
    # 'dir_predict': Batch process all images in a folder and save results.
    # -------------------------------------------------------------------------
    mode = "predict"

    # Class names for reporting (Background + Target Layers)
    class_names = ["background", "ice_layer", "bedrock"]

    # Paths for batch prediction (dir_predict mode)
    dir_origin_path = "data/input/"
    dir_save_path   = "data/output/"

    # Initialize the predictor
    # Parameters like model_path and cuda can be passed here
    predictor = UnetInference()

    if mode == "predict":
        """ Single image prediction mode """
        while True:
            img_path = input('Input image filename (or "quit" to exit): ')
            if img_path.lower() == 'quit':
                break
            try:
                image = Image.open(img_path)
            except Exception as e:
                print(f'Error opening image: {e}')
                continue
            else:
                result_image = predictor.detect_image(image)
                result_image.show()

    elif mode == "fps":
        """ Inference speed benchmarking """
        # Ensure a sample image exists at this path for testing
        test_image_path = "data/input/sample_test.png"
        try:
            img = Image.open(test_image_path)
            test_interval = 100
            tact_time = predictor.get_FPS(img, test_interval)
            print(f"Average time: {tact_time:.4f} seconds")
            print(f"FPS: {1/tact_time:.2f} (@batch_size 1)")
        except FileNotFoundError:
            print(f"Sample image not found at {test_image_path} for FPS test.")

    elif mode == "dir_predict":
        """ Batch prediction for all images in a directory """
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        print(f"Processing images in {dir_origin_path}...")
        
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                try:
                    image = Image.open(image_path)
                    r_image = predictor.detect_image(image)
                    r_image.save(os.path.join(dir_save_path, img_name))
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    else:
        raise AssertionError("Please specify a valid mode: 'predict', 'fps', or 'dir_predict'.")
