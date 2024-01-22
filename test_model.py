import os
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import cv2
from matplotlib import pyplot as plt

# Root directory of the project
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models/mask_rcnn_shapes_0030.h5")

# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 5 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 768

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
# Load weights trained on MS COCO, but skip layers that
# are different due to the different number of classes
# See README for instructions to download the COCO weights
model.load_weights(MODEL_WEIGHTS, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'breech-face impression', 'aperture shear',
               'firing pin impression', 'firing pin drag']

# Visualize results
bgr_colors = [(0.0, 0.0, 1.0),  # breech-face impression (red)
              (0.0, 1.0, 0.0),  # aperture shear (green)
              (128 / 255.0, 0.0, 128 / 255.0),  # firing pin impression (purple)
              (1.0, 1.0, 0.0)]  # firing pin drag (light blue)

# GUI
def select_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg;*.png")])
    if file_paths:
        listbox.delete(0, tk.END)
        for file_path in file_paths:
            listbox.insert(tk.END, file_path)


def generate_images():
    file_paths = listbox.get(0, tk.END)
    if not file_paths:
        messagebox.showerror("Error", "No files selected!")
        return

    for file_path in file_paths:
        try:
            # Load the selected image
            image = cv2.imread(file_path)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # check current time
            current = datetime.now()
            # Run detection
            results = model.detect([image], verbose=1)
            end = datetime.now()
            print("Detected time:", end - current)

            # Generate a related image filename
            name, ext = os.path.splitext(os.path.basename(file_path))
            new_file_path = ROOT_DIR + "/" + name + "_masked" + ext
            # Visualize results
            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'], show_bbox=False,
                                        colors=bgr_colors, file_path=new_file_path, captions=None)
            print(new_file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate the related image for {file_path}.\n{str(e)}")
            continue

    messagebox.showinfo("Success", "Related masking generated at path: {}".format(ROOT_DIR))


# Create the main window
root = tk.Tk()
root.title("Cartridge Case Image Masker             by Devin Liu")

root.geometry("600x400")

label = tk.Label(root, text="Selected Images:")
label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

listbox = tk.Listbox(root, width=50, selectmode=tk.EXTENDED)
listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

# Create a "Select" button for choosing images
select_button = tk.Button(root, text="Select Images", command=select_images)
select_button.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5, 0))

# Create a "Mask" button to create related images
generate_button = tk.Button(root, text="Mask Images", command=generate_images)
generate_button.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5, 10))

root.mainloop()
