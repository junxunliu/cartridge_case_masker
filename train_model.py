import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from PIL import Image

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


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
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 4 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
config.display()


class CartridgeCasesDataset(utils.Dataset):

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # def from_yaml_get_class(self, image_id):
    #     info = self.image_info[image_id]
    #     with open(info['yaml_path']) as f:
    #         data = yaml.load(f.read(), Loader=yaml.FullLoader)
    #         labels = data['label_names']
    #         del labels[0]
    #     return labels

    def from_txt_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['txt_path']) as f:
            labels = [line.strip() for line in f.readlines()]
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # override load_shapes
    def load_shapes(self, count, img_folder, mask_folder, img_list, dataset_root_path):
        self.add_class("shapes", 1, "breech-face impression")
        self.add_class("shapes", 2, "aperture shear")
        self.add_class("shapes", 3, "firing pin impression")
        self.add_class("shapes", 4, "firing pin drag")

        for i in range(count):
            file_str = img_list[i].split(".")[0]

            mask_path = mask_folder + "/" + file_str + ".png"
            txt_path = dataset_root_path + "labelme_json/" + file_str + "/label_names.txt"
            print(dataset_root_path + "labelme_json/" + file_str + "/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + file_str + "/img.png")

            self.add_image("shapes", image_id=i, path=img_folder + "/" + img_list[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, txt_path=txt_path)

    # override load_mask
    def load_mask(self, image_id):
        """Load instance masks for the given image_id.
        """

        global iter_num

        # print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1
        img = Image.open(info["mask_path"])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, -1]))

        labels = []
        labels = self.from_txt_get_class(image_id)
        labels_from = []

        for i in range(len(labels)):
            if labels[i] == "breech-face impression":
                labels_from.append("breech-face impression")
            if labels[i] == "aperture shear":
                labels_from.append("aperture shear")
            if labels[i] == "firing pin impression":
                labels_from.append("firing pin impression")
            if labels[i] == "firing pin drag":
                labels_from.append("firing pin drag")

        class_ids = np.array([self.class_names.index(s) for s in labels_from])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


dataset_root_path = "mydata/"
img_folder = dataset_root_path + "pic"
mask_folder = dataset_root_path + "cv2_mask"
img_list = os.listdir(img_folder)
count = len(img_list)

# prepare for train and val
dataset_train = CartridgeCasesDataset()
dataset_train.load_shapes(count, img_folder, mask_folder, img_list, dataset_root_path)
dataset_train.prepare()

print("dataset_train-->", dataset_train.image_ids)

dataset_val = CartridgeCasesDataset()
dataset_val.load_shapes(25, img_folder, mask_folder, img_list, dataset_root_path)
dataset_val.prepare()

print("dataset_val-->", dataset_val.image_ids)

# # Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30,
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_cartridge_cases_real_size.h5")
# model.keras_model.save_weights(model_path)

