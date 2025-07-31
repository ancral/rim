import os
import numpy as np
from PIL import Image
import tensorflow as tf
import configparser

from models.rim import Rim
from models.unet import Unet
from utils.generate_mask import createAugment, rearrange_image_and_create_mask
from utils.others import loadFiles

CONFIG_PATH = "./config/models.ini"
MASK_MATRIX = "./utils/mask_cube.npy"
CHECKPOINT_PATH = "./checkpoints/final_model.weights.h5"
OUTPUT_PATH = "./outputs"

# Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

DIM = int(config["GENERAL"]["dim_size"])
BATCH_SIZE = int(config["GENERAL"]["batch_size"])
NUM_SENSORS = int(config["GENERAL"]["num_sensors"])
SHAPE_SENSORS = int(config["GENERAL"]["shape_sensors"])
TYPE_DATA = config["GENERAL"]["type_data"]
DELETE_INDICES = [int(i.strip()) for i in config["GENERAL"]["delete_ind"].split(",") if i.strip()]
MASK_CUSTOM = config["GENERAL"].getboolean("mask_custom")

# Load test images
DATASET_TEST = os.path.join(".", "dataset", TYPE_DATA, "test")
x_test = loadFiles(DATASET_TEST, dim=DIM)

mask, padding_zone = None, None
if MASK_CUSTOM:
    mask_array = np.load(MASK_MATRIX)
    mask, padding_zone = rearrange_image_and_create_mask(mask_array, DIM)

# Create test data generator
test_gen = createAugment(
    x_test, x_test,
    mask=mask,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_sensors=NUM_SENSORS,
    type_scheme_mask=SHAPE_SENSORS,
    delete_data=DELETE_INDICES
)

# Build and load model
denoising_unet = Unet()
model = Rim(
    denoising_model=denoising_unet,
    padding_zone=padding_zone,
    rim_cfg=config["RIM"]
)

# Dummy call to build the model
_ = model(tf.zeros((1, DIM, DIM, 3), dtype=tf.float32))

# Load pretrained weights
model.load_weights(CHECKPOINT_PATH)
print("Model loaded successfully")

# Inference
preds = model.predict(test_gen, batch_size=1, verbose=1)

# Save predictions
os.makedirs(OUTPUT_PATH, exist_ok=True)
for i, pred in enumerate(preds):
    pred_img = (pred * 255).astype(np.uint8)
    Image.fromarray(pred_img).save(os.path.join(OUTPUT_PATH, f"pred_{i}.png"))

print("Inference completed. Results saved in:", OUTPUT_PATH)