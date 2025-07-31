import os
import numpy as np
import tensorflow as tf
import configparser

from models.rim import Rim
from models.unet import Unet
from utils.generate_mask import createAugment, rearrange_image_and_create_mask
from utils.metrics import loss_inpainting, gradient_diff_loss, loss_min_max
from utils.others import loadFiles

# Paths
CONFIG_PATH = "./config/models.ini"
MASK_MATRIX = "./utils/mask_cube.npy"

# Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

DIM = int(config["GENERAL"]["dim_size"])
BATCH_SIZE = int(config["GENERAL"]["batch_size"])
NUM_SENSORS = int(config["GENERAL"]["num_sensors"])
SHAPE_SENSORS = int(config["GENERAL"]["shape_sensors"])
EPOCHS = int(config["GENERAL"]["epochs"])
LR = float(config["GENERAL"]["learning_rate"])
MASK_CUSTOM = config["GENERAL"].getboolean("mask_custom")
DELETE_INDICES = [int(i.strip()) for i in config["GENERAL"]["delete_ind"].split(",") if i.strip()]
TRAINING_FROM_IMAGE = config["GENERAL"].getboolean("training_from_image")
TYPE_DATA = config["GENERAL"]["type_data"]

# Dataset paths
DATASET_TRAIN = os.path.join(".", "dataset", TYPE_DATA, "train")
DATASET_TEST = os.path.join(".", "dataset", TYPE_DATA, "test")

# Load training and test data
x_train = loadFiles(DATASET_TRAIN, dim=DIM)
x_test = loadFiles(DATASET_TEST, dim=DIM)

# Load custom mask if enabled
mask, padding_zone = None, None
if MASK_CUSTOM:
    mask_array = np.load(MASK_MATRIX)
    mask, padding_zone = rearrange_image_and_create_mask(mask_array, DIM)

# Data generators
train_gen = createAugment(
    x_train, x_train,
    mask=mask,
    batch_size=BATCH_SIZE,
    num_sensors=NUM_SENSORS,
    type_scheme_mask=SHAPE_SENSORS,
    delete_data=DELETE_INDICES
)

test_gen = createAugment(
    x_test, x_test,
    mask=mask,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_sensors=NUM_SENSORS,
    type_scheme_mask=SHAPE_SENSORS,
    delete_data=DELETE_INDICES
)

# Build model
denoising_unet = Unet()
model = Rim(
    denoising_model=denoising_unet,
    padding_zone=padding_zone,
    rim_cfg=config["RIM"]
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_inpainting,
    metrics=[gradient_diff_loss, loss_min_max]
)

# Train model
model.fit(
    train_gen,
    verbose=1,
    validation_data=test_gen,
    epochs=EPOCHS
)

# Ensure model is built before saving weights
_ = model(tf.zeros((1, DIM, DIM, 3), dtype=tf.float32))

# Save weights
os.makedirs("./checkpoints", exist_ok=True)
model.save_weights("./checkpoints/final_model.weights.h5")