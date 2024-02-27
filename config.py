import os

import numpy as np
import torch
from monai.losses import GeneralizedDiceLoss, MaskedDiceLoss
from torch.nn import CrossEntropyLoss

from utils.MaskedDiceCELoss import MaskedDiceCELoss
from utils.MaskedGeneralizedDiceCELoss import MaskedGeneralizedDiceCELoss
from utils.MaskedGeneralizedDiceLoss import MaskedGeneralizedDiceLoss

# base path of the dataset
DATASET_PATH = os.path.join("/work3/s210289/msc_thesis/data", "volumes")

# define the path to the images and masks dataset
IMG_NAME = 'downsampledR1.nii'
SEG_NAME = 'downsampledR1_54AxonSeg.nii'
SEGX_NAME = 'combined_IntensityVolume_R1to4_ExtraAxonalStructuresSegmentation.nii'
IMAGE_PATH = os.path.join(DATASET_PATH, IMG_NAME)
SEG_PATH = os.path.join(DATASET_PATH, SEG_NAME)
SEGX_PATH = os.path.join(DATASET_PATH, SEGX_NAME)

# define the test split
TEST_SPLIT = 0.1

# determine (based on availability) the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = DEVICE == "cuda"

# define the number of channels in the input, number of classes, and number of levels in the U-Net model
NUM_CHANNELS = 1  # black and white image
NUM_CLASSES = 6

# Unet
CH = False
K = 3

# define the input image dimensions
INPUT_IMAGE_WIDTH = 410
INPUT_IMAGE_HEIGHT = 410
INPUT_IMAGE_DEPTH = 410

# TorchIO: define the input patch dimensions, saples per volume, stride, and queue length
PATCH_SIZE = 128
SAMPLES_PER_VOLUME = 40
STRIDE = np.array([20, 20, 20])
MAX_QUEUE_LENGTH = 300
PATCH_OVERLAP = np.array(-(STRIDE - PATCH_SIZE))

# initialize learning rate, number of epochs to train for, the batch size, and the number of workers for process parallelization
INIT_LR = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 9
NUM_WORKERS = 0

# Aggregate final volume
AGGREGATOR = True

# LOSS
NAME_LOSS = 'MaskedGenDice'
LOSS = MaskedGeneralizedDiceLoss(include_background=True, to_onehot_y=True)
# LOSS = MaskedDiceLoss(include_background = True, to_onehot_y=True)
# LOSS = MaskedGeneralizedDiceCELoss(include_background = True, to_onehot_y=True)
# LOSS = GeneralizedDiceLoss(include_background = True, to_onehot_y=True)
# LOSS = CELoss(include_background = True, to_onehot_y=True)
# LOSS = MaskedDiceCELoss(include_background = True, to_onehot_y=True)

TEST_LOSS = MaskedGeneralizedDiceLoss(include_background=True, to_onehot_y=True)

# define threshold to filter weak predictions (help classify the pixels into one of the two classes in our binary classification-based segmentation task)
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"
