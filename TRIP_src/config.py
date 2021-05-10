# Set up configurations and
# constants for model training and testing, dataset loading
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_EPOCHS = 200
IMAGE_SIZE = 256
CHANNELS_IMG = 3
SAVE_MODEL = True

L1_LAMBDA = 100
LAMBDA_GP = 10
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.0

NUM_WORKERS = 4
NUM_EPOCHS = 500

LOAD_MODEL = True
SAVE_MODEL = False

CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"

CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

TRAIN_DIR = "data/train_dir"
TEST_DIR = "data/test_dir"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0:" "image"},
)