# Set up configurations and
# constants for model training and testing, dataset loading

import torch

# Common parameters for CNN training
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

# TRIP project specific paramaters
BOX_TYPE = "EBOX"  # EBOX = estimation box, TBOX = true value box (YOLO-based)
EXECUTION_MODE = "TRAIN"  # TRAIN / TEST
MINIBATCH_SIZE = 32  # 32, 16, 64
EVAL_INTERVAL = 5
SAVE_INTERVAL = 10
LAYER_NAME = "CONV33"  # CONV33, CONV39, CONV45 (YOLO-based)
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
COMPARISON_LOSS_MARGIN = 0.3
THRESHOLD_SIMILAR_RISK = 0.11
RISK_TYPE = 'seq_risk'
WEIGHT_DECAY = 0
MOMENTUM = 0
OPTIMIZER = "ADADELTA"  # adam | adadelta | adagrad [lr=0.001]  |rmsprop [lr=0.01] | momentum_sgd [lr=0.01 momentum=0.9] |
# nesterovag [lr=0.01 momentum=0.9] | rmspropgraves [lr=0.0001 momentum=0.9] | sgd [lr=0.01] | smorms3 [lr=0.001]

GPU_ID = 0
ROI_BG = "BG_ZERO"  # Region of interest background, BG_ZERO = background zero, BG_GN = gaussian noise

# Checkpoint and directories
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"

CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

TRAIN_DIR = "data/train_dir"
TEST_DIR = "data/test_dir"
