from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1

# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
# Set GPU
# Set N_GPU_USE to 0 if there is no GPU
# Set N_GPU_USE to positive value to use GPUs
_C.USE_CUDA = True
_C.N_GPU_USE = 1
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bilinear"
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Mean and std (default: ImageNet)
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Random crop
_C.INPUT.CROP_PADDING = 4
# Random resized crop
_C.INPUT.RRCROP_SCALE = (0.08, 1.0)
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.0
_C.INPUT.GN_STD = 0.15
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10
# ColorJitter (brightness, contrast, saturation, hue)
_C.INPUT.COLORJITTER_B = 0.4
_C.INPUT.COLORJITTER_C = 0.4
_C.INPUT.COLORJITTER_S = 0.4
_C.INPUT.COLORJITTER_H = 0.1
# Random gray scale's probability
_C.INPUT.RGS_P = 0.2
# Gaussian blur
_C.INPUT.GB_P = 0.5  # propability of applying this operation
_C.INPUT.GB_K = 21  # kernel size (should be an odd number)

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ""
_C.DATASET.NAME = ""
_C.DATASET.NUM_CLASSES = 0
# Percentage of validation data (only used for SSL datasets)
# Set to 0 if do not want to use val data
_C.DATASET.VAL_PERCENT = 0.2

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# img0 denotes image tensor without augmentation
# Useful for consistency learning
_C.DATALOADER.RETURN_IMG0 = False
# Setting for the train data-loader
_C.DATALOADER.TRAIN = CN()
_C.DATALOADER.TRAIN.BATCH_SIZE = 32

# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.BATCH_SIZE = 32

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ""
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ""
_C.MODEL.BACKBONE.PRETRAINED = True
# Definition of embedding layers
_C.MODEL.HEAD = CN()
# If none, do not construct embedding layers, the
# backbone's output will be passed to the classifier
_C.MODEL.HEAD.NAME = ""
# Structure of hidden layers (a list), e.g. [512, 512]
# If undefined, no embedding layer will be constructed
_C.MODEL.HEAD.HIDDEN_LAYERS = ()
_C.MODEL.HEAD.ACTIVATION = "relu"
_C.MODEL.HEAD.BN = True
_C.MODEL.HEAD.DROPOUT = 0.0

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "adam"
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
# The following also apply to other
# adaptive optimizers like adamw
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "single_step"
# -1 or 0 means the stepsize is equal to max_epoch
_C.OPTIM.STEPSIZE = (-1, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10
# Set WARMUP_EPOCH larger than 0 to activate warmup training
_C.OPTIM.WARMUP_EPOCH = -1
# Either linear or constant
_C.OPTIM.WARMUP_TYPE = "linear"
# Constant learning rate when type=constant
_C.OPTIM.WARMUP_CONS_LR = 1e-5
# Minimum learning rate when type=linear
_C.OPTIM.WARMUP_MIN_LR = 1e-5
# Recount epoch for the next scheduler (last_epoch=-1)
# Otherwise last_epoch=warmup_epoch
_C.OPTIM.WARMUP_RECOUNT = True

###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to only save the last one
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Use 'train_x', 'train_u' or 'smaller_one' to count
# the number of iterations in an epoch (for DA and SSL)
_C.TRAIN.COUNT_ITER = "train_x"

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = "Classification"
_C.TEST.PER_CLASS_RESULT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# Use test or val set for FINAL evaluation
_C.TEST.SPLIT = "test"
# Which model to test after training (last_step or best_val)
# If best_val, evaluation is done every epoch (if val data
# is unavailable, test data will be used)
_C.TEST.FINAL_MODEL = "last_step"

###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ""

######
# Specific hyperparameters for methods
######
# Baseline
_C.TRAINER.BASELINE = CN()
_C.TRAINER.BASELINE.ONLY = True
# MCD
_C.TRAINER.MCD = CN()
_C.TRAINER.MCD.N_STEP_F = 4  # number of steps to train F
# MME
_C.TRAINER.MME = CN()
_C.TRAINER.MME.LMDA = 0.1  # weight for the entropy loss
# CDAC
_C.TRAINER.CDAC = CN()
_C.TRAINER.CDAC.CLASS_LR_MULTI = 10
_C.TRAINER.CDAC.RAMPUP_COEF = 30
_C.TRAINER.CDAC.RAMPUP_ITRS = 1000
_C.TRAINER.CDAC.TOPK_MATCH = 5
_C.TRAINER.CDAC.P_THRESH = 0.95
_C.TRAINER.CDAC.STRONG_TRANSFORMS = ()
# SE (SelfEnsembling)
_C.TRAINER.SE = CN()
_C.TRAINER.SE.EMA_ALPHA = 0.999
_C.TRAINER.SE.CONF_THRE = 0.95
_C.TRAINER.SE.RAMPUP = 300
# M3SDA
_C.TRAINER.M3SDA = CN()
_C.TRAINER.M3SDA.LMDA = 0.5  # weight for the moment distance loss
_C.TRAINER.M3SDA.N_STEP_F = 4  # follow MCD
# DAEL
_C.TRAINER.DAEL = CN()
_C.TRAINER.DAEL.WEIGHT_U = 0.5  # weight on the unlabeled loss
_C.TRAINER.DAEL.CONF_THRE = 0.95  # confidence threshold
_C.TRAINER.DAEL.STRONG_TRANSFORMS = ()
