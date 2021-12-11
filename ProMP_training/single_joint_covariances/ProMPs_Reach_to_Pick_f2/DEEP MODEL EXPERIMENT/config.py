from datetime import datetime
import os,inspect

#FOLDERS DEFINITNION
#Define date.
now = datetime.now().strftime("%d_%m__%H_%M")
# Experiment directory.
EXP_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
J_DIR =os.path.dirname(EXP_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(J_DIR))
# Data.
#RGB images folder.
IMAGE_PATH = os.path.join(ROOT_DIR, "data/color_img/")
DEPTH_PATH = os.path.join(ROOT_DIR, "data/depth/")

#Annotations folder.
ANNOTATION_PATH = os.path.join(J_DIR, "annotations/")

# ProMPs deep model outputs
OUTPUT_PATH= os.path.join(EXP_DIR, "PLOTS/traj & weights pred distributions")
LOSS_PATH = os.path.join(EXP_DIR, "LOSS", now)
METRIC_PATH = os.path.join(EXP_DIR, "METRIC")
LOG_DIR = os.path.join(EXP_DIR, "LOGS", now)
#Save the model
MODEL_FOLDER= os.path.join(EXP_DIR, "MODELS")
MODEL_NAME = f"model_{now}"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

# HYPERPARAMETERS.
#Batch sze.
batch = 5
# Number of epochs
epochs = 600
# Learning rate
lr = 0.0001
# Validation set percentage
val_frac = 0.2
# Wether to use validation samples during training
use_val_in_train = False
# Number of test samples.
N_test = 7
# Seed of the random state.
random_state = 42
# Kernel regularization.
l1_reg = 0
l2_reg = 0

# CALLBACKS.
# Early stopping
es = {"enable":False,
      "delta": 0,
      "patience": 10,
      "restore_best_weight":True}
# Reduce learning rate
rl = {"enable":True,
      "factor": 0.99,
      "min_lr": 0.000000001,
      "patience": 5}
