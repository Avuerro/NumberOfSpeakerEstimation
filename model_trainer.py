import tensorflow as tf
from model import crnn
from src import data
from src.data import DataSet
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
import glob
import datetime
import pickle

# Parameters #
DATA_DIR = '/vol/tensusers3/camghane/ASR/jdy/data/trainset750/merged/train/*/*.wav'
BATCH_SIZE = 32
SAVE_MODEL_DIR = '/vol/tensusers3/camghane/ASR/jdy/weights'
## Model ##
LEARNING_RATE = 0.001
### Training ###
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# callbacks TODO Move to separate file ?
## model callbacks
model_checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = SAVE_MODEL_DIR,
    save_weights_only = False, #for safety saving full model
    monitor = 'val_class_mae',
    mode = 'min',
    save_best_only = True
)
## learningrate callback
model_learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_class_mae',
    factor = 0.9,
    patience = 10,
    verbose = 1,
    mode = 'min'
)

model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_class_mae',
    min_delta=0.01,
    patience=10,
    mode='min'
)

# WANDB
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
## init wandb project
wandb.init(project='asr_speaker_estimation', entity='chickens4peace',config=tf.compat.v1.flags.FLAGS, sync_tensorboard=True, dir='/vol/tensusers3/camghane/ASR/run_metadata')

# init dataset
## create train and validation splits
filenames = glob.glob(DATA_DIR)
labels = list( map(lambda x: x.split("/")[-2] , filenames) )
X_train, X_val, y_train, y_val = train_test_split(filenames, labels, test_size=VALIDATION_SPLIT, random_state=42) 
## create dataset objects
train_dataset_object = DataSet(X_train, scale_data = True)
validation_dataset_object = DataSet(X_val, scale_data = True)
## get data
training_data = train_dataset_object.get_data()
validation_data = validation_dataset_object.get_data()


## init model
model = crnn.CRNN(LEARNING_RATE).get_model()
### start training
model.fit(training_data, 
          epochs = EPOCHS, 
          verbose = 2, 
          validation_data=validation_data,
          callbacks = [model_checkpoints_callback, model_early_stopping_callback, WandbCallback()])
