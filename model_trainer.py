import tensorflow as tf
from model import crnn
from src import data

import wandb
from wandb.keras import WandbCallback

import datetime
# Parameters #
DATA_DIR = '../'
BATCH_SIZE = 32
SAVE_MODEL_DIR='./weights'
## Model ##
LEARNING_RATE = 0.001
### Training ###
EPOCHS = 10
VALIDATION_SPLIT = 0.3

# callbacks TODO Move to separate file ?
## model callbacks
model_checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = SAVE_MODEL_DIR,
    save_weights_only = False, #for safety saving full model
    monitor = 'class_mae',
    mode = 'min',
    save_best_only = True
)
## learningrate callback
model_learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'class_mae',
    factor = 0.9,
    patience = 10,
    verbose = 1,
    mode = 'min'
)

# WANDB
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
## init wandb project
wandb.init(project='asr_speaker_estimation', entity='chickens4peace',config=tf.compat.v1.flags.FLAGS, sync_tensorboard=True)

# init dataset
training_data, validation_data = data.create_tensorflow_dataset(DATA_DIR, batch_size=32, val_split = VALIDATION_SPLIT)
## init model
model = crnn.CRNN(LEARNING_RATE).get_model()
### start training
model.fit(training_data, 
          epochs = EPOCHS, 
          verbose = 2, 
          validation_data=validation_data,
          callbacks = [model_checkpoints_callback, WandbCallback()])
