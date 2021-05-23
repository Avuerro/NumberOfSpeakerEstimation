import tensorflow as tf 
from tensorflow.keras import backend as K 
import pdb

# copied from https://github.com/faroit/CountNet/blob/master/predict.py
def class_mae(y_true,y_pred):
    return K.mean(
        tf.cast( K.abs(
                K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
            ),dtype=tf.float32) ,
        axis=-1
    )