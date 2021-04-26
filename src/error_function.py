import tensorflow as tf 
import keras
from keras import backend as K 


# copied from https://github.com/faroit/CountNet/blob/master/predict.py
def class_mae(y_true,y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )