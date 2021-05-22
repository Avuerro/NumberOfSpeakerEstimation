import tensorflow as tf 
from tensorflow.keras import backend as K 
import pdb

# copied from https://github.com/faroit/CountNet/blob/master/predict.py
def class_mae(y_true,y_pred):
#     tf.print( K.argmax(y_pred, axis=-1))
#     tf.print( K.argmax(y_true, axis=-1))
#     tf.print('----')
#     tf.print( K.argmax(y_pred, axis=-1) -  K.argmax(y_true, axis=-1) )
#     tf.print('----')
#     tf.print( K.abs(  K.argmax(y_pred, axis=-1) -  K.argmax(y_true, axis=-1) ) )
#     tf.print('-----')
#     tf.print( K.mean( tf.cast( K.abs(  K.argmax(y_pred, axis=-1) -  K.argmax(y_true, axis=-1) ) ,dtype=tf.float32) ) )
#     tf.print( tf.reduce_mean( K.abs(  K.argmax(y_pred, axis=-1) -  K.argmax(y_true, axis=-1) ) )  ) 
    return K.mean(
        tf.cast( K.abs(
                K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
            ),dtype=tf.float32) ,
        axis=-1
    )