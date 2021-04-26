import tensorflow as tf 
import keras 
from keras import backend as k 


def create_tensorflow_dataset(data_dir, batch_size, data_format='channels_first',color_mode='grayscale', target_size=(500,201) ,datagenerator=None):
    if datagenerator is None:
        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator()
    

    dataset =   tf.keras.preprocessing.image.DirectoryIterator(data_dir, datagenerator,batch_size=batch_size, data_format=data_format, color_mode=color_mode, target_size=target_size)
    return dataset
