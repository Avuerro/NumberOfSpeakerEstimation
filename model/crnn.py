import tensorflow as tf
from tensorflow.keras import Sequential,layers
# from keras import backend as K

from src import error_function


class CRNN(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get_model(self):
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=None)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-8)
        model = Sequential([
                    layers.ZeroPadding2D(padding=((0,0),(0,0)), batch_input_shape=(None, 1,500,201),name="zero1",data_format="channels_last"),
                    layers.Conv2D(64,kernel_size=(3,3),strides=(1,1),name="conv1",activation="relu",kernel_initializer=initializer,data_format="channels_first"),
                    layers.Conv2D(32,kernel_size=(3,3),strides=(1,1),name="conv2",activation="relu",kernel_initializer=initializer,data_format="channels_first"),
                    layers.MaxPool2D(pool_size=(3,3),strides=(3,3),trainable=True,name="pool1",data_format="channels_first"),
                    layers.Conv2D(128,kernel_size=(3,3),strides=(1,1),name="conv3",activation="relu",kernel_initializer=initializer,data_format="channels_first"),
                    layers.Conv2D(64,kernel_size=(3,3),strides=(1,1),name="conv4",activation="relu",kernel_initializer=initializer, data_format="channels_first"),
                    layers.MaxPool2D(pool_size=(3,3),strides=(3,3),trainable=True,name="pool2",data_format="channels_first"),
                    layers.Dropout(0.5, trainable=True, name="dropout_1"),
                    layers.Permute((2,1,3),trainable=True,name="permute_1"),
                    layers.Reshape((53,-1),trainable=True,name="reshape_1"),
                    layers.LSTM(40, return_sequences=True,kernel_initializer=initializer,name="lstm_1",batch_input_shape=(None, None, 1280)),
                    layers.MaxPool1D(pool_size=(2,),strides=(2,),trainable=True, name="maxpooling1d_1", data_format="channels_last"),
                    layers.Flatten(name="flatten1",trainable=True),
                    layers.Dense(11,activation="linear",kernel_initializer=initializer,trainable=True,name="dense_1", batch_input_shape=(None, 1040)),
                    layers.Softmax(trainable=True)
                ])
        model.compile(optimizer=optimizer,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy', {'class_mae':error_function.class_mae}]
             )

        return model