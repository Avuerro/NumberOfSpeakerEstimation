import tensorflow as tf 
from model import crnn 
from src import data
from src import error_function
from src.data import DataSet
import wandb
from tqdm import tqdm
import numpy as np

import glob 
import datetime
import json
import pdb
# Parameters
DATA_DIR = '/vol/tensusers3/camghane/ASR/LibriSpeech_test_clean/data/test-clean/merged/train/*/*.wav'
BATCH_SIZE = 32
SAVED_MODEL_DIR = 'model-best-500.h5'

## loading the model

model = tf.keras.models.load_model(SAVED_MODEL_DIR, custom_objects={
    'class_mae': error_function.class_mae
})
### loading the data
filenames = glob.glob('/vol/tensusers3/camghane/ASR/LibriSpeech_test_clean/data/test-clean/merged/train/*/*.wav')
labels = list( map(lambda x: x.split("/")[-2] , filenames) )

test_dataset_object = DataSet(filenames, scale_data = True)
test_dataset = DataSet(filenames, scale_data = True).get_data()


#### Predicting
y_true = []
y_pred = []
for audio,label in tqdm(test_dataset.as_numpy_iterator()):
	predictions = model.predict_on_batch(audio)
	labels_converted = [int(np.argmax(x)) for x in label]
	predictions_converted = [int(np.argmax(x)) for x in predictions]
	y_true.extend(labels_converted)
	y_pred.extend(predictions_converted)

performance_dictionary = {'y_true': y_true, 'y_pred': y_pred}

with open('/vol/tensusers3/camghane/ASR/jdy/predictions_500.json', 'w') as filewriter:
    json.dump(performance_dictionary, filewriter)
