import tensorflow as tf 
import keras 
from keras import backend as k 
import pdb
import os


def parse_function(filename, label):
#     audio_sample, sample_rate = tf.audio.decode_wav(filename)    
    audio_binary = tf.read_file(filename)
    audio_decoded = tf.contrib.ffmpeg.decode_audio(
        audio_binary,
        file_format="wav",
        samples_per_second=16000,
        channel_count=1
    )
    return tf.cast(audio_decoded, tf.float32),label


def select_random_excerpt(file,label): #length can be changed..
    excerpt_duration=1
    excerpt_duration_frames = excerpt_duration * 16000
    limit = 80000 - excerpt_duration_frames
    start = np.random.randint(0,limit)
    end = start + excerpt_duration_frames
#     pdb.set_trace()
    return tf.slice(tf.squeeze(file), [start], [excerpt_duration_frames]),label


## TODO 
### add stft conversion
### put this into a class
filenames = glob.glob('../data/merged_outputtrain/*/*')
labels = list(map(obtain_label, filenames))
dataset = tf.data.Dataset.from_tensor_slices( (filenames,labels) )
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(select_random_excerpt, num_parallel_calls=4)
dataset = dataset.batch(32)
dataset = dataset.prefetch(1)




## NOTICE
## everything below this comment is most likely obsolete
class CustomDataIterator(tf.keras.preprocessing.image.DirectoryIterator):

    def __init__(self,*args,**kwargs):
        self.label_dir = kwargs.pop('label_dir')
        super(self.__class__, self).__init__(*args, **kwargs)
        self.used_indices = []

    def __getitem__(self,idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = []
        speakers_in_batch = []
        # pdb.set_trace()
        for index in self.index_array:
            
            if index in self.used_indices:
                continue
            
            if len(speakers_in_batch) == 32:
                break

            filename = self.filepaths[index].split("/")[-1].replace(".png",".txt")
            nr_of_speakers = filename.split("_")[0]
            label_location = os.path.join(self.label_dir,nr_of_speakers,filename)
            speakers_in_sample = []
            with open(label_location) as f:
                speakers_in_sample = [line.rstrip('\n') for line in f]
            if speakers_in_sample not in speakers_in_batch:
                speakers_in_batch.append(speakers_in_sample)
                index_array.append(index)
                self.used_indices.append(index)            

        return self._get_batches_of_transformed_samples(index_array)

def custom_preprocessing(sound_sample):
    print(sound_sample.shape)



def create_tensorflow_dataset(data_dir, batch_size, data_format='channels_first',color_mode='grayscale', target_size=(500,201) ,datagenerator=None, val_split = None):
    if datagenerator is None:
        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split = val_split, preprocessing_function=custom_preprocessing)
    # train_dataset = tf.keras.preprocessing.image.DirectoryIterator(data_dir, datagenerator,batch_size=batch_size, data_format=data_format, color_mode=color_mode, target_size=target_size, subset = 'training')
    # validation_dataset = tf.keras.preprocessing.image.DirectoryIterator(data_dir, datagenerator,batch_size=batch_size, data_format=data_format, color_mode=color_mode, target_size=target_size, subset = 'validation')
    train_dataset = CustomDataIterator(data_dir, 
                            datagenerator,
                            label_dir='../data/merged_outputmetadata/',
                            batch_size=batch_size, 
                            data_format=data_format, 
                            color_mode=color_mode, 
                            target_size=target_size, 
                            subset = 'training')
    validation_dataset = CustomDataIterator(data_dir, 
                            datagenerator,
                            label_dir='../data/merged_outputmetadata/',
                            batch_size=batch_size, 
                            data_format=data_format, 
                            color_mode=color_mode, 
                            target_size=target_size, 
                            subset = 'validation')
    return train_dataset, validation_dataset

