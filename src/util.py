import numpy as np
import scipy as scipy
import os
import sys
from scipy.io import wavfile
import soundfile as sf


def flacs_to_wavs(input_dir, output_dir):
    print(input_dir)
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".flac"):
                data, samplerate = sf.read(filepath)
                speaker_id,book_id,chapter = file.split("-")
                new_file_name =file.replace('.flac', '.wav')
                location_dir = os.path.join(output_dir,speaker_id)
                new_file_path = os.path.join(output_dir,speaker_id,new_file_name)
                if not os.path.exists(location_dir):
                    os.makedirs(location_dir)

                wavfile.write(new_file_path, samplerate, data)



def split_audio_in_samples(t = 5, new_dir = "./data/train100/",data_dir = "./data/wavs100/", a = -1, b = 1):
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".wav"):
                data, samplerate = sf.read(filepath)
                #data = a + ((data - data.min()) * (b - a))/(data.max() - data.min())
                previous_cut = 0
                filepath = filepath.replace(file, "")
                
                clean_filepath = filepath.replace("./data/wavs100/", "")
                #print(clean_filepath)
                if not os.path.exists(new_dir+clean_filepath):

                    os.makedirs(new_dir+clean_filepath)
                
                file = file.replace(".wav","")
                
                nr_of_splits = int(len(data)/(t*samplerate))

                if nr_of_splits == 0:
                    amount_of_padding = np.zeros(t*samplerate - len(data))


                    new_file = '{}_split_{}.wav'.format(file,nr_of_splits)
                    wavfile.write(new_dir+clean_filepath+new_file, samplerate, np.concatenate((data, amount_of_padding)))
                else:
                    for i in range(0,nr_of_splits):
                        
                        new_file = '{}_split_{}.wav'.format(file,i)
                        
                        split = data[previous_cut: previous_cut+t*samplerate]
                        previous_cut = t*samplerate + 1
                        wavfile.write(new_dir+clean_filepath+new_file, samplerate, split)