import numpy as np
import scipy as scipy
import os
import sys
from scipy.io import wavfile
import soundfile as sf


def flacs_to_wavs(new_dir = "./data/test_wavs/"):
    for subdir, dirs, files in os.walk("./data/test/"):
        for file in files:
            #print( os.path.join(subdir, file))
            filepath = subdir + os.sep + file

            if filepath.endswith(".flac"):
                data, samplerate = sf.read(filepath)

                filepath = filepath.replace(file, "")
                clean_filepath = filepath.replace("./data/test/dev-clean/", "")
                new_file = file.replace('.flac', '.wav')
                if not os.path.exists(new_dir+clean_filepath):
                    os.makedirs(new_dir+filepath)

                wavfile.write(new_dir+clean_filepath+new_file, samplerate, data)

def split_audio_in_samples(t = 5, new_dir = "./data/wav_splits/",data_dir = "./data/wavs/"):
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".wav"):
                data, samplerate = sf.read(filepath)
                
                previous_cut = 0
                filepath = filepath.replace(file, "")
                
                clean_filepath = filepath.replace("./data/wavs/", "")

                if not os.path.exists(new_dir+clean_filepath):
                    os.makedirs(new_dir+clean_filepath)
                
                file = file.replace(".wav","")
                
                for i in range(0,int(len(data)/(t*samplerate))):
                    
                    new_file = '{}_split_{}.wav'.format(file,i)
                    
                    split = data[previous_cut: previous_cut+t*samplerate]
                    previous_cut = t*samplerate + 1
                    wavfile.write(new_dir+clean_filepath+new_file, samplerate, split)