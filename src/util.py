import numpy as np
import scipy as scipy
import os
import sys
from scipy.io import wavfile
import soundfile as sf


def flacs_to_wavs(new_dir = "./data/wavs/"):
    for subdir, dirs, files in os.walk("./data/LibriSpeech/"):

        for file in files:
            #print( os.path.join(subdir, file))
            filepath = subdir + os.sep + file

            if filepath.endswith(".flac"):
                data, samplerate = sf.read(filepath)

                filepath = filepath.replace(file, "")
                clean_filepath = filepath.replace("./data/LibriSpeech/dev-clean/", "")
                new_file = file.replace('.flac', '.wav')
                if not os.path.exists(new_dir+clean_filepath):
                    os.makedirs(new_dir+filepath)

                wavfile.write(new_dir+clean_filepath+new_file, samplerate, data)
