import numpy as np
import scipy as scipy
import math
import os
import sys
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import librosa

def flacs_to_wavs(data_dir = "./data/LibriSpeech/", new_dir = "./data/wavs100/"):

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    for subdir, dirs, files in tqdm(os.walk(data_dir)):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".flac"):
                data, samplerate = sf.read(filepath)
 
                filepath = filepath.replace(file, "")
                clean_filepath = filepath.replace(data_dir, "")
                
                new_file = file.replace('.flac', '.wav')
                if not os.path.exists(new_dir+clean_filepath):
                    os.makedirs(new_dir+clean_filepath)

                wavfile.write(new_dir+clean_filepath+new_file, samplerate, data)

def pad_audio_file(audio_file, sample_rate, total_time):
    amount_of_padding = np.zeros(total_time*sample_rate - len(audio_file))
    return np.concatenate( (audio_file, amount_of_padding) )

# OLD FUNCTION ::
# def split_audio_in_samples(data_dir = "./data/wavs100/", new_dir = "./data/splits100/", t = 5):

#     for subdir, dirs, files in tqdm(os.walk(data_dir)):
#         for file in files:
#             filepath = subdir + os.sep + file

#             if filepath.endswith(".wav"):
#                 data, samplerate = sf.read(filepath)
#                 previous_cut = 0
#                 filepath = filepath.replace(file, "")
                
#                 clean_filepath = filepath.replace(data_dir, "")
                
#                 if not os.path.exists(new_dir+clean_filepath):

#                     os.makedirs(new_dir+clean_filepath)
                
#                 file = file.replace(".wav","")
                
#                 nr_of_splits = int(len(data)/(t*samplerate))

#                 if nr_of_splits == 0:
#                     padded_audio_file = pad_audio_file(data, samplerate, t)
#                     new_file = '{}_split_{}.wav'.format(file,nr_of_splits)
#                     wavfile.write(new_dir+clean_filepath+new_file, samplerate, padded_audio_file)
#                 else:
#                     for i in range(0,nr_of_splits):
                        
#                         new_file = '{}_split_{}.wav'.format(file,i)
                        
#                         split = data[previous_cut: previous_cut+t*samplerate]
#                         previous_cut = t*samplerate + 1
#                         wavfile.write(new_dir+clean_filepath+new_file, samplerate, split)
# OLD FUNCTION ^^

def create_audio_splits(data_dir = "./data/wavs100/", new_dir = "./data/splits100/", t = 10):
    nr_of_files = 0
    files_per_speaker = []

    # First generate a list of all speaker files
    folder = os.listdir(data_dir)[0]
    splits_dir = data_dir + folder
    speakers = os.listdir(splits_dir)
    speakers_done = []

    for speaker in speakers:
        speaker_files = []
        for subdir, dirs, files in os.walk(splits_dir+"/{}".format(speaker)):
            for f in files:
                if f.endswith('.wav'):
                    speaker_files.append("{}/{}".format(subdir, f))
            if len(speaker_files) > 0 and speaker not in speakers_done:
                nr_of_files += len(speaker_files)
                speakers_done.append(speaker)
                files_per_speaker.append(speaker_files)
    print("{} files found in total".format(nr_of_files))
    files_per_speaker = np.array(files_per_speaker)

    for speaker in files_per_speaker:
        complete_audio = []
        while len(speaker) > 0:
            filepath = speaker.pop(0)
            speaker_id = filepath.split('/')[4]
            
            data, sr = sf.read(filepath)
            #print(len(data))
            complete_audio.extend(data)
        #print(len(complete_audio))
        # amount of splits to make:
        nr_of_splits = math.ceil(len(complete_audio)/(t*sr))
        
        audio_splits = np.array_split(complete_audio, nr_of_splits)
        
        i = 0
        for split in audio_splits:
            if len(split) < (t*sr):
                
                to_pad = (t*sr) - len(split)
                padding = split[0:to_pad]
                
                split = np.concatenate( (split, np.array(padding)) )
            new_file = '{}_split_{}.wav'.format(speaker_id,i)
            clean_filepath = "train-clean-100/{}/".format(speaker_id)
            if not os.path.exists(new_dir+clean_filepath):
                    os.makedirs(new_dir+clean_filepath)
            wavfile.write(new_dir+clean_filepath+new_file, sr, split)
            i += 1



def compute_loudness(audio, sr = 16000):
    
    return 10*np.log10(np.sum(audio*audio) / (audio.shape[0] * 4 * (10**-10)) )

def change_loudness(data_dir = "./data/splits100/", new_dir = "./data/normalized_splits100/", target = 70.):

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for subdir, dirs, files in tqdm(os.walk(data_dir)):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".wav"):
                data, samplerate = sf.read(filepath)
                loudness = compute_loudness(data, samplerate)
                scaling = target - loudness

                clean_filepath = filepath.replace(file, "")
                
                clean_filepath = clean_filepath.replace(data_dir, "")
                
                if not os.path.exists(new_dir+clean_filepath):
                    os.makedirs(new_dir+clean_filepath)

                subprocess.call('ffmpeg -i {} -filter:a "volume={}dB" {}'.format(filepath, scaling, new_dir+clean_filepath+file), shell=True)


def min_max_normalization(data, a = -1, b = 1):
    data = a + ((data - data.min()) * (b - a))/(data.max() - data.min())
    return data

def merge_audiofiles(data_dir = './data/train100/', new_dir = "./data/trainset/", max_nr_of_speakers = 10, a =-1, b = 1):

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    if not os.path.exists(new_dir+"metadata"):
        os.makedirs(new_dir+"metadata")

    nr_of_files = 0
    files_per_speaker = []

    # First generate a list of all speaker files
    folder = os.listdir(data_dir)[0]
    splits_dir = data_dir + folder
    speakers = os.listdir(splits_dir)
    speakers_done = []

    for speaker in speakers:
        speaker_files = []
        for subdir, dirs, files in os.walk(splits_dir+"/{}".format(speaker)):
            for f in files:
                if f.endswith('.wav'):
                    #nr_of_files +=1
                    speaker_files.append("{}/{}".format(subdir, f))


            if len(speaker_files) > 0 and speaker not in speakers_done:
                nr_of_files += len(speaker_files)
                speakers_done.append(speaker)
                files_per_speaker.append(speaker_files)
    print("{} files found in total".format(nr_of_files))
    files_per_speaker = np.array(files_per_speaker)
    
    print("Now starting to merge files...")
    i = 1
    amount_of_datapoints = 0
    progress_bar = tqdm(total = amount_of_datapoints) 
    
    while(files_per_speaker.shape[0] > 0):
        # Calculate how many speakers should be merged
        amount_of_speakers = i % (max_nr_of_speakers + 1)
        
        # Check how many speakers are left
        number_of_rows = files_per_speaker.shape[0]

        # When the amount of speakers is larger than nr of rows, decrease amount of speakers until we can sample again
        while( amount_of_speakers > number_of_rows):
            amount_of_speakers -= 1

        # Sample Random Speaker IDs
        random_speaker_ids = np.random.choice(number_of_rows, size=amount_of_speakers, replace=False)
        
        ids_to_remove = []
        files_to_merge = []
        real_ids = []

        for speaker_id in random_speaker_ids:
            # Load all files for this speaker:
            speaker_files = files_per_speaker[speaker_id]
            # For each random speaker, pick one random file:
            random_file = np.random.choice(speaker_files)

            real_ids.append(random_file.split('/')[4])

            files_to_merge.append(random_file)
            # Remove file from original set, to prevent duplicates among merged files
            files_per_speaker[speaker_id] = np.delete(files_per_speaker[speaker_id], np.where(files_per_speaker[speaker_id] == random_file)[0])
            # If all files from a single speaker are used: remove the speakers, to prevent sampling from empty lists
            if len(files_per_speaker[speaker_id]) == 0:
                ids_to_remove.append(speaker_id)
        
        # Now start the actual merging
        data = np.zeros(80000)

        for audio_file in files_to_merge:
            # Load file
            sample, samplerate = sf.read(audio_file)

            # make sure length is 160000
            if sample.shape[0] != 160000:
                to_append = 160000 - sample.shape[0]
                zeros = np.zeros(to_append)
                sample = np.concatenate((sample, zeros))
            
            # Add to data
            data += sample
        
        # Min Max Normalisation before saving as .wav to prevent clipping, if amount of speakers is not zero
        if amount_of_speakers > 0:
            data = min_max_normalization(data)

        # Write to wav:
        if not os.path.exists("{}train/{}".format(new_dir, amount_of_speakers)):
            os.makedirs("{}train/{}".format(new_dir, amount_of_speakers))

        wavfile.write("{}train/{}/{}_{}.wav".format(new_dir, amount_of_speakers, amount_of_speakers, amount_of_datapoints), samplerate, data)

        # Write metadata:
        if not os.path.exists("{}metadata/{}".format(new_dir, amount_of_speakers)):
            os.makedirs("{}metadata/{}".format(new_dir, amount_of_speakers))

        with open("{}metadata/{}/{}_{}.txt".format(new_dir, amount_of_speakers, amount_of_speakers, amount_of_datapoints), 'w') as filehandle:
            filehandle.writelines("%s\n" % sid for sid in real_ids)

        files_per_speaker = np.delete(files_per_speaker, ids_to_remove)
        i += 1
        amount_of_datapoints +=1
        progress_bar.update(len(ids_to_remove)) 
    progress_bar.close()
    print("Created {} unique datapoints".format(amount_of_datapoints))



def stft(data,n_fft=400,hop_length=160,window=signal.windows.hann):
    return np.abs(librosa.stft(data, n_fft=n_fft,hop_length=hop_length,window=window)).T

def create_spectrograms(input_dir, spectro_output_dir, n_fft=400,hop_length=160,window=signal.windows.hann):
    for dirpath,dirname,files in tqdm(os.walk(input_dir)):
        if len(files) > 0:
            speaker_id = dirpath.split("/")[-1]
            output_dir = os.path.join(spectro_output_dir, speaker_id)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for merged_file in tqdm(files, position=0):
                audio_file_location = os.path.join(dirpath,merged_file)

                file_name = merged_file.replace(".wav","") + '.png'

                new_file_name = os.path.join(output_dir,file_name)
                sample,sr = sf.read(audio_file_location)
                X_stft = np.flipud(stft(sample, n_fft=400,hop_length=160,window=signal.windows.hann).T)
                

                plt.imsave(new_file_name, X_stft, cmap = 'hot', vmin=0,vmax=10)
