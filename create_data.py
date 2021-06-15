import sys
from src import util
import os 

DATASETS=['mls_dutch', 'mls_german', 'mls_french']
BASE_DATA_DIR = '/vol/tensusers3/camghane/ASR/MLS'


for dataset in DATASETS:
	data_dir = os.path.join(BASE_DATA_DIR,dataset)

	#converting to wavs
	print("converting to wavs")
	in_dir = os.path.join(data_dir,'train','audio')
	wav_data = os.path.join(data_dir, 'data','wav')
	util.flacs_to_wavs(in_dir, wav_data+'/')


	#splitting
	print("splitting")
	split_data = os.path.join(data_dir, 'data','split')
	t=10
	util.create_audio_splits(wav_data, split_data+'/', t)

	#loudness
	print("normalizing")
	normalized_data = os.path.join(data_dir, 'data','normalized')
	target = 70.
	util.change_loudness(split_data, normalized_data +'/', target)

	#merging
	print("merging data")
	merged_data = os.path.join(data_dir, 'data', 'merged')
	max_nr_of_speakers = 10
	util.merge_audiofiles(normalized_data, merged_data +'/', max_nr_of_speakers)
