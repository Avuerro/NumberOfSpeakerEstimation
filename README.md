# NumberOfSpeakerEstimation

The goal of this project is to estimate the number of speakers that appear in an audio fragment.
Initial inspiration for this project can be found in the paper [CountNet: Estimating the Number of Concurrent Speakers Using Supervised Learning](https://ieeexplore.ieee.org/document/8506601).

## Structure of this repo

This repo is structured as follows:
 - In `samples`, some sample audio files can be found. This folder will be updated with new samples as the project progresses.
  - In `src`, all python files containing code can be found.
  - The Jupyter Notebook `Number of Speaker Estimation ` shows how all files in `src` can be used, as to showcase a pipeline of this repo.

In order to get the code working right off the bat, it is required to have the [LibriSpeech Clean Development set](https://www.openslr.org/12) downloaded, and to unwrap this data in a folder called `data` located at the root of this repo. Lastly, the code requires that the data is in `.wav` format. Conversion from the original `.flac` files to `.wav` is done by calling the function `flacs_to_wavs` in `src\utils.py`.