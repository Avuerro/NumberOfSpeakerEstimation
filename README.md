# NumberOfSpeakerEstimation

The goal of this project is to estimate the number of speakers that appear in an audio fragment.
The first goal is to replicate the following study: [CountNet: Estimating the Number of Concurrent Speakers Using Supervised Learning](https://ieeexplore.ieee.org/document/8506601).
On top of that, two experiments are executed with this model:

 - Investigating the effect of the number of unique speakers in the dataset on this model
 - TODO


## Structure of this repo

This repo contains the following folders:
 - In `model`, the code for the general model can be found.
 - In `pretrained models`, some pretrained models of this project can be found. More specifically:
     - `model-best-baseline.h5` is the baseline model, trained completely on LibriSpeech-360 Clean.
     - `model-best-{250, 750, 750}.h5` are the models used for investigating the effect of the number of unique speakers in the dataset.
     - TODO
 - In `samples`, some sample audio files can be found. This folder will be updated with new samples as the project progresses.
 - In `src`, all python files containing code can be found.
 
 
Besides these folders, there are the following notebook:
  - `Creating Dataset.ipynb` demonstrates the full pipeline of creating the dataset used for training the baseline model.
  - `Experimental Datasets Unique Speakers.ipynb` demonstrates all code used for testing the effect of the number of unique speakers in the dataset on the performance of the model.
  - TODO: dataloader.ipynb?
  

## How do I use this repo?

Evidently, running the notebooks is fairly straightforward.
If you want to train a model, make sure the correct path to the data is set in `model_trainer.py`.
After this, simply run the command `./run_model.sh train`
This will initiate a run on [Weights & Biases](wandb.ai).
After training, make sure to download the trained model from this specific run.

If you want to test the model that has been trained, make sure that the correct path to the test set is set in `model_test.py`.
Besides this, also make sure that the correct path to the pretrained model is set in `model_test.py`.
After making sure these things are set correctly, simply run the command: `./run_model.sh'.