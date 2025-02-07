# NumberOfSpeakerEstimation

The goal of this project is to estimate the number of speakers that appear in an audio fragment.
The first goal is to replicate the following study: [CountNet: Estimating the Number of Concurrent Speakers Using Supervised Learning](https://ieeexplore.ieee.org/document/8506601).
On top of that, two experiments are executed with this model:

 - Investigating the effect of the number of unique speakers in the dataset on this model
 - Investigating the effect of Multilingual data on the performance of the model and its ability to generalize to unseen data

## Structure of this repo

This repo contains the following folders:
 - In `model`, the code for the general model can be found.
 - In `pretrained_models`, some pretrained models of this project can be found. More specifically:
     - `model-best-baseline.h5` is the baseline model, trained completely on LibriSpeech-360 Clean.
     - `model-best-{250, 750, 750}.h5` are the models used for investigating the effect of the number of unique speakers in the dataset.
     - `multilingual-model-best.h5` is the model trained on the multilingual dataset and used to investigate whether it performs better than the baseline model.
 - In `src`, all python files containing code can be found.
 
 
Besides these folders, there are the following notebook:
  - `Creating Dataset.ipynb` demonstrates the full pipeline of creating the dataset used for training the baseline model.
  - `Experimental Datasets Unique Speakers.ipynb` demonstrates all code used for testing the effect of the number of unique speakers in the dataset on the performance of the model.
   - `Experimens with Multilingual Datasets.ipynb` demonstrates all code used for testing the effect of training the model on a multilingual dataset on the performance of the model.
   - `SaliencyMaps.ipynb` contains all code used to create the saliency maps for the models.

## How do I use this repo?

Evidently, running the notebooks is fairly straightforward.
If you want to train a model, make sure the correct path to the data is set in `model_trainer.py`.
After this, simply run the command `./run_model.sh train`
This will initiate a run on [Weights & Biases](wandb.ai).
After training, make sure to download the trained model from this specific run.

If you want to test the model that has been trained, make sure that the correct path to the test set is set in `model_test.py`.
Besides this, also make sure that the correct path to the pretrained model is set in `model_test.py`.
After making sure these things are set correctly, simply run the command: `./run_model.sh`.
