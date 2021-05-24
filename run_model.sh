#!/usr/bin/env bash

mode=$1

source '/vol/tensusers3/camghane/py37asr/bin/activate'
if [ "$mode" == "train" ]
then 
  python model_trainer.py
else
  python model_test.py
fi

