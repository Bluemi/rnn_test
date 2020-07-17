#!/bin/bash

database='res/right_hand'

if [ "$1" == "test" ]; then
	python3 ./hand_tracker/main.py test-camera
elif [ "$1" == "show" ]; then
	python3 ./hand_tracker/main.py show-dataset $database
elif [ "$1" == "create" ]; then
	python3 ./hand_tracker/main.py create-video-dataset $database
elif [ "$1" == "anno" ]; then
	python3 ./hand_tracker/main.py annotate-dataset $database
elif [ "$1" == "train" ]; then
	TF_CPP_MIN_LOG_LEVEL=3 python3 ./hand_tracker/main.py train-conv-model $database --show
elif [ "$1" == "import" ]; then
	python3 ./hand_tracker/main.py import-video $database "$2"
else
	det experiment create determined_experiments/const.yaml ./src
fi
