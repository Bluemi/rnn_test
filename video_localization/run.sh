#!/bin/bash

database='res/right_hand'

if [ "$1" == "test" ]; then
	python3 ./src/main.py test-camera
elif [ "$1" == "show" ]; then
	python3 ./src/main.py show-dataset $database
elif [ "$1" == "create" ]; then
	python3 ./src/main.py create-video-dataset $database
elif [ "$1" == "anno" ]; then
	python3 ./src/main.py annotate-dataset $database
elif [ "$1" == "train" ]; then
	python3 ./src/main.py train-conv-model $database --show
elif [ "$1" == "import" ]; then
	python3 ./src/main.py import-video $database "$2"
fi
