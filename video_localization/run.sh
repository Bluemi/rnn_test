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
	TF_CPP_MIN_LOG_LEVEL=3 python3 ./src/main.py train-conv-model $database --show
elif [ "$1" == "import" ]; then
	python3 ./src/main.py import-video $database "$2"
elif [ "$1" == "det" ]; then
	expdef="$2"
	if [ -z "$expdef" ]; then
		expdef="determined_experiments/hsearch.yaml"
	fi
	det experiment create "$expdef" ./src
elif [ "$1" == "run" ]; then
	python3 ./src/main.py run-model "$2"
elif [ "$1" == "eval" ]; then
	python3 ./src/main.py eval-model "$database" "$2"
else
	python3 ./src/main.py --help
	echo "no command defined"
fi
