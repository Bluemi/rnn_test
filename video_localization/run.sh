#!/bin/bash

# python3 ./src/main.py test-camera
# python3 ./src/main.py create-video-dataset 'res'
# python3 ./src/main.py show-dataset 'res'
# python3 ./src/main.py annotate-dataset 'res'
python3 ./src/main.py train-conv-model 'res'
