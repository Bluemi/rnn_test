FROM docker.io/determinedai/environments:cuda-10.0-pytorch-1.4-tf-1.14-gpu-0c9e956

RUN pip install opencv-python & apt update && apt install -y libsm6 libxext6 libxrender-dev

ADD requirements.txt /root/requirements.txt

RUN pip install -r /root/requirements.txt
