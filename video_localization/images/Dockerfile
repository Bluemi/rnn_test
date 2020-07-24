FROM docker.io/determinedai/environments:cuda-10.1-pytorch-1.4-tf-2.2-gpu-0.5.0

RUN pip install opencv-python & apt update && apt install -y libsm6 libxext6 libxrender-dev

ADD requirements.txt /root/requirements.txt

RUN pip install -r /root/requirements.txt
