# To install horovid on Ubuntu 18.04 with cuda10.1

## Step 1: NCCL

Download NCCL from https://developer.nvidia.com/nccl/nccl-download#a-collapse278-101

Install it as:<P>
sudo dpkg -i nccl-repo-<version>.deb <P>
sudo apt update<P>
sudo apt install libnccl2=2.4.8-1+cuda10.1 libnccl-dev=2.4.8-1+cuda10.1

for the detail, see https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian

## Step 2: Open MPI

gunzip -c openmpi-4.0.5.tar.gz | tar xf -  <P>
cd openmpi-4.0.5 <P>
./configure --prefix=/usr/local <P>
make all install<P>

## Step 3: horovod 

HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod

### Example: To run on 4GPUs, you can use the command 

horovodrun -np 4 -H localhost:4 python train.py
