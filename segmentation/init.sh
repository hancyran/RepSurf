#!/bin/sh

mkdir -p log/PointAnalysis/log/S3DIS
mkdir -p log/PointAnalysis/log/ScanNet
mkdir -p data/S3DIS
mkdir -p data/ScanNet

conda create -n repsurf-seg python=3.7 -y
conda activate repsurf-seg

#conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch -c conda-forge -y
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx -y

cd modules/pointops
python3 setup.py install
cd -
