#!/bin/sh

mkdir -p log/PointAnalysis/log/
mkdir -p data/

conda create -n repsurf python=3.7 -y
conda activate repsurf

conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch -c conda-forge -y
conda install -c anaconda h5py -y

cd modules/pointops
python3 setup.py install
cd -
