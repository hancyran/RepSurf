#!/bin/sh

mkdir -p log/PointAnalysis/log/ScanObjectNN
mkdir -p data/

conda create -n repsurf-cls python=3.7 -y
conda activate repsurf-cls

conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch -c conda-forge -y
conda install -c anaconda h5py -y

cd modules/pointops
python3 setup.py install
cd -
