#!/bin/bash

export PYTHONPATH=./

log_dir='pointtransformer_A5'

python3 tool/test_s3dis.py --log_dir ${log_dir} \
          --batch_size_test 12 \
          --gpu_id 0 \
          --model pointtransformer.pointtransformer \
          --test_area 5 \
          --filter