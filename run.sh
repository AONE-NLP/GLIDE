#!/usr/bin/env bash

# Example GLIDE training commands.
# Make sure the corresponding processed files exist under dataset/<DATASET>/.

python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000
python app.py --dataset COVID19 --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000
python app.py --dataset Citibike --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000
python app.py --dataset Crime --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000

