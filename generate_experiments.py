#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster

SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/vaelib'
base_call = (f"python train.py --dataset cifar100 "
             f"--dataset_path {DATA_HOME}/data "
             f"--layers 28 --widen-factor 2 "
             f"--epochs 250 "
             f"--checkpoint_dir {DATA_HOME}/logs/ ")

repeats = 1

experiment = "cifar100"
dataset = [experiment]
learning_rate = [.01]
sloss = [True, False]
sloss_weight=[.1, .05, 0.01, .005, .001, 0.0001]

settings = [(lr, sloss_, sw_, dataset_, rep)
            for lr in learning_rate
            for sloss_ in sloss
            for sw_ in sloss_weight
            for dataset_ in dataset
            for rep in range(repeats)]

nr_expts = len(settings)

nr_servers = 20
avg_expt_time = 60*4  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experiment.txt", "w")

for (lr, sloss_, sw_, dataset_, rep) in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    if not sloss_ and sw_ != .1:
        continue

    expt_call = (
        f"{base_call} " +
        f"--lr {lr} " +
        f"--sloss_weight {sw_} " +
        (f"--no-sloss " if not sloss_ else "")
    )
    print(expt_call, file=output_file)

output_file.close()