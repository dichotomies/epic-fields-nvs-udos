
# EPIC Fields Benchmark: NVS and UDOS

## About

This repository provides the code to reproduce the results for Task 1 and Task 2 from *EPIC Fields: Marrying 3D Geometry and Video Understanding*. It provides details for
- [Setting up the required data](#Setup)
- [Evaluating Task 1 (NVS)](#task-1-nvs)
- [Evaluating Task 2 (UDOS)](#task-2-udos)

## Setup

Use the following commands to setup the folder structure. This will download and extract the
- Annotations
- Dataset split
- Cache (outputs from models)

Make sure to set `DIR_EK100` as it is linked in the commands. This directory contains the EpicKitchens-100 dataset. It should contain tar files and be in the form `EpicKitchens-100/{pid}/rgb_frames/{vid}.tar` where pid and vid represent the Person ID and Video ID respectively.

```
# go to main folder `benchmarking
cd benchmarking

# annotations
gdrive_download 15HMb8lo2B3D7Dnb52N4NaZrQv7JbC9Dr
tar -xzvf annotations.tar.gz
cd annotations
for x in *; do unzip $x; done
cd ../

# split
gdrive_download 14cFk1AY1-n5i38M0QUhxsBqonuMebydY
tar -xzvf split.tar.gz

# cache
gdrive_download TODO
tar -xzvf cache.tar.gz

# link EpicKitchens-100
ln -s $DIR_EK100 .
```

## Task 1: NVS

To calculate the results for the NVS Benchmark, use the following command:

```
python evaluate_nvs.py --dir_cache=cache/task1 \
	--dir_output=outputs/task1 \
	--model=neuraldiff --action_type=within_action
```

Select the model and action type you are interested in.

## Task 2: UDOS

To calculate the results for the UDOS Benchmark, use the following command:

```
python evaluate_udos.py --dir_cache=cache/task2 \
	--dir_output='outputs/task2/' \
	--model=neuraldiff --vid=P12_03
```

This will output segmentations for the three motion types. Evaluate further videos by selecting the corresponding video id (`vid`).

