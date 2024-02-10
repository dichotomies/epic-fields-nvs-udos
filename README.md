
# EPIC Fields Benchmark: NVS and UDOS

## About

This repository provides the code to reproduce the results for Task 1 and Task 2 from *EPIC Fields: Marrying 3D Geometry and Video Understanding*. It provides details for
- [Setting up the required data](#setup)
- [Evaluating Task 1 (NVS)](#task-1-nvs)
- [Evaluating Task 2 (UDOS)](#task-2-udos)

## Updates

10.2.24: We updated the annotations. The previous annotations of the first arxiv release can be found [here](https://drive.google.com/file/d/1dzP4ECsLNnh721gaPbogSw3OizAcCy_l/view?usp=sharing). The corresponding git commit for processing the previous annotations can be accessed [here](https://github.com/dichotomies/epic-fields-nvs-udos/tree/24d8ba048f0f13b5b3de4cf5d5f3bd2c4505a5d3).

## Setup

Use the following commands to setup the folder structure. This will download and extract the
- Annotations
- Dataset split
- Cache (outputs from models)

Make sure to set `DIR_EK100` as it is linked in the commands. This directory contains the [EPIC-KITCHENS-100 dataset](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m). It should contain tar files and be in the form `EpicKitchens-100/{pid}/rgb_frames/{vid}.tar` where pid and vid represent the Person ID and Video ID respectively.

If you don't have a google account or haven't installed gdrive for downloading files from google drive via CLI, then download the files manually with the provided links (they are publicly accessible).

```
# annotations
# https://drive.google.com/file/d/1F0-jIYhY_hx4rr5oV3Z8o2V9gaBAIcuz/view?usp=sharing
gdrive download 1F0-jIYhY_hx4rr5oV3Z8o2V9gaBAIcuz
tar -xzvf annotations.tar.gz
cd annotations
for x in *; do unzip $x; done
cd ../

# split
# https://drive.google.com/file/d/1af9crxPDR-5JuBK3oX6U8sRSa2VA9Oq1/view?usp=sharing
gdrive download 1af9crxPDR-5JuBK3oX6U8sRSa2VA9Oq1
tar -xzvf split.tar.gz

# cache
# https://drive.google.com/file/d/1H0T7Qx0Ab4KEDo24CD83DdCip__ssVzS/view?usp=sharing
gdrive download 1H0T7Qx0Ab4KEDo24CD83DdCip__ssVzS
tar -xvf cache.tar

# link EpicKitchens-100
ln -s $DIR_EK100 ek100
```

## Task 1: NVS

To calculate the results for the NVS Benchmark, use the following command:

```
python evaluate_nvs.py --dir_cache=cache/task1 \
	--dir_output=outputs/task1 \
	--model=neuraldiff --action_type=within_action
```

Select the model and action type you are interested in.

Alternatively, calculate the results automatically with `sh evaluate_nvs.sh`.

## Task 2: UDOS

To calculate the results for the UDOS Benchmark, use the following command:

```
python evaluate_udos.py --dir_cache=cache/task2 \
	--dir_output='outputs/task2/' \
	--model=neuraldiff --vid=P12_03
```

This will output segmentations for the three motion types. Evaluate further videos by selecting the corresponding video id (`vid`).

Alternatively, calculate the results automatically with `sh evaluate_udos.sh`.

## Summarising the Results for Both Tasks

To create a summary of the results, use `python summarise.py`.
