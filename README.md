# KLT-Feature-Tracker-Acceleration-on-GPUs

## Dataset Setup

This project uses our **PPM Image Dataset for KLT Feature Tracking**, hosted on Hugging Face, in addition to the dataset available in the /data directory in this repository.

To automatically download and extract both image sets:

```bash
cd src/V1
python download_dataset.py
```

## Makefile Usage

```bash
make clean
make lib
make gprof
make dot
make png
make pdf
```
