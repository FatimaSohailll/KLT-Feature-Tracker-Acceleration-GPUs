# KLT-Feature-Tracker-Acceleration-on-GPUs

## Dataset Setup

This project uses our **PPM Image Dataset for KLT Feature Tracking**, hosted on Hugging Face, in addition to the dataset available in the /data directory in this repository.

To automatically download and extract both image sets:

```bash
cd src/V2
python download_dataset.py
```

## Deliverable 1 Makefile Usage

```bash
cd src/V1
make clean
make lib
make gprof
make dot
make png
make pdf
```

## Deliverable 2 Makefile Usage

# For CPU-only Execution 
```bash
cd src/V2
make clean
make cpu
make run_cpu
make gprof
make dot
make png
make pdf
```

# For Naive GPU Execution 
```bash
cd src/V2
make clean
make gpu
make run_gpu
```
