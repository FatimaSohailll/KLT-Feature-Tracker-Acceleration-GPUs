# KLT-Feature-Tracker-Acceleration-on-GPUs

## Dataset Setup

This project uses our **PPM Image Dataset for KLT Feature Tracking**, hosted on Hugging Face, in addition to the dataset available in the /data directory in this repository.

To automatically download and extract 3 image sets:

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

### For CPU-only Execution 
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

### For Naive GPU Execution 
```bash
cd src/V2
make clean
make gpu
make run_gpu
```

## Deliverable 3 Makefile Usage

### For CPU-only Execution 
```bash
cd src/V3
make clean
make cpu
make run_cpu <datasetName> <Features> <Frames>
make gprof
make dot
make png
make pdf
```
### For Optimised GPU Execution 
```bash
cd src/V3
make clean
make gpu
make run_gpu <datasetName> <Features> <Frames>
```

## Deliverable 4 Makefile Usage

### For CPU-only Execution 
```bash
cd src/V4
make clean
make run_cpu
make run_cpu <datasetName> <Features> <Frames>
```

### For OpenAcc Execution 
```bash
cd src/V4
make clean
make run_gpu
make run_gpu <datasetName> <Features> <Frames>
make nsys <datasetName> <Features> <Frames>
```
