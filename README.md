# KLT-Feature-Tracker-Acceleration-on-GPUs

This project accelerates the **Kanade–Lucas–Tomasi (KLT) Feature Tracker** using GPU computing. The goal is to profile the baseline CPU implementation, identify bottlenecks, and progressively optimize the application using **CUDA** and **OpenACC** while preserving correctness.

### Versions

* **V1 – CPU Baseline:**
  Original sequential implementation with profiling to identify hotspots.

* **V2 – Naive GPU:**
  Basic CUDA port of compute-intensive kernels.

* **V3 – Optimized GPU:**
  Performance-tuned CUDA implementation using improved launch configuration, occupancy, memory hierarchy, and reduced CPU–GPU communication.

* **V4 – OpenACC:**
  Directive-based GPU acceleration using OpenACC and comparison with CUDA implementations.

This repository includes complete source code and Makefiles to compile and run all versions on the GPU server. Moreover, reports for each deliverable are also included in this repository that document our work and results.

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
