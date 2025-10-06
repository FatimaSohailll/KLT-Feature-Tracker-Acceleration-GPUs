# KLT-Feature-Tracker-Acceleration-on-GPUs

## Dataset Setup

This project uses the **PPM Image Dataset for KLT Feature Tracking**, hosted on Hugging Face.

To automatically download and extract both image sets:

```bash
cd src
python download_dataset.py
```

## Makefile Usage

```bash
make clean
make lib
make gprof example3
make dot example3
make png example3
make pdf example3
```
