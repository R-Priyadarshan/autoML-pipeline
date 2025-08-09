# Project 3 — AutoML Pipeline for IoT Classification (TPOT)

## Overview
Generates a synthetic IoT dataset and runs TPOT to search for a strong classification pipeline. The best pipeline is exported to `best_pipeline.py`.

## Run
```bash
pip install -r requirements.txt
python generate_dataset.py
python automl_run.py
```
