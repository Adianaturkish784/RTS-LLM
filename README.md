# RTS-LLM

Reprogramming Time Series with Large Language Models.

## Requirements

- Python >= 3.11 (recommended: MiniConda)
- CUDA compatible GPU

### Quick Install

```bash
# 1. Create and activate a conda environment
conda create -n RTS-LLM python=3.11
conda activate RTS-LLM

# 2. Install the project and its dependencies in editable mode
pip install -e .

# 3. If mpi4py is missing, install it via conda
conda install mpi4py
```

### Core Dependencies

See the `dependencies` field in [pyproject.toml](pyproject.toml) for details, which mainly include:

- `torch==2.2.2`
- `transformers==4.47.1`
- `accelerate==0.28.0`
- `deepspeed==0.14.0`
- `peft==0.4.0`

## Datasets

You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Quick Demos
1. Download datasets and place them under `./dataset`
2. Train the model. We provide training scripts under the folder `./scripts/train`. For example, you can run:

```bash
# ETTm1
bash ./scripts/train/ETTm1.sh

# Weather
bash ./scripts/train/Weather.sh
```


## Quick Replication

To quickly replicate our results on the ECL, ETTm1, and Weather datasets (prediction lengths: 96, 192, 336, 720), follow these steps:

### 1. Download Checkpoints
Download our pre-trained checkpoints from the Releases page:
- [[ECL Checkpoints]](https://github.com/Taihuachen-cfair/RTS-LLM/releases/download/v1.0.0/ECL.zip)
- [[ETTm1 Checkpoints]](https://github.com/Taihuachen-cfair/RTS-LLM/releases/download/v1.0.0/ETTm1.zip)
- [[Weather Checkpoints]](https://github.com/Taihuachen-cfair/RTS-LLM/releases/download/v1.0.0/Weather.zip)

Then extract them under the `./rts_ckpt` directory:
```bash
unzip ECL.zip -d ./rts_ckpt/
unzip ETTm1.zip -d ./rts_ckpt/
unzip Weather.zip -d ./rts_ckpt/
```

### 2. Download Datasets
Download the pre-processed datasets and place them under the `./dataset` directory (refer to the **Datasets** section above). Ensure the dataset paths are correct:
- ECL: `./dataset/electricity/electricity.csv`
- ETTm1: `./dataset/ETT-small/ETTm1.csv`
- Weather: `./dataset/weather/weather.csv`

### 3. Run Evaluation
Run the evaluation scripts:
```bash
# Evaluate ECL
bash ./scripts/eval/ECL.sh

# Evaluate ETTm1
bash ./scripts/eval/ETTm1.sh

# Evaluate Weather
bash ./scripts/eval/Weather.sh
```

## Detailed Usage

Please refer to `run_main.py`, `run_m4.py` for the detailed description of each hyperparameter.

## Acknowledgement

Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) ,  [Time-LLM](https://github.com/KimMeen/Time-LLM) and [OFA (GPT4TS)](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
