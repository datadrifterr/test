import os
import subprocess

# -----------------------------------------------------------------
# Your configurable settings
# -----------------------------------------------------------------

# WANDB settings
ENABLE_WANDB=False
WANDB_PREFIX="RWKV-v5-Finetune"
WANDB_PROJECT="RWKV-v5-Finetune"

# Project directory offset (you need to modify if, you move the notebook into another dir)
PROJECT_DIR_OFFSET="../../"

# Config dir (relative to the notebook, excluding ending slash)
# to use, with the config filename
CONFIG_FILE_DIR="."
CONFIG_FILE_NAME="Eagle-x-openhermes1-instruct"

# The model to use
MODEL_NAME="RWKV-5-World-0.1B-v1-20230803-ctx4096.pth"
MODEL_URL="https://huggingface.co/BlinkDL/rwkv-5-world/resolve/main/RWKV-5-World-0.1B-v1-20230803-ctx4096.pth?download=true"

# GPU count to use
GPU_DEVICES="auto"

# -----------------------------------------------------------------
# Lets detect the GPU vram sizes, and suggest a resonable default
# based on the detected VRAM sizes
# -----------------------------------------------------------------

# Default settings
# NOTE: If your not using cuda, you may want to manually change this around
DEEPSPEED_STRAT="deepspeed_stage_2"
TRAINING_CTX_LEN=2048
MICROBATCH_SIZE=1

import torch
if torch.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    print("No CUDA compatible GPU found, using default settings")
else:
    # -----------------------------------------------------------------
    # Auto select the strategy based on the detected VRAM size
    # -----------------------------------------------------------------

    GPU_COUNT=torch.cuda.device_count()
    GPU_0_VRAM_SIZE_GB=torch.cuda.get_device_properties(0).total_memory / 1024**3
    if GPU_DEVICES != "auto":
        GPU_COUNT=int(GPU_DEVICES)
    print("GPU_COUNT:", GPU_COUNT)
    print("GPU_0_VRAM_SIZE (GB):", GPU_0_VRAM_SIZE_GB)

    if GPU_0_VRAM_SIZE_GB < 17:
        # This takes about 17.5GB vram on a single GPU
        # We DO NOT recommend training with ctx_len=128, as the training
        # quality will degrade noticably. But it will work!
        DEEPSPEED_STRAT="deepspeed_stage_2"
        TRAINING_CTX_LEN=4096
        MICROBATCH_SIZE=2
    elif GPU_0_VRAM_SIZE_GB < 23:
        # This takes about 17.5GB vram on a single GPU
        # We DO NOT recommend training with ctx_len=128, as the training
        # quality will degrade noticably. But it will work!
        DEEPSPEED_STRAT="deepspeed_stage_2_offload"
        TRAINING_CTX_LEN=128
        MICROBATCH_SIZE=1
    elif GPU_0_VRAM_SIZE_GB < 25:
        # This takes about 21GB vram on a single GPU
        DEEPSPEED_STRAT="deepspeed_stage_2_offload"
        TRAINING_CTX_LEN=2048
        MICROBATCH_SIZE=2
    elif GPU_0_VRAM_SIZE_GB < 78:
        # This takes about 23GB vram on a single GPU
        DEEPSPEED_STRAT="deepspeed_stage_2"
        TRAINING_CTX_LEN=4096
        MICROBATCH_SIZE=2
        if GPU_COUNT >= 8:
            MICROBATCH_SIZE=4
    else:
        # This is now the 80GB vram class
        DEEPSPEED_STRAT="deepspeed_stage_2"
        TRAINING_CTX_LEN=4096
        MICROBATCH_SIZE=4
        if GPU_COUNT >= 8:
            MICROBATCH_SIZE=8

# -----------------------------------------------------------------
# # Training settings you can use to override the "auto" default above
# -----------------------------------------------------------------
# DEEPSPEED_STRAT="deepspeed_stage_1"
# TRAINING_CTX_LEN=4096
# MICROBATCH_SIZE=8

# ---
print("ENABLE_WANDB:", ENABLE_WANDB)
print("GPU_DEVICES:", GPU_DEVICES)
print("DEEPSPEED_STRAT:", DEEPSPEED_STRAT)
print("TRAINING_CTX_LEN:", TRAINING_CTX_LEN)
if ENABLE_WANDB:
    WANDB_MODE="online"
else:
    WANDB_MODE="disabled"

# Computing the notebook, and various paths
import os
NOTEBOOK_DIR=os.path.dirname(os.path.abspath("__file__"))
PROJECT_DIR=os.path.abspath(os.path.join(NOTEBOOK_DIR, PROJECT_DIR_OFFSET))
TRAINER_DIR=os.path.abspath(os.path.join(PROJECT_DIR, "./RWKV-v5/"))
print("NOTEBOOK_DIR:", NOTEBOOK_DIR)
print("TRAINER_DIR:", TRAINER_DIR)
print("PROJECT_DIR:", PROJECT_DIR)

# Check if the directory exists
if not os.path.exists(TRAINER_DIR):
    raise Exception("The trainer directory does not exists. Did you move the notebook?")

os.chdir(PROJECT_DIR)
os.makedirs("./model", exist_ok=True)

model_path = os.path.join(PROJECT_DIR, "model", MODEL_NAME)
if not os.path.exists(model_path):
    # Download the model
    subprocess.run(["wget", "-nc", MODEL_URL, "-O", model_path], check=True)

# Preload the required dataset
os.chdir(TRAINER_DIR)
subprocess.run(["python3", "preload_datapath.py", f"{NOTEBOOK_DIR}/{CONFIG_FILE_DIR}/{CONFIG_FILE_NAME}.yaml"], check=True)

# Setup the checkpoint dir
os.chdir(PROJECT_DIR)
os.makedirs(f"./checkpoint/{CONFIG_FILE_NAME}/", exist_ok=True)

# Start the training
os.chdir(TRAINER_DIR)
os.environ["WANDB_MODE"] = WANDB_MODE
subprocess.run([
    "python3", "lightning_trainer.py", "fit",
    "-c", f"{NOTEBOOK_DIR}/{CONFIG_FILE_DIR}/{CONFIG_FILE_NAME}.yaml",
    "--model.load_model", "../model/{MODEL_NAME}",
    "--data.skip_datapath_setup", "True",
    "--trainer.callbacks.init_args.dirpath", f"../checkpoint/{CONFIG_FILE_NAME}/",
    "--trainer.logger.init_args.name", f"{WANDB_PREFIX} - {CONFIG_FILE_NAME} (tctxlen={TRAINING_CTX_LEN}, {DEEPSPEED_STRAT})",
    "--trainer.logger.init_args.project", f"{WANDB_PROJECT}",
    "--trainer.strategy", f"{DEEPSPEED_STRAT}",
    "--trainer.target_batch_size", "64",
    "--trainer.microbatch_size", f"{MICROBATCH_SIZE}",
    "--model.ctx_len", f"{TRAINING_CTX_LEN}",
    "--trainer.devices", f"{GPU_DEVICES}"
], check=True)

# Export the model from the checkpoint
subprocess.run(["python", "export_checkpoint.py", f"../checkpoint/{CONFIG_FILE_NAME}/last.ckpt", f"../model/{CONFIG_FILE_NAME}.pth"], check=True)

# List the exported model
subprocess.run(["ls", "-alh", f"../model/{CONFIG_FILE_NAME}.pth"], check=True)

# Do a quick dragon prompt validation
subprocess.run(["python3", "dragon_test.py", f"../model/{CONFIG_FILE_NAME}.pth", "cuda bf16"], check=True)
