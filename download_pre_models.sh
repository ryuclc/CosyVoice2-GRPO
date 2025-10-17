#!/bin/bash

# Define the base URL
BASE_URL="https://www.modelscope.cn/iic/"

# List of model names
model_names=(
  # "CosyVoice2-0.5B"
  # "CosyVoice-300M"
  # "CosyVoice-300M-25Hz"
  # "CosyVoice-300M-SFT"
  # "CosyVoice-300M-Instruct"
  "CosyVoice-ttsfrd"
)

mkdir -p pretrained_models
git lfs install
# Loop through each model name and clone the repository
for model_name in "${model_names[@]}"; do
  echo "Cloning ${model_name}..."
  git clone "${BASE_URL}${model_name}.git" pretrained_models/${model_name}
done
