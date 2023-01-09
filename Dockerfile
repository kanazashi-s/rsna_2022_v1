# 開発・デバッグ用イメージ作成
FROM nvcr.io/nvidia/pytorch:22.10-py3

WORKDIR /workspace

RUN apt-get update && apt-get install libgl1 -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

ENV IS_KAGGLE_ENVIRONMENT=0
ENV TRANSFORMERS_CACHE=/workspace/transformers_cache
ENV TOKENIZERS_PARALLELISM=false

ENV TORCH_HOME=/workspace/torch_cache
