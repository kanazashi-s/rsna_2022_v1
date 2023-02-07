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

# install nvjpeg2k https://github.com/louis-she/nvjpeg2k-python
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/libnvjpeg-2k/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-0.6.0.28-archive.tar.xz
RUN tar Jxfv libnvjpeg_2k-linux-x86_64-0.6.0.28-archive.tar.xz
RUN git clone --recursive https://github.com/louis-she/nvjpeg2k-python.git

WORKDIR /tmp/nvjpeg2k-python/build
ARG PATH_OF_THE_LIBNVJPEG_2K="/tmp/libnvjpeg_2k-linux-x86_64-0.6.0.28-archive"

RUN cmake .. -DCMAKE_BUILD_TYPE=Debug -DNVJPEG2K_PATH=${PATH_OF_THE_LIBNVJPEG_2K} -DNVJPEG2K_LIB=${PATH_OF_THE_LIBNVJPEG_2K}/lib/libnvjpeg2k_static.a
RUN make

WORKDIR /workspace
