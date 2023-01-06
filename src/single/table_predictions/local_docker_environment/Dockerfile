# 開発・デバッグ用イメージ作成
FROM python:3.10.9

WORKDIR /workspace

RUN apt-get update && apt-get install libgl1 -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt