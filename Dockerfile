# 阶段 1：原生架构下载模型（不模拟、不 import torch）
FROM --platform=$BUILDPLATFORM python:3.11-slim AS model
ENV HF_HOME=/models
RUN pip install --no-cache-dir huggingface_hub==1.33.0 && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5')"


FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt . 
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
RUN pip install -r requirements.txt --no-cache-dir

ENV HF_HOME=/models
COPY --from=model /models /models
ENV HF_HUB_OFFLINE=1

COPY --from=notes . /app/notes
ENV NOTES_DIR=/app/notes
COPY . .


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
