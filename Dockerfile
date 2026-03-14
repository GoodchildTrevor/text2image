FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

RUN uv pip install --system --no-cache-dir \
    --trusted-host pypi.nvidia.com \
    --extra-index-url https://pypi.nvidia.com \
    --concurrent-downloads 2 \
    nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12

RUN uv pip install --system --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --trusted-host download.pytorch.org \
    --trusted-host pypi.nvidia.com \
    --concurrent-downloads 4 \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    xformers==0.0.33

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
