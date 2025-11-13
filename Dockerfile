FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    --trusted-host download.pytorch.org \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    xformers==0.0.33

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
