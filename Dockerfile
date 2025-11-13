FROM python:3.13-slim

WORKDIR /app

COPY resources/pip.conf /etc/pip.conf

COPY image_gen/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY image_gen/ .

COPY image_gen/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]