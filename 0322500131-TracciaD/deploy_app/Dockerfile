FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/holoscan/app

ENV HOLOSCAN_INPUT_PATH=/var/holoscan/input/
ENV HOLOSCAN_OUTPUT_PATH=/var/holoscan/output/
ENV HOLOSCAN_WORKDIR=/var/holoscan/
ENV HOLOSCAN_MODEL_PATH=/opt/holoscan/models/

WORKDIR /var/holoscan

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /tmp/requirements.txt

RUN mkdir -p \
    /opt/holoscan/app \
    /opt/holoscan/models \
    /opt/holoscan/docs \
    /etc/holoscan \
    /var/holoscan/input \
    /var/holoscan/output

COPY app.py /opt/holoscan/app/app.py
COPY src/ /opt/holoscan/app/src/
COPY models/ /opt/holoscan/models/
COPY README.md /opt/holoscan/docs/README.md

COPY app.json /etc/holoscan/app.json
COPY pkg.json /etc/holoscan/pkg.json
LABEL org.opencontainers.image.title="Pneumonia Detection MAP"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="MONAI-based pneumonia classification from DICOM images"

ENTRYPOINT ["python", "-u", "/opt/holoscan/app/app.py"]
