# Ambiente per addestramento, valutazione e test (il MAP e' in deploy_app/)
FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless numpy==2.4.6
COPY . .
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
