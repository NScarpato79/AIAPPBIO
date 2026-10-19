FROM python:3.11-slim

# g++ consente a PyTensor (backend di PyMC) di compilare i grafi in C: senza, il
# campionamento NUTS funziona ma e' molto piu' lento.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "vancotwin", "--output", "outputs"]
