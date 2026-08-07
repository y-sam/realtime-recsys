FROM python:3.12-slim
WORKDIR /srv
# libgomp1: LightGBM's compiled extension needs OpenMP, absent from the slim base image.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Single worker (the event loop blocks on synchronous LightGBM/PyTorch inference,
# so throughput is flat across concurrency with only one) is the dev-safe default;
# production should set UVICORN_WORKERS to the host's vCPU count. Each worker loads
# its own copy of the ranker and two-tower artifacts -- ~612KB today, but a cost
# that scales with worker count if the models grow.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
