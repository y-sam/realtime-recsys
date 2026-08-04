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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
