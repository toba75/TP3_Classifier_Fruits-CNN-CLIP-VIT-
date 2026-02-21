FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

COPY . /workspace
RUN chmod +x /workspace/run_experiments.sh

CMD ["bash", "/workspace/run_experiments.sh"]
