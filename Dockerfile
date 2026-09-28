FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/rcrn/src

WORKDIR /opt/rcrn

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "rcrn.cli.evaluate", "--config", "configs/experiment/_smoke.yaml"]
