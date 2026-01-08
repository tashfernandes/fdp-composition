FROM python:3.12-slim

WORKDIR /artifact

ENV PYTHONPATH=/artifact

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super \
    dvipng

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
RUN mkdir -p figures

CMD ["python3","./scripts/generate_plots.py"]

