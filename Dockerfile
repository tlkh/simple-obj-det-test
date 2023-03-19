
FROM nvcr.io/nvidia/pytorch:23.02-py3
LABEL maintainer="Timothy Liu <timothy_liu@mymail.sutd.edu.sg>"
USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade --no-cache-dir \
       tqdm ipywidgets jupyterlab matplotlib streamlit \
       datasets transformers evaluate timm albumentations \
       && \
    jupyter lab clean

COPY . /opt/

RUN python3 /opt/cache.py

