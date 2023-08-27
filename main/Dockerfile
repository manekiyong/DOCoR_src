FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV DEBIAN_FRONTEND noninteractive
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1


RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install make

RUN pip install spacy==3.1.2

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.1.0/en_core_web_lg-3.1.0-py3-none-any.whl

ADD build /build
WORKDIR /build
RUN make upgrade_pip
RUN make python_requirements


ADD src/ /src

WORKDIR /src
