FROM ubuntu:jellyfish

RUN apt update && apt install python3.10-dev git

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.10