FROM ubuntu:latest
FROM python:3.9-slim-bookworm
# FROM gcc:4.9

RUN apt update
RUN apt install python3 -y
# RUN apt-get install gcc python3-dev
RUN apt install gcc -y

WORKDIR /usr/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /usr/app

COPY . . 
