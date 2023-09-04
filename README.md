# vae


## Introduction

This project implements a variational autoencoder with convolution. Variational autoenconders were introduced by Kingma and Welling in their seminal [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf).

## Train and Eval

Use docker for both train and eval. First build the image by running

    docker compose build

Run the train script in container:

    docker compose up train

Once training is complete (since we need the model checkpoint), run the eval script:

    docker compose up evaluate
