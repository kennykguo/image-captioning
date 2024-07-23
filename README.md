# Image Captioning

This repository contains my implementation of an image captioning model. The model takes an image as input and generates a descriptive English caption.

## Project Overview

- I used several different model architectures such as CNN-LSTMs and CNN-Transformers.
- The project involves the MSCOCO2017 dataset. I initially used the Flickr30k, but I found that my captioning results were much better on MSCOCO2017, most likely because it has more data.

## Results

This model achieved a maximum BLEU-4 caption score of 11.0

![download](https://github.com/user-attachments/assets/ac774416-19c5-44df-a223-c63617234290)

Original Caption: <START> polar bear swimming in the water by wall <END>

Generated Caption: polar bear swimming by large wave <END>


## Inspiration

This project was inspired by the following papers:
- https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
- https://arxiv.org/pdf/1411.4555
