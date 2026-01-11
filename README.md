# Image Captioning

This repository contains my implementation of an image captioning model. The older notebooks uses features and functions coded from scratch (for learning purposes), while newer ones utilize more up to date techniques for reaching higher model accuracy. 

The model takes an image as input and generates a descriptive English caption.

## Project Overview

- I used several different model architectures, with focus on vision transformers.

## Benchmark (COCO val2017)

- Checkpoint: `models/vit_transformer_coco_captioner_2026-01-08_best.ckpt`
- Decode: beam search
- Images: 100 random samples from COCO val2017
- BLEU-1: 0.7087
- BLEU-2: 0.5555
- BLEU-3: 0.4337
- BLEU-4: 0.3365
- METEOR: 0.5016
- ROUGE-L: 0.5462
- CIDEr: 0.8862
- Speed: 132.62s total, 0.754 img/sec


## Inspiration

This project was inspired by the following papers:
- https://cs.stanford.edu/people/karpathy/cvpr2015.pdf
- https://arxiv.org/pdf/1411.4555
