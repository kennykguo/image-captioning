# Image captioning

A friend challenged me to build an image captioning model when I had no deep learning experience. I spent my first year summer working through the history of deep learning, starting with LSTMs and later moving to Transformers. I wanted to understand the models well enough to build the captioning system myself, including the word tokenizer, data loading, training, and inference.

I came back to the project later with a vision transformer as the image encoder. The older notebooks are still here, so you can follow the progression. The latest version is in [the January 8 notebook](01-08-26-coco2017-image-captioning.ipynb).

## From an image to words

The model needs some representation of the image that it can use to predict a caption. The earlier experiments used ResNet-50 to produce one image vector. In the current version, I use a pretrained ViT-B/16 and keep its patch features.

![A fox image beside the same image split into patches](images/readme/fox-patches.jpg)

The input is cropped to 224 × 224 pixels, then divided into 16 × 16 patches. That gives 196 patches. Each patch becomes a vector, and the ViT processes those vectors together, along with a class token. Instead of using the final classification output, I take this sequence of features and project it from 768 dimensions to 512.

Keeping the sequence gives the caption decoder several image features to attend to as it generates words. The patch illustration above just shows how the image is divided up; the grid doesn't show what the model is paying attention to.

The decoder has four Transformer layers with eight attention heads. It uses the image features and the words so far to predict the next word. For example, after seeing `a person`, it produces a distribution over what could follow. The chosen word is added to the sequence, and the process repeats until `<END>` or the length limit. The notebooks include greedy decoding, beam search, and top-p sampling.

During training, I already have the complete caption. I shift it by one word, so the input might be `<START> a person` while the target is `a person skiing`. A causal mask keeps each position from looking ahead at the answer. The loss compares the predictions with the target words and ignores padding.

## Turning words into numbers

The tokenizer is a dictionary built from the training captions. I lowercase the text, remove punctuation, and count the words. Words appearing fewer than five times become `<UNK>`. Each remaining word gets an integer ID, with four reserved IDs for `<PAD>`, `<START>`, `<END>`, and `<UNK>`. The model looks up a learned embedding for each ID, and I use the reverse dictionary to turn its predictions back into words.

The current encoder uses pretrained ImageNet weights from torchvision and stays frozen during training. The decoder uses PyTorch's Transformer layers. I wrote the captioning model and the pipeline around these components; the older notebooks include my earlier implementations.

## Some outputs

![Deer, fox, and skiing photos used in the saved inference run](images/readme/sample-images.jpg)

These are saved outputs from [benchmark.ipynb](benchmark.ipynb):

| Image | Model caption |
| --- | --- |
| Skiers | a person riding skis down a snow covered slope |
| Fox | a brown and white cat standing next to a tree |
| Deer | a couple of giraffe standing next to each other |

The skiing caption makes sense. The fox gets called a cat, and the deer become giraffes. The model has learned enough about sentences to produce something that sounds reasonable, even when it gets the actual subject wrong.

## The notebooks

| Notebook | What's in it |
| --- | --- |
| [01-08-26-coco2017-image-captioning.ipynb](01-08-26-coco2017-image-captioning.ipynb) | Current ViT + Transformer model, COCO preprocessing, training, checkpoints, and sampling. Start here for the model. |
| [benchmark.ipynb](benchmark.ipynb) | Load a checkpoint, caption your own images, and evaluate on COCO. Start here to try inference. |
| [01-07-26-coco2017-image-captioning.ipynb](01-07-26-coco2017-image-captioning.ipynb) | Earlier January experiment. |
| [07-01-24-lstm-coco2017-image-captioning.ipynb](07-01-24-lstm-coco2017-image-captioning.ipynb) | Early LSTM experiment on COCO. |
| [07-01-24-lstm-flickr30k-image-captioning.ipynb](07-01-24-lstm-flickr30k-image-captioning.ipynb) | Early LSTM experiment on Flickr30k. |
| [07-01-24-vanilla-transformer-flickr30k-image-captioning.ipynb](07-01-24-vanilla-transformer-flickr30k-image-captioning.ipynb) | Early Transformer experiment on Flickr30k. |

## Run it

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyterlab matplotlib
jupyter lab
```

Run the notebooks from the repo root. The January training notebooks were written on Kaggle, so change the dataset paths before running them locally.

### Caption your own images

Open `benchmark.ipynb` and find the **tweakable constants** cell:

```python
CHECKPOINT_PATH = 'models/vit_transformer_coco_captioner_2026-01-08_best.ckpt'
IMAGES_DIR = './images'
MODE = 'beam'  # also 'greedy' or 'top_p'
RUN_COCO = False
```

Put your photos in `images/` and run the cells in order. Setting `RUN_COCO = False` lets you try your photos without downloading COCO. PyTorch selects CUDA when available and otherwise uses the CPU. The ViT constructor also downloads its pretrained weights on first use if they aren't cached.

The trained checkpoint and datasets aren't checked into Git. You need an existing checkpoint or a training run first. The checkpoint must contain `model_state_dict`, `config`, `stoi`, and `itos`; a bare `.pth` state dictionary isn't enough for the benchmark loader.

### Train a model

Open `01-08-26-coco2017-image-captioning.ipynb` and change `coco_path` to your COCO 2017 directory. It should contain `train2017/`, `val2017/`, and `annotations/` with `captions_train2017.json` and `captions_val2017.json`.

Run the notebook in order. The training constants are set to 10 epochs and a batch size of 512; lower the batch size if it doesn't fit your GPU. Training uses AdamW, a cosine learning-rate schedule, and gradient clipping.

The notebook saves best and last `.ckpt` files plus a `.pth` weights file in its working directory. Filenames include the run date. Point `CHECKPOINT_PATH` in the benchmark notebook at your saved `.ckpt`.

## Evaluation

The saved benchmark used **100 randomly sampled COCO val2017 images**, seed 0, with beam search and `vit_transformer_coco_captioner_2026-01-08_best.ckpt`.

| Metric | Saved result |
| --- | ---: |
| BLEU-1 | 0.7087 |
| BLEU-2 | 0.5555 |
| BLEU-3 | 0.4337 |
| BLEU-4 | 0.3365 |
| METEOR | 0.5016 |
| ROUGE-L | 0.5462 |
| CIDEr | 0.8862 |

That run took 132.62 seconds, or 0.754 images per second. The hardware wasn't recorded with the result.

These numbers only cover the 100 sampled images. BLEU and METEOR use NLTK, ROUGE-L uses the notebook's LCS implementation, and CIDEr uses `pycocoevalcap`.

To repeat the setup, set `RUN_COCO = True`, check `COCO_IMAGES_DIR` and `COCO_ANN_JSON`, and leave `COCO_EVAL_N = 100`. For METEOR and CIDEr, also install:

```bash
python -m nltk.downloader wordnet omw-1.4
pip install git+https://github.com/salaniz/pycocoevalcap
```

## Papers I started with

- [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
