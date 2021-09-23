# LEEP
Implementation of LEEP: A New Measure to Evaluate Transferability of Learned Representations: http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf 

## Requirements

```
pip3 install --user -r requirements.txt
```

## Setup
Run `sh setup.sh` to create the right directory paths.

## Calculating LEEP

The class expects a path to a `.txt` file consisting of the following:
```
# <unique gold labels target dataset>
# <unique gold labels source dataset>
<output probabilities of pretrained model of source set Z applied on target set Y> <gold label instance from Y>
```
For example, this is in your `output.txt` file:
```
# [A, B, C, D]
# [U, V, W, X, Y, Z]
[0.00342972157523036, 0.03722788393497467, 1.3426588907350379e-07, 0.8358138203620911, 0.007074566558003426, 0.11645391583442688] A
...
```
To run the script:

```bash
python leep.py output.txt
# returns e.g. -8.695098322753905
```

## Embedding LEEP

The `emblem.py` script can be used to generate a LEEP-ready file which will have the desired target label set and pre-trained embeddings as the source labels (i.e. embedding feature dimensions are treated as classes). It supports the following encoding schemes:

* fasttext
* GloVe
* HuggingFace Transformers

### Input Format

Convert your  target dataset into the following format for token-wise classification tasks:

```
["token0", "token1", ..., "tokenN"] ["label0", "label0", ..., "labelN"]
["token0", ..., "tokenM"] ["label0", ..., "labelM"]
...
```

Use the following format for sentence-wise classification tasks:

```
["token0", "token1", ..., "tokenN"] "label0"
["token0", ..., "tokenM"] "label1"
...
```

### Generating Embeddings

After the input file is in the above format, the `embleep.py` script can be used to generate embeddings for the input dataset:

```bash
python embleep.py input.txt output.txt embedding_model \
	--pooling strategy \
	--batch_size batch_size
```

Specify an embedding models using the syntax `model_type:arguments`. Embedding representations for sentences must be pooled from token-level embeddings. As such, please also specify a pooling strategy from among `mean` (mean-pool all tokens in the sentence), `first` (use the first embedding, e.g. [CLS] token).

```bash
# fasttext
python embleep.py input.txt output.txt \
	"fasttext:/path/to/fasttext.vec" \
	--pooling mean
# GloVe
python embleep.py input.txt output.txt \
	"glove:/path/to/glove.vec" \
	--pooling mean
# Transformer Language Model
python embleep.py input.txt output.txt \
	"transformer:model_name" \
	--pooling mean
# Transformer Language Model (CLS)
python embleep.py input.txt output.txt \
	"transformer+cls:model_name" \
	--pooling first
```

The output file follows the format:

```
# ["class0", "class1", ... "classN"]
# ["dim0", "dim1", ..., "dimD"]
[0.0629, 0.0029, ..., probD] "label0"
...
```

It can be supplied to the `leep.py` script to obtain the LEEP score.

### Applying PCA

In order to normalize embeddings of different dimensionalities, larger embeddings can be reduced to their N principal components using Principal Component Analysis (PCA). This is disabled by default and can be enabled by using the `-pca` flag:

```bash
python embleep.py input.txt output.txt embedding_model \
	-pca num_components
```

