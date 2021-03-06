# plur: **P**re-trained **L**anguage Models for **U**nder-**r**epresented Languages

This repository contains pre-trained language models for under-represented languages in NLP.

Language models are available for Flair and ELMo (soon: XLNet). All trained language models
are evaluated on NER and PoS tagging downstream tasks with Flair.

# Basque

## Corpus

Flair Embeddings and ELMo are trained on a recent Wikipedia dump and various texts are
collected from OPUS and the Leipzig Corpora Collection.

Some statistics:

* Number of tokens: 57,110,741 (untokenized), 72,683,662 (tokenized)
* Size: 417M (untokenized), 440M (tokenized)

Remember: Flair Embeddings are trained on raw and untokenized texts, so no tokenization is needed.
The underlying language model is a character-based one, in contrast to ELMo: ELMo needs tokenized
input. For tokenization we use a very simple tokenization method that is adopted from the
Tensor2Tensor repository.

## ELMo

We use the official implementation from the [`bilm-tf` repository](https://github.com/allenai/bilm-tf).
Due to limited hardware resources, we limit the vocabulary to 700,000 tokens. We train for 10 epochs
on a GTX 1080.

### Release:

* [ELMo options file](https://schweter.eu/cloud/eu-elmo/options.json)
* [ELMo weights](https://schweter.eu/cloud/eu-elmo/weights.hdf5)

### Flair import

The trained ELMo model can easily be used in Flair:

```python
from flair.embeddings import ELMoEmbeddings

embeddings = ELMoEmbeddings(options_file="https://schweter.eu/cloud/eu-elmo/options.json", 
                            weight_file="https://schweter.eu/cloud/eu-elmo/weights.hdf5")
```

## Flair Embeddings

We follow the official recommendations for training Flair Embeddings from the
[Flair documentation](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).

The following parameters are used:

| Parameter         | Value
| ----------------- | ------
| `hidden_size`     | 2048
| `dropout`         | 0.1
| `nlayers`         | 1
| `sequence_length` | 250
| `mini_batch_size` | 100
| `max_epochs`      | 10
| `learning_rate`   | 20

We did not decrease the initial learning rate during training.

### Release:

* [Forward Flair Embeddings](https://schweter.eu/cloud/flair-lms/lm-eu-opus-large-forward-v0.2.pt)
* [Backward Flair Embeddings](https://schweter.eu/cloud/flair-lms/lm-eu-opus-large-backward-v0.2.pt)

### Flair import

```python
from flair.embeddings import FlairEmbeddings

embeddings_forward  = FlairEmbeddings("lm-eu-opus-large-forward-v0.2.pt")
embeddings_backward = FlairEmbeddings("lm-eu-opus-large-backward-v0.2.pt")
```

**Notice**: Our trained embeddings are included in Flair >= *0.4.3*. So you can easily load them with:

```python
from flair.embeddings import FlairEmbeddings

embeddings_forward  = FlairEmbeddings("eu-forward")
embeddings_backward = FlairEmbeddings("eu-backward")
```

## NER

We use the Basque Named Entities Corpus (EIEC) that can be obtained from [here](http://ixa.eus/node/4486?language=en).
This corpus has a total of 2552 training and 842 test sentences. For evaluation, the official
CoNLL-2003 evaluation script is used. We report averaged F-Score over three runs.

| Language model   | Run 1 | Run 2 | Run 3 | Final F-Score
| ---------------- | ----- | ----- | ----- | -------------
| ELMo             | 81.50 | 83.13 | 81.41 | **82.01**
| Flair Embeddings | 81.62 | 81.56 | 81.51 | 81.56

## UD

We use the Basque Universal Dependencies in version 1.2 for comparison.
The corpus has a total of 5,396 training, 1,798 development and 1,799 test sentences.
We report averaged accuracy over three runs.

| Language model   | Run 1 | Run 2 | Run 3 | Final Accuracy
| ---------------- | ----- | ----- | ----- | --------------
| ELMo             | 97.35 | 97.33 | 97.38 | 97.35
| Flair Embeddings | 97.60 | 97.67 | 97.67 | **97.65**
| mBERT uncased    | 95.06 | 94.62 | 94.70 | 94.79
| mBERT cased      | 94.26 | 94.43 | 94.33 | 94.35

## WikiANN

Experiments on the WikiANN dataset for Basque are coming soon.

# Tamil

## Corpus

Flair Embeddings and ELMo are trained on a recent Wikipedia dump and various texts are
collected from OPUS and the Leipzig Corpora Collection.

Some statistics:

* Number of tokens: 18,365,106 (untokenized), 21,581,878 (tokenized)
* Size: 423M (untokenized), 426M (tokenized)

## ELMo

We use the official implementation from the [`bilm-tf` repository](https://github.com/allenai/bilm-tf).
Due to limited hardware resources, we limit the vocabulary to 700,000 tokens. We train for 10 epochs
on a GTX 1080.

### Release:

* [ELMo options file](https://schweter.eu/cloud/ta-elmo/options.json)
* [ELMo weights](https://schweter.eu/cloud/ta-elmo/weights.hdf5)

### Flair import

The trained ELMo model can easily be used in Flair:

```python
from flair.embeddings import ELMoEmbeddings

embeddings = ELMoEmbeddings(options_file="https://schweter.eu/cloud/ta-elmo/options.json",
                            weight_file="https://schweter.eu/cloud/ta-elmo/weights.hdf5")
```

## Flair Embeddings

We follow the official recommendations for training Flair Embeddings from the
[Flair documentation](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md).

The following parameters are used:

| Parameter         | Value
| ----------------- | ------
| `hidden_size`     | 2048
| `dropout`         | 0.1
| `nlayers`         | 1
| `sequence_length` | 250
| `mini_batch_size` | 100
| `max_epochs`      | 10
| `learning_rate`   | 20

We did not decrease the initial learning rate during training.

### Release:

* [Forward Flair Embeddings](https://schweter.eu/cloud/flair-lms/lm-ta-opus-large-forward-v0.1.pt)
* [Backward Flair Embeddings](https://schweter.eu/cloud/flair-lms/lm-ta-opus-large-forward-v0.1.pt)

### Flair import

```python
from flair.embeddings import FlairEmbeddings

embeddings_forward  = FlairEmbeddings("lm-ta-opus-large-forward-v0.1.pt")
embeddings_backward = FlairEmbeddings("lm-ta-opus-large-forward-v0.1.pt")
```

**Notice**: Our trained embeddings are included in Flair >= *0.4.3*. So you can easily load them with:

```python
from flair.embeddings import FlairEmbeddings

embeddings_forward  = FlairEmbeddings("ta-forward")
embeddings_backward = FlairEmbeddings("ta-backward")
```

## UD

We use the Tamil Universal Dependencies in version 1.2 for comparison.
The corpus has a total of 400 training, 80 development and 120 test sentences.
We report averaged accuracy over three runs. We use Subword Embeddings with different
vocabulary sizes and a fixed dimension of 300 for both Flair and ELMo models.

### Flair

| BPE vocab | Run 1 | Run 2 | Run 3 | Final Accuracy
| --------- | ----- | ----- | ----- | --------------
| 200,000   | 92.31 | 91.55 | 92.46 | 92.11
| 100,000   | 92.06 | 92.51 | 92.51 | 92.36
| 50,000    | 92.51 | 92.61 | 93.11 | **92.74**
| 25,000    | 92.61 | 92.06 | 92.81 | 92.49
| 10,000    | 91.86 | 92.31 | 91.30 | 91.82
|  5,000    | 92.06 | 92.56 | 92.51 | 92.37
|  3,000    | 92.31 | 92.86 | 92.76 | 92.64
|  1,000    | 92.41 | 92.36 | 93.31 | 92.69

### ELMo

| BPE vocab | Run 1 | Run 2 | Run 3 | Final Accuracy
| --------- | ----- | ----- | ----- | --------------
| 200,000   | 91.91 | 91.45 | 92.76 | 92.04
| 100,000   | 91.96 | 92.01 | 92.16 | 92.04
| 50,000    | 91.96 | 92.46 | 91.75 | **92.06**
| 25,000    | 92.26 | 90.90 | 92.11 | 91.76
| 10,000    | 91.91 | 91.50 | 91.65 | 91.69
|  5,000    | 92.36 | 91.55 | 91.91 | 91.94
|  3,000    | 92.06 | 91.96 | 92.06 | 92.03
|  1,000    | 92.06 | 91.80 | 91.70 | 91.85

# ToDo

* [ ] WikiANN experiments
* [ ] Run NER and PoS tagging experiments on (already) trained XLNet models
* [ ] Add training scripts
* [ ] Play around with `allennlp` to add configuration for training NER and PoS tagging models
