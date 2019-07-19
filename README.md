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

* [Forward Flair Embeddings](https://schweter.eu/cloud/flair-lms/lm-eu-large-forward-v0.2.pt)
* [Backward Flair Embeddings](https://schweter.eu/cloud/flair-lms/lm-eu-large-backward-v0.2.pt)

### Flair import

```python
from flair.embeddings import FlairEmbeddings

embeddings_forward  = FlairEmbeddings("lm-eu-large-forward-v0.2.pt")
embeddings_backward = FlairEmbeddings("lm-eu-large-backward-v0.2.pt")
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

| Language model   | Run 1 | Run 2 | Run 3 | Final F-Score
| ---------------- | ----- | ----- | ----- | -------------
| ELMo             | 97.35 | 97.33 | 97.38 | 97.35
| Flair Embeddings | 97.60 | 97.67 | 97.67 | **97.65**
| mBERT uncased    | 95.06 | 94.62 | 94.70 | 94.79
| mBERT cased      | 94.26 | 94.43 | 94.33 | 94.35

## WikiANN

Experiments on the WikiANN dataset for Basque are coming soon.

# ToDo

* [ ] WikiANN experiments
* [ ] Run NER and PoS tagging experiments on (already) trained XLNet models
* [ ] Add training scripts
* [ ] Play around with `allennlp` to add configuration for training NER and PoS tagging models