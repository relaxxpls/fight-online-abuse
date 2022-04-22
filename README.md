# Fight Online Abuse

## Introduction

Natural language processing based solution to confidently and accurately tell whether a particular comment is abusive. I used a transformer (BERT) in PyTorch to solve the **multilabel text classification** problem.

## Problem analysis

### Multilabel Classification

This is one of the most common business problems where a given piece of text/sentence/document needs to be classified into one or more of categories out of the given list. For example, a movie can be categorized into 1 or more genres.

## Dataset

* **Jigsaw Toxic Comment Classification Challenge**: [Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
* We are training only to the first csv file from the data dump: `train.csv`. This file has the format as follows:
  * Comment Text
  * `toxic`
  * `severe_toxic`
  * `obscene`
  * `threat`
  * `insult`
  * `identity_hate`
* Each comment can be marked for multiple categories. If the comment is `toxic` and `obscene`, then for both those headers the value will be `1` and for the others it will be `0`.

## Requirements

* Python 3.9+
* GPU enabled setup
* Pytorch, Transformers, SKLearn and other stock Python ML libraries

## Training details

* It is to be noted that the overall mechanisms for a multiclass and multilabel problems are similar.

### Model: `DistilBert`

* DistilBERT is a smaller transformer model as compared to BERT or Roberta. It is created by process of distillation applied to Bert.
* [Blog-Post](https://medium.com/huggingface/distilbert-8cf3380435b5)
* [Research Paper](https://arxiv.org/pdf/1910.01108)
* [Documentation for python](https://huggingface.co/docs/transformers/model_doc/distilbert)

### Criterion: Binary Cross Entropy (BCE)

* Loss function is designed to evaluate all the probability of categories individually rather than as compared to other categories. Hence the use of `BCE` rather than `Cross Entropy` when defining loss.

### Activation: `Sigmoid`

* Sigmoid of the outputs calcuated to rather than Softmax due to above reasons.

### Loss metrics

* The [loss metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html) and **Hamming Score**  are used for direct comparison of expected vs predicted.
