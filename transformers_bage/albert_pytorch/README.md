## albert_zh_pytorch

This repository contains a PyTorch implementation of the albert model from the paper 

[A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

by Zhenzhong Lan. Mingda Chen....

arxiv: https://arxiv.org/pdf/1909.11942.pdf

## Pre-LN and Post-LN

* Post-LN: . In the original Transformer, after Layer Norm followed Residual, we called this `Post-LN Transformer`

* Pre-LN: Change the position of Layer Norm, for example in the process of Residual (called `Pre-LN Transformer`)

![](https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/Selection_001.png)

paper: [On Layer Normalization in the Transformer Architecture](https://openreview.net/forum?id=B1x8anVFPr)

**Way of use**

According to the model weight file provided by [brightmart] (https://github.com/brightmart/albert_zh), you need to add the `ln_type` parameter to the configuration file, as follows:

```json
{
  "attention_probs_dropout_prob": 0.0,
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.0,
  "hidden_size": 768,
  "embedding_size": 128,
  "initializer_range": 0.02, 
  "intermediate_size": 3072 ,
  "max_position_embeddings": 512, 
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": 21128,
  "ln_type":"postln"  # postln or preln
}
```
## show type 

**Cross-Layer Parameter Sharing**: ALBERT use cross-layer parameter sharing in Attention and FFN(FeedForward Network) to reduce number of parameter.

modify the `share_type` parameter:

* all: Attention and FFN layer parameters are shared
* ffn:　Only share FFN layer parameters
* attention: Only share the attention layer parameter
* None:  No parameter sharing

**Way of use**

When loading `config`, specify the `share_type` parameter as follows:

```python
config = BertConfig.from_pretrained(bert_config_file,share_type=share_type)
```
## Download Pre-trained Models of Chinese

Thanks to Brightmart for providing Chinese model weights.：[github](https://github.com/brightmart/albert_zh)

1. [albert_large_zh](https://storage.googleapis.com/albert_zh/albert_large_zh.zip) Parameter quantity, number of layers 24, size 64M

2. [albert_base_zh(Small model experience version)](https://storage.googleapis.com/albert_zh/albert_base_zh.zip), The parameter quantity is 12M, the number of layers is 12, and the size is 40M.

3. [albert_xlarge_zh](https://storage.googleapis.com/albert_zh/albert_xlarge_zh.zip) The parameter quantity, the number of layers is 24, and the file size is 230M.

## Pre-training

**n-gram**: In the original paper, n-grams are randomly generated according to the following distribution, and the default max_n is 3

   <p align="center"><img width="200" src="https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/n-gram.png" /></p>
1. Convert text data into a one-line, one-sentence format, and use `\n` to split between different documents

2. Run `python prepare_lm_data_ngram.py --do_data` to generate the ngram mask format data set respectively.

3. Run `python run_pretraining.py --share_type=all` for model pre-training

** Model size**

The following is the result of an experiment with `bert-base`

| embedding_size | share_type | model_size |
| :------- | :---------: | :---------: |
| 768 | None | 476.5M |
| 768 | attention | 372.4M |
| 768 | ffn | 268.6M|
| 768 |all | 164.6M|
| |  |  |
| 128 | None | 369.1M |
| 128 | attention | 265.1M |
| 128 | ffn | 161.2M|
| 128 |all | 57.2M|


## Downstream task Fine-tuning

1. Download the pre-trained albert model

2. Run `python convert_albert_tf_checkpoint_to_pytorch.py` to convert the TF model weights to the pytorch model weights (by default shar_type=all)

3. Download the corresponding dataset, such as the [LCQMC] (https://drive.google.com/open?id=1HXYMqsXjmA5uIfu_SFqP7r_vZZG-m_H0) dataset, which contains training, validation, and test sets. The training set contains 240,000 colloquial descriptions of Chinese. Sentence pairs, the label is 1 or 0. 1 is semantically similar to sentences, and 0 is semantically dissimilar.

4. Run `python run_classifier.py --do_train` for Fine-tuning training

5. Run `python run_classifier.py --do_test` to test evaluation

## Results

Question matching language task: LCQMC (Sentence Pair Matching)

Model | Development Set (Dev) | Test Set (Test) |
| :------- | :---------: | :---------: |
| ALBERT-zh-base(tf) | 86.4 | 86.3 |
| ALBERT-zh-base(pytorch) | 87.4 | 86.4 |




