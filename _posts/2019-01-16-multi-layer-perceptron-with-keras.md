---
layout: post
comments: true
date: 2019-02-16 12:00:00
title: Multi-layer Perceptron with Keras
tags: supervised-learning
---

Recently, I discovered xLearn which is a high performance, scalable ML package that implements factorization machines (FM) and field-aware factorization machines (FFM). The first version of xLearn was released about 1 year ago as of writing.

![xlearn]({{ '/images/xlearn_logo.png' | relative_url }})

<!-- more -->

FM was initially introduced and popularised by Steffen Rendle after winning 4th positions in both tracks in KDD cup 2012.

FM is a supervised learning algorithm that can be used for regression, classifications, and ranking. They are non-linear in nature and are trained using stochastic gradient descent (SGD).

FFM was later introduced and further improves FM. It was then used to win the 1st prize of three CTR competitions hosted by Criteo, Avazu, Outbrain, and also the 3rd prize of RecSys Challenge 2015.

To understand FM and FFM, we need to first understand a logistic regression model and a Poly2 model.

## Context

Assume we have the following dataset where we want to predict Clicked outcome using Publisher, Advertiser, and Gender:

$$\mathcal{A}$$


## Reference

[1] Bryan McCann, et al. ["Learned in translation: Contextualized word vectors."](https://arxiv.org/abs/1708.00107) NIPS. 2017.

[2] Kevin Clark et al. ["Semi-Supervised Sequence Modeling with Cross-View Training."](https://arxiv.org/abs/1809.08370) EMNLP 2018.

---

Thank you for reading the post! See you in the next one!