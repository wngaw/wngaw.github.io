---
layout: post
comments: true
date: 2019-02-16 12:00:00
title: Field-aware Factorization Machines with xLearn
tags: supervised-learning
---

Recently, I discovered xLearn which is a high performance, scalable ML package that implements factorization machines (FM) and field-aware factorization machines (FFM). The first version of xLearn was released about 1 year ago as of writing.

![xlearn]({{ '/images/xlearn_logo.png' | relative_url }})

FM was initially introduced and popularised by *Steffen Rendle* after winning 4th positions in both tracks in KDD cup 2012.

FM is a supervised learning algorithm that can be used for regression, classifications, and ranking. They are non-linear in nature and are trained using stochastic gradient descent (SGD).

FFM was later introduced and further improves FM. It was then used to win the 1st prize of three CTR competitions hosted by Criteo, Avazu, Outbrain, and also the 3rd prize of RecSys Challenge 2015.

To understand FM and FFM, we need to first understand a logistic regression model and a Poly2 model.

## Context

Assume we have the following dataset where we want to predict Clicked outcome using Publisher, Advertiser, and Gender:

| **Dataset** | **Clicked**  | **Publisher** | **Advertiser** | **Gender** |
| :---: | :---: | :---: | :---: | :---: |
| TRAIN | YES | ESPN | NIKE | MALE |
| TRAIN | NO | NBC | ADIDAS | MALE |

## The Optimisation Problem

The model $$\pmb{w}$$ for logistic regression, Poly2, FM, and FFM, is obtained by solving the following optimisation problem:

$$\underset{\pmb{w}}{\min} \ \ \frac{\lambda}{2} \left\|\pmb{w} \right\|^{2} + \sum\limits_{i=1}^m log(1 + exp(-y_{i}\phi(\pmb{w}, \pmb{x_i}))$$

where

- dataset contains  $$m$$ instances $$(y_{i}, \pmb{x_i})$$
- $$y_i$$ is the label and $$\pmb{x_i}$$ is a n-dimensional feature vector
- $$\lambda$$ is a regularisation parameter
- $$\phi(\pmb{w}, \pmb{x})$$ is the association between $$\pmb{w}$$ and $$\pmb{x}$$

## Logistic Regression

A logistic regression estimates the outcome by learning the weight for each feature.

For logistic regression,

$$\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i $$

where

- $$w_0$$ is the global bias
- $$w_i$$ is the strength of the i-th variable

In our context,

$$\phi(\pmb{w}, \pmb{x}) = w_0 + w_{ESPN}x_{ESPN} + w_{NIKE}x_{NIKE} + w_{ADIDAS}x_{ADIDAS} + w_{NBC}x_{NBC}$$

However, logistic regression only learns the effect of each features individually rather than in a combination.

## Degree-2 Polynomial Mappings (Poly2)

A Poly2 model captures this pair-wise feature interaction by learning a dedicated weight for each feature pair.

For Poly2,

$$\phi(\pmb{w}, \pmb{x}) = w_0 + \sum\limits_{i=1}^n w_i x_i + \sum\limits_{i=1}^n \sum\limits_{j=i + 1}^n w_{h(i, j)} x_i x_j$$

In our context,

$$\phi(\pmb{w}, \pmb{x}) =  w_0 + w_{ESPN}x_{ESPN} + w_{NIKE}x_{NIKE} + w_{ADIDAS}x_{ADIDAS} + w_{NBC}x_{NBC} + w_{MALE}x_{MALE} + w_{ESPN, NIKE}x_{ESPN}x_{NIKE} + w_{ESPN, MALE}x_{ESPN}x_{MALE} + ...$$

![poly2]({{ '/images/poly2.png' | relative_url }})
*Fig. 1. Poly2 model - A dedicated weight is learned for each feature pair (linear terms ignored in diagram) (Image source: [here](http://ailab.criteo.com/ctr-prediction-linear-model-field-aware-factorization-machines/))*
<br />

However, a Poly2 model is computationally expensive as it requires the computation of all feature pair combinations. Also, when data is sparse, there might be some unseen pairs in the test set.

## Factorization Machines

FM solves this problem by learning the pairwise feature interactions in a latent space. Each feature has an associated latent vector. The interaction between two features is an inner-product of their respective latent vectors.

For FM,









## Reference

[1] Bryan McCann, et al. ["Learned in translation: Contextualized word vectors."](https://arxiv.org/abs/1708.00107) NIPS. 2017.

[2] Kevin Clark et al. ["Semi-Supervised Sequence Modeling with Cross-View Training."](https://arxiv.org/abs/1809.08370) EMNLP 2018.

---

Thank you for reading the post! See you in the next one!
