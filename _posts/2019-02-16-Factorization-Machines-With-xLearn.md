---
layout: post
title: Factorization Machines with xLearn
---

![xlearn_logo](images/xlearn_logo.png)

## Background

Recently, I discovered xLearn which is a high performance, scalable ML package that implements factorization machines (FM) and field-aware factorization machines (FFM). The first version of xLearn was released about 1 year ago as of writing.

FM was initially introduced and popularised by Steffen Rendle after winning 4th positions in both tracks in KDD cup 2012.

FM is a supervised learning algorithm that can be used for regression, classifications, and ranking. They are non-linear in nature and are trained using stochastic gradient descent (SGD).

FFM was later introduced and further improves FM. It was then used to win the 1st prize of three CTR competitions hosted by Criteo, Avazu, Outbrain, and also the 3rd prize of RecSys Challenge 2015.

To understand FM and FFM, we need to first understand a logistic regression model and a Poly2 model.
