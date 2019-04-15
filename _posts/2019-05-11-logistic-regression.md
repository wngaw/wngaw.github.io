---
layout: post
comments: true
date: 2019-04-07 12:00:00
title: Logistic Regression
tags: supervised-learning
---

Logistic regression is one of most widely used classification learning algorithms in various fields, including machine learning, most medical fields, and social sciences. Similar to the post on linear regression, I will go into the mechanics behind logistic regression in order for us to gain a deeper understanding of it.

## What is logistic regression?

Logistic regression is a supervised learning algorithm that outputs values between zero and one.

## Hypothesis

The objective of a logistic regression is to learn a function that outputs the probability that the dependent variable is one for each training sample.

To achieve that, a sigmoid / logistic function is required for the transformation.

A sigmoid function is as follows:

$$
\begin{align}
f(x) = \frac{1}{1 + e^{-x}}
\end{align}
$$

Visually, it looks like this:

![sigmoid_function]({{ '/images/sigmoid_function.png' | relative_url }})
<br />
*Fig. 1. Sigmoid Function - (Image source: [here](https://en.wikipedia.org/wiki/Sigmoid_function))*
<br />


This hypothesis is typically represented by the following function:

$$
\begin{align}
&\text{h}_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}
\end{align}
$$

where

- $$\theta$$ is a vector of parameters that correponds to each independent variable
- $$x$$ is a vector of independent variables

## Cost function

The cost function for logistic regression is derived from statistics using the [principle of maximum likelyhood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), which allows efficient identification of parameters. In addition the covex properrt of the cost function allow gradient descent to work effectively.

$$
\begin{align}
J(\theta) =  \frac{1}{m} \sum\limits_{i=1}^m (h_\theta(x^i) - y^i)^2 \\
(h_\theta(x^i) - y^i)^2 =
\left \{
\begin{array}{l}
-log(h_\theta(x^i)) \ \text{if} \ y^i = 1\\
-log(1 - h_\theta(x^i)) \ \text{if} \ y^i = 0
\end{array}
\right .
\end{align}
$$

where

- $$i$$ is one of the training samples
- $$h_\theta(x^i)$$ is the predicted value for the training sample $$i$$
- $$y^i$$ is the actual value for the training sample $$i$$

To understand the cost function, we can look into each of the two components in isolation:

Suppose $$y^i = 1$$:

- if $$h_\theta(x^i) = 1 $$, then the prediction error = 0
- if $$h_\theta(x^i) = 0 $$, then the prediction error approaches infinity
These two scenarios are represented by the blue line in Figure 2 below.

Suppose $$y^i = 0$$:

- if $$h_\theta(x^i) = 0 $$, then the prediction error = 0
- if $$h_\theta(x^i) = 1 $$, then the prediction error approaches infinity
These two scenarios are represented by the red line in Figure 2 below.

![cost_function]({{ '/images/logistic_regression_cost_function.png' | relative_url }})
<br />
*Fig. 2. Logistic Regression Cost Function - (Image source: [here](https://cognitree.com/blog/logistic-regression/))*
<br />

The logistic regression cost function can be further simplified into a one line equation:

$$
\begin{align}
J(\theta) =  \frac{1}{m} \sum\limits_{i=1}^m (h_\theta(x^i) - y^i)^2 \\
(h_\theta(x^i) - y^i)^2 = -y^i log(h_\theta(x^i)) - (1-y^i)log(1- h_\theta(x^i)
\end{align}
$$

The overall objective is to minimise the cost function by iterating through different values of $$\theta$$.

$$
\begin{align}
\underset{\theta_0,\ \ldots,\ \theta_n}{\min} J(\theta_0,\ \ldots,\ \theta_n)
\end{align}
$$

## Gradient Descent

The gradient descent algorithm is as follows:

$$
\begin{align}
\text{repeat until convergence} \ \left \{
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0,\ \ldots,\ \theta_n) \right \}\\
\end{align}
$$

where

- values of $$j = 0, 1,\ \ldots,\ n$$
- $$\alpha$$ is the learning rate

Note: The gradient descent algorithm is identical to linear regression's.

## Advanced Optimization Algorithms

Gradient descent is not the only algorithm that can minimize the cost function.

- Conjudate gradient
- BFGS
- L-BFGS

Advantages:
- Do not need to pick learning rate $$\alpha$$
- Converges faster

Disadvantages:
- More complex

## Multiclass Classification

One-vs-rest is a method where you turn a n-class classification problem into a nth seperate binary classification problem.

To deal with a multiclass problem, we then train a logistic regression binary classifier $$h_\theta^i (x)$$ for each class $$i$$ to predict the probability that y = i.

The prediction output for a given new input $$x$$ will be chosen based on the classifier that has the highest probability.

$$
\underset{i}{\min} h_\theta^i (x)
$$

where
- $$h_\theta^i (x)$$ is the $$i^{th}$$ binary classifier

## Reference

[1] Andrew Ng [Coursera: Machine Learning](https://www.coursera.org/learn/machine-learning)

---

Thank you for reading! See you in the next post!
