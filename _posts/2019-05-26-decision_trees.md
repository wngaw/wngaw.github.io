---
layout: post
comments: true
date: 2019-05-26 12:00:00
title: Decision Trees
tags: supervised-learning
---

Decision tree is one of the most interpretable machine learning model. However, it tends to overfit when learning the decision boundaries of the training data. Hence, decision trees are usually used as the base learner of ensembling methods such as Gradient Boosting and Random Forest. These ensemnbling methods have seen extreme success in the data science world. XGBoost, LightGBM, and Catboost are some of the popular gradient boosting methods that are widely used in competitions. But before getting there, we need to first understand how decision tree works as the base learner.

## What is Decision Tree?

Decision Tree is a supervised learning algorithm that is capable of learning non-linear hypothesis through series of if-else statements. Compared to regressions and neural networks, decision tree is a much simpler learning algorithm. Decision tree can be used for both regression and classification problems.

## Hypothesis

The hypothesis of a decision tree is represented by a series of if-else statements that outputs the probability of a class for classification and a numerical number for regression.

## Context

Assume we have the following dataset where we want to predict the a fruit label given the color and diameter of it:

| **Dataset** | **Color** | **Diameter**  | **Label** |
| :---: | :---: | :---: | :---: |
| TRAIN | GREEN | 3 | APPLE |
| TRAIN | YELLOW | 3 | APPLE |
| TRAIN | RED | 1 | GRAPE |
| TRAIN | RED | 1 | GRAPE |
| TRAIN | YELLOW | 3 | LEMON |

## An intuitive explanation

The learning algorithm starts with the root node where it will receive the entire training set. Since CART is used as the learning algorithm for a classifcation problem, Gini Impurity is the metric we calculate.

The algorithm will start by calculating the Gini Impurity at the root node, which turns out to be 0.64. Then it will iterate over every value to create multiple true-false question where we can partition the data.

Possible questions are:

- Is the diameter >= 3?
- Is the diameter >=1?
- Is the color green?
- Is the color yellow?
- Is the color red?

At each iteration, Gini Impurity will be calculated for both child nodes. A weighted average Gini Impurity will be calculated based on the number of data points in each child nodes. This is because we care more about the Gini Impurity of a child node with more data points.

The information gain is then calculated by taking the difference between the Gini Impurity at the root node and the weighted average Gini Impurity at the child nodes. The best question is the one with the highest information gain.

For example, using the question $$\textit{Is the color green}$$, the weighted average Gini Impurity of the child nodes is 0.5 $$[(\frac{4}{5} * 0.62) + (\frac{1}{5} * 0)]$$, giving us an information gain of 0.14 $$(0.64 - 0.5)$$

![is_color_green]({{ '/images/is_color_green.png' | relative_url }})
<br />
*Fig. 1. Is Color Green? - (Image source: [here](https://www.youtube.com/watch?v=LDRbO9a6XPU))*
<br />

The information gain is then calculated for every question:

| **Question** | **Information Gain** |
| :---: | :---: |
| Diameter >= 3? | 0.37 |
| Diameter >=1? | 0 |
| Color == Green? | 0.14 |
| Color == Yellow? | 0.17 |
| Color == Red? | 0.37 |

Here we can see that there is a tie between $$\textit{Is diameter >= 3?}$$ and $$\textit{Is the color red?}$$. Hence, we will choose the first question to be the best one. We then split the root node using this question.

Next, we will then repeat the process of splitting on the $$\textit{true}$$ branch. Here, we found out that $$\textit{Is color yellow?}$$ is the best question at this child node. We then split the child node using this question.

![is_color_yellow]({{ '/images/is_color_yellow.png' | relative_url }})
<br />
*Fig. 1. Is Color Yellow? - (Image source: [here](https://www.youtube.com/watch?v=LDRbO9a6XPU))*
<br />

Continuing the process, we found out that the information gain is $$0$$ since there is no more question to be asked as both data points on the $$\textit{true}$$ branch have identical features of $$Color == Yellow$$ and $$diameter == 3$$. Hence, this node becomes a leaf node which predicts Apple at 50% and Lemon at 50%.

Since the node at the end of the true branch has turned into a leaf node, we will then go over to the false branch to continue splitting. Over here, we found out that there is also no information gain since there is only one data point. Hence, this turns into a leaf node as well. At this point, $$Is the color yellow?$$ becomes a decision node as it is where a decision has to be made.

![is_color_yellow_leaf_nodes]({{ '/images/is_color_yellow_leaf_nodes.png' | relative_url }})
<br />
*Fig. 1. Is Color Yellow? Leaf Nodes - (Image source: [here](https://www.youtube.com/watch?v=LDRbO9a6XPU))*
<br />

Now, we return to the root node to split for the false branch. Similarly, we realise that there are no further questions to be asked that can give us information gain. Hence, this turns into a leaf node as well. At this point, $$Is the diameter >= 3?$$ becomes a decision node as well.

![is_diameter_more_than_equal_3]({{ '/images/is_diameter_more_than_equal_3.png' | relative_url }})
<br />
*Fig. 1. Is Diameter >= 3? - (Image source: [here](https://www.youtube.com/watch?v=LDRbO9a6XPU))*
<br />

And finally, the decision tree is fully built and ready to be used to predict new data points!

Although the entire process seems complicated, it can actually be represented by a concise block of code as follows:

```python
def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)
```

## Types of Decision Trees

There are two kinds of decision trees. Namely, classification tree and regression tree. In a classification tree, the predicted class is obtained by selecting the class with the highest probability at the leaf node where if-else criteria are met. In a regression tree, the predicted value is the average response of observations falling in that leaf node.

## Decision Tree Learning Algorithms

Under the hood, the series of if-else statements of a decision tree can be generated using several kinds of learning algorithms:

- *[CART (Classification and regression tree)](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)*: uses *[Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)* for classification and *[Variance](https://en.wikipedia.org/wiki/Decision_tree_learning#Variance_reduction)* for regression as metrics to decide split points
- *[C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm)*: uses information gain to decide split points.
- Others such as ID3 and C5.0

## Metrics

For decision trees, different algorithms use different metrics to decide on the best split points. To complicate that further, within each algorithm, classification and regression use different metrics as well.

For classification, Entropy and Gini Impurity are common metrics to evaluate at each split points when generating a decision tree.

For regression, Variance is usually evaluated at each split point when generating a decision tree.

### Entropy

This metric measures the amount of disorder in a data set, with a lowest value of 0. In laymen terms, it is how surprising a randomly selected item from the set is. If the entire data set were As, you will never be surprised to see an A, so entropy is 0.

The formula for Entropy is as follows:

$$H(X) = \sum\limits_{i=1}^n p(x_i)log_2(\frac{1}{p(x_i)} )$$

### Gini Impurity

This metric ranges between 0 and 1, where a lower value is better as it indicates less uncertainty at a node. It represents the probability of being incorrect if you randomly pick an item and guess its label.

The formula for Gini Impurity is as follows:

$$I_G(i) = 1 - \sum\limits_{j=1}^m f(i,j)^2$$

### Variance

This metric measures how much a list of numbers varies from the average value. It is calculated by averaging the squared difference of every number from the mean.

The formula for Variance is as follows:

$$\sigma^2 = \frac{1}{N} \sum\limits_{i=1}^N (x_i - \bar{x})^2$$

## Reference

[1] Josh Gordon [Machine Learning Recipes #8](https://www.youtube.com/watch?v=LDRbO9a6XPU)

[2] Toby Segaram [Programming Collective Intelligence](http://shop.oreilly.com/product/9780596529321.do)
---

Thank you for reading! See you in the next post!
