---
layout: post
title: Loss Functions
---

A loss function evaluates and helps to quantify the difference between the model's predictions against the labels. We describe MSE, logistic loss, cross-entropy loss, and contrastive learninig.

### Mean Squared Error
The mean squared error loss over $N$ examples is the following. For each example $i$, let $y_i$ denote its true label, and $\hat{y}_{i}$ denote its predicted label:
$$L = \frac{1}{N} \sum_{1 \le i \le N} (y_{i} - \hat{y}_{i})^{2}$$

### Logistic Regression Loss
The logistic regression loss function, or the binary cross-entropy loss for a single example is:
$$L = - y \text{ log}(\hat{y}) + (1 - y) \text{ log}(1 - \hat{y})$$

### Cross-Entropy for Multiple Classes
When there are multiple classes $k \in K$, let $\hat{y}_{k}$ be the predicted (softmax) probability of a particular example belonging to class $k$, then the cross-entropy loss for a single example is:
$$L = - \sum_{1 \le k \le K} y_{k} \text{ log}(\hat{y}_{k})$$

## Contrastive Learning

The aim of contrastive learning is to learn effective representation by pulling semantically close neighbors together and pushing apart non-neighbors. Assume we are given a set of sematically related (positive) example pairs $D = \{(x_i, x_{i'})\}$, for $1 \le i \le N$. 

Let $h_i$ and $h_{i'}$ denote the representations of $x_i$ and $x_{i'}$. 
Using in-batch negatives (batch size $B$) with a cross-entropy objective: 

$$\text{loss}_i = -\text{log}\frac{e^{\text{sim}(h_i, h_{i'})/\tau}}{\sum_{1 \le j \le B} e^{\text{sim}(h_i, h_{j'})/\tau}}$$

* $\tau$ is a temperature hyperparameter
* $\text{sim}(h_i, h_j)$ is cosine similarity


