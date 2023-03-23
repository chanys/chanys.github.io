---
layout: post
title: Loss Functions
---

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
${x^+}_{i}$

The aim of contrastive learning is to learn effective representation by pulling semantically close neighbors together and pushing apart non-neighbors. Assume we are given a set of sematically related (positive) example pairs $D = \{(x_i, {x^+}_{i})\}_{i=1}^{N}$. Let $\mathbf{h}_i$ and $\mathbf{h}_{i}^{+}$ denote the representations of $x_i$ and $x_{i}^{+}$. Using in-batch negatives (batch size $B$) with a cross-entropy objective: 
$$\text{loss}_i = -\text{log}\frac{e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_{i}^{+})/\tau}}{\sum_{j=1}^{B} e^{\text{sim}(\mathbf{h}_i, \mathbf{h}_{j}^{+})/\tau}}$$
* $\tau$ is a temperature hyperparameter
* $\text{sim}(\mathbf{h}_i, \mathbf{h}_j)$ is the cosine similarity $\frac{\mathbf{h}_{i}^{T}\mathbf{h}_j}{||\mathbf{h}_i|| \cdot ||\mathbf{h}_j||}$
