---
layout: post
title: Relative Position Embedding
---

The idea of learning relative position embeddings was introduced in the paper "Self-Attention with Relative Position Representations" by Google, published in 2018.

## Self Attention in Transformer
In the Transformer architecture, the self-attention is a set-to-set layer, where each output element $y_i$ is computed as a weighted sum of the input embeddings $\textbf{x} = [x_1, \ldots, x_j, \ldots, x_N]$:
$$y_i = \sum_{j=1}^{N} w_{ij} (x_j V)$$
where $V$ is a learnable parameter matrix implemented as a linear layer. 

Each weight coefficient $w_{ij}$ represents the affinity or similarity between input embeddings $x_i$ and $x_j$, and is calculated by performing softmax over dot-products: $w_{ij} = \frac{\text{exp}(e_{ij})}{\sum_j \text{exp}(e_{ij})}$

$$e_{ij} = \frac{(x_i Q)^{T}(x_j K)}{\sqrt{d_x}}$$
where $d_x$ is the length (dimensionality) of vector $x$.

## Relative Position
As shown above, the self-attention layer has no access to the sequential structure of the inputs. Thus, we encode the position index of each input as position embeddings. One way to do this is to use the absolute position of each token to index into a learnable embeddings table. Another approach is to use relative position embeddings. To illustrate, imagine that we use the following learnable *relative position embedding* (RPE) table:
|Index|Interpretation|
|:-:|:-:|
|0|for token at position $i-3$|
|1|for token at position $i-2$|
|2|for token at position $i-1$|
|3|for token at position $i$|
|4|for token at position $i+1$|
|5|for token at position $i+2$|
|6|for token at position $i+3$|

Then when given the sentence "The brown fox jumps over the box":
* With the current word being "jumps", we will be indexing into the RPE table for each token in the sentence as: $[0, 1, 2, 3, 4, 5, 6]$.
* With the current word being "fox", we will be indexing into the RPE table for each token in the sentence as: $[1, 2, 3, 4, 5, 6, 6]$. Notice that we use the same RPE index $6$ for both the tokens "the" and "box". This is because the maximum relative position we will consider is clipped to a maximum window size $k$, where we have defined here to be $k=3$.
* If the current word is "box", then we will be indexing into the RPE table for each token in the sentence as: $[0, 0, 0, 0, 1, 2, 3]$, where we have clipped the window from the left.

There are two reasons for defining a maximum relative position window size:
* Neighbouring words are more important, so it might very well be the case that position information beyond a certain window size is not useful.
* Clipping to a maximum window size also enables the model to generalize to sequence lengths beyond that seen during training.

### Additive Modifications to Self Attention
To incorporate relative position embedding information, we make the following additive changes to the self-attention calculation:
$$y_i = \sum_{j=1}^{N} w_{ij} (x_j V + A_{ij})$$
$$e_{ij} = \frac{(x_i Q)^{T}(x_j K + B_{ij})}{\sqrt{d_x}}$$
Notice that we use two separate learnable RPE weight matrices $A$ and $B$.
