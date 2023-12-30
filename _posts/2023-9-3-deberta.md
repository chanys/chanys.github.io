---
layout: post
title: DeBERTa
---

In Oct-2021, researchers from Microsoft introduced the DeBERTa encoder model in "Deberta: decoding-enhanced BERT with disentangled attention", which performs better BERT and RoBERTa. The main contribution of DeBERTa is in introducing new/separate embeddings for relative positions. This is in contrast with the usual encoder transformers where position information is additive to the word/content embeddings at input time. Specifically, DeBERTa:
* **Keeps the content embeddings separate from the relative position embeddings**. Deberta introduces new relative-position projection matrices $W$ for the query and key.
* When calculating self-attention, besides considering *content-to-content* dot-product for self-attention score, **Deberta also includes *content-to-position* and *position-to-content***
* Due to the above, **position information is supplied to each transformer layer**. Constrast this with the usual transformer, where the position information is given as additive to the input/content embeddings only at the very beginning.

## Standard Self-Attention
The standard self-attention is formulated as follows:
$$
\begin{aligned}
Q = HW_{q} & \quad K = HW_{k} & \quad V = HW_{v}\\
A = \frac{QK^{T}}{\sqrt{d}} & \quad H_{o} = \text{softmax}(A)V
\end{aligned}
$$
* $H \in \mathbb{R}^{N \times d}$ represents the input hidden vectors
* $H_{o} \in \mathbb{R}^{N \times d}$ the output of the self-attention module
* $W_{q}, W_{k}, W_{v} \in \mathbb{R}^{d \times d}$ are the projection matrices
* $A \in \mathbb{R}^{N \times N}$ is the attention matrix
* $N$ is the sequence length, $d$ is the dimension of hidden states

## Disentangled Self-Attention
Disentangled self-attention uses separate matrices for content vs position:
$$
\begin{aligned}
Q_{c} = H W_{q,c} & \quad K_{c} = H W_{k,c} & \quad V_{c} = HW_{v,c} \\
Q_{r} = PW_{q,r} & \quad K_{r} = PW_{k,r} &\\
\end{aligned}
$$

$$
A_{i,j} = H_{i} H_{j}^{T} + H_{i} P_{j|i}^{T} + P_{i|j} H_{j}^{T} \\
H_{o} = \text{softmax}(\frac{A}{\sqrt{3d}})V
$$
* Denoting $k$ as the maximum relative distance, then $P \in \mathbb{R}^{2k \times d}$ is a newly introduced relative position embedding/table shared across all layers
* $W_{q,r}$ and $W_{k,r}$ are new projection matrices for $P$.
* Hence, disentangled attention introduced three additional sets of parameters: $W_{q,r}, W_{k,r} \in \mathbb{R}^{d \times d}$ and $P \in \mathbb{R}^{2k \times d}$. The total increase in model parameters is $2L \times d^{2} + 2k \times d$

## Enhanced Mask Decoder (EMD)
Deberta incorporates absolute positions right after all the transformer layers, but before the softmax layer for masked token prediction. That is, Deberta captures the relative positions in al the transformer layers and only uses absolute positions as complementary information when decoding the masked words.
