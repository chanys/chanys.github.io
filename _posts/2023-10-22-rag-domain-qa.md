---
layout: post
title: RAG-end2end for Domain Specific QA
---

When the original RAG model was introduced, the passage encoding and indexing are fixed, since re-encoding the external knowledge base passages during training is expensive. Despite this, the original RAG model performed well when evaluated on Wikipedia associated evaluation sets, since the dense passage retriever (DPR) used there had been trained on Wikipedia-based datasets. This paper, "Improving the domain adapation of retrieval augmented generation (RAG) models for open domain question answering" from 10/6/2022 in contrast, explores **using RAG for domain-specific QA**.

Specifically, the paper proposed the following:
* RAG-end2end: update all RAG components during training, including the external knowledge base, the DPR model, and the seq2seq BART model.
* The paper proposed an auxiliary training signal (via generating a concise and factual statement about a document) to help the model learn more domain specific knowledge.

## RAG Retriever and Generator Training
**DPR retriever**: this consists of a BERT-based passage encoder $E_{P}$ and a separate BERT-based question encoder $E_{Q}$, using the CLS token embeddings as representations. The similarity between a question $q$ and passage $p$ is calculated by taking the dot product: $\text{sim}(p, q) \approx E_{Q}(q)^{T} E_{P}(p)$.

**RAG generator**: this is a seq2seq BART.

**RAG Loss**: The original RAG model minimizes the following token-loss:
$$p_{\text{token-loss}}(y|x) = \prod_{i}^{n} \sum_{z \in \text{top}-k P(\cdot|x)} P_{\text{DPR}}(z|x) P_{\theta}(y_{i} | x, z, y_{1:i-1})$$

### End-to-End Retriever Training
For true end-to-end training of the retriever, the passages will have to be re-encoded and re-indexed during the training process. The authors of this work proposed to use two asynchronous processes to re-encode and re-index teh external KB, and these two processes run independently to the main training loop.

### Auxiliary Training Signal
The authors also proposed reconstructing the input query as an auxiliary training signal. I.e. given an input question $q$, first retrieve the most similar set of passages. Then ask the BART model to regenerate $q$ using the retrieved passages.

## Experiments
The authors experimented on three domain specific datasets: COVID-19 QA, News QA, and Conversation QA. For instance, for the COVID-19 QA, they used 5,000 full-text scientific articles from the CORD-19 dataset, which is split into 250K 100-word passages. For the statement/question reconstuction auxiliary signal, they us sentences from the abstract section of research articles as reconstruction targets.
