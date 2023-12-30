---
layout: post
title: Fine-Tuning Language Models for Factuality
---

A recent paper "Fine-tuning language models for factuality" on 11/14/2023, shows that it is possible to fine-tune language models to improve factuality generation. In particular, the authors generated their own factuality datasets and used the recently introduced direct preference optimization (DPO) method to fine-tune LLMs.

## Approach

First, generate a dataset consisting of examples $\{x, y_{w}, y_{l}\}$, where $x$ denotes an input prompt, $y_{w}$ denotes the more factual generation, while $y_{l}$ denotes the less factual generation:
* For each input text/prompt, we sample/generate $n$ candidate responses with temperature 1.0.
* Compute the factuality score for each response.
* For all $n \choose 2$ pairs of responses (for each prompt), choose the response with the higher factuality score as the preferred response $y_{w}$.

### Facuality Score Calculation
To calculate the factuality score of a response text, the authors proposed two approaches. The approach which performs better in their experiments (i.e. resulting in a higher performing factuality fine-tuned LM), leveraged the FactScore algorithm ("Factscore: fine-grained atomic evaluation of factual precision in long form text generation", Min et al. 2023). To evaluate the factuality of a given piece of text:
* First extract a list of the atomic claims present in the text using GPT-3.5.
* For each atomic claim, use a smaller more efficient model (e.g. Llama-1-7B that has been fine-tuned for fact-checking) to determine if the claim is supported by reference text (e.g. Wikipedia articles).
* The input text's factuality score is the fraction of the atomic claims that are estimated to be supported by the reference text.

### Experiments
The authors performed experiments on biographies and medical QA. They showed that their approach of fine-tuning for factuality preferences using the DPO algorithm can improve upon models such as Llama-2-7b-Chat (the "Chat" models were already instruction-tuned and RLHF tuned). As evaluation metric, they also consulted GPT-4 and performed some human evaluation. 
