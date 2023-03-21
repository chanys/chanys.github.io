---
layout: page
title: About
permalink: /about/
---

Hi! I am Dr. Yee Seng Chan and this is my blog where I write about Artificial Intelligence and Machine Learning, focusing on Natural Language Processing (NLP) technologies such as large language models, i.e Transformers such as ChatGPT, LaMDA, BERT, GPT, etc.

## Short Bio
I did a PhD in NLP, followed by a Post-doc at UIUC. I then worked at Raytheon BBN Technologies for many years. BBN is a R&D company that was awarded the National Medal of Technology and Innovation in 2013. To get out of my comfort zone, I left BBN to join a startup, which focused on applying NLP to tackle Search Engine Optimization (SEO).
 
While at BBN, I was the Principal Investigator and Engineering Lead in multiple multi-million NLP programs. I was also the proud creator, architect, and primary developer of BBN's deep learning R&D system that was used in almost all of BBN's NLP programs. The system that I developed builds upon Transformer based models to perform various information extraction tasks such as event extraction, relation extraction, named entity recognition, etc. For 2 years running, I was also selected and served as an expert advisor and judge for Raytheon Innovation Challenge, on topics regarding "AI, ML, and Expert systems for National Security". 

In my role at the SEO startup, I managed the NLP, ML, and Data Science team, where we applied state-of-the-art NLP techniques to create solutions for search intent, content generation, etc.

## Organization of Blog Posts
The following is an organization of my blog posts:

### Deep Neural Network Basics
* [Transformer Architecture Explained](https://chanys.github.io/transformer-architecture/)
* [Unicode and UTF-8](https://chanys.github.io/unicode/)
* [Subword Tokenization](https://chanys.github.io/subword-tokenization/): Byte pair encoding (BPE), WordPiece tokenization, and Unigram tokenization
* [Relative Position Embedding](https://chanys.github.io/relative-position-embedding/)
* [Entropy and Cross-Entropy](https://chanys.github.io/entropy/)
* [Perplexity](https://chanys.github.io/perplexity/) is commonly used as an intrinsic evaluation of a language model
* [Likelihood based Generative Models](https://chanys.github.io/likelihood/)

### Dialog Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[TransferTransfo](https://chanys.github.io/huggingface-dialog-model/)|~117M|Jan-2019|Hugging Face|Open-source dialog model that takes on persona|
|[InstructGPT](https://chanys.github.io/chatgpt/)|175G|Sept-2020|OpenAI|Leveraged human preferences, reward modeling, and reinforcement learning to improve GPT-3 models. Predecessor of ChatGPT|
|[LaMDA](https://chanys.github.io/lamda/)|137B|Jan-2022|Google|Fine-tune decoder model for quality, safety, and groundedness|

### Encoder Language Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[BERT](https://chanys.github.io/bert/)|110M, 340M|Oct-2018|Google|Introduced masked language modeling (MLM)|
|[RoBERTa](https://chanys.github.io/roberta/)|125M, 355M|Jul-2019|UWash, Meta|Replication of BERT with more robust training
|[XLM](https://chanys.github.io/xlm/)|XLM-100 has 570M|Jan-2019|Meta|Cross-lingual model that uses translation language modeling (TLM) and MLM|
|[XLM-R](https://chanys.github.io/xlmr/)|270M, 550M|Nov-2019|Meta|Cross-lingual model that uses MLM with more robust training|
|[ELECTRA](https://chanys.github.io/electra/)|110M, 335M|Mar-2020|Stanford, Google|Introduced "replaced token detection" pretraining, that is more sample efficient than MLM|
{:mbtablestyle}

### Decoder Language Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[GPT-1](https://chanys.github.io/gpt1/)|117M|Jun-2018|OpenAI|Demonstrated pretraining Transformers enable effective transfer learning to downstream NLP tasks|
|[GPT-2](https://chanys.github.io/gpt2/)|1.5B|Feb-2019|OpenAI|Focused on zero-shot in-context prompting|
|[GPT-3](https://chanys.github.io/gpt3/)|175B|Jul-2020|OpenAI|Focused on one-shot and few-shot prompting|
|[MPNet](https://chanys.github.io/mpnet/)|110M|Apr-2020|Nanjing University and Microsoft|Fused and improved on MLM + PLM for pretraining|
|[FLAN](https://chanys.github.io/flan/)|137B|Sept-2021|Google|Shows that multitask fine-tuning of LaMDA-PT improves zero-shot generalization to new tasks|
|[GLaM](https://chanys.github.io/glam/)|1.2T|Dec-2021|Google|Decoder-only language model that does conditional computation using mixture of experts (MoE)|
|[Chinchilla](https://chanys.github.io/chinchilla/)|70B|Mar-2022|DeepMind|Shows that number of training tokens should scale equally with model size. Outperforms GPT-3 (175B)|
|[PaLM](https://chanys.github.io/palm/)|540B|Apr-2022|Google|Likely the best decoder-only pretrained model at time of publication|
|[FLAN-PaLM](https://chanys.github.io/flan-palm/)|540B|Oct-2022|Google|Multitask instruction fine-tuning on PaLM. Likely the best decoder-only model at time of publication, but probably under-trained|

### Encoder-Decoder Language Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[T5](https://chanys.github.io/t5/)|11B|Oct-2019|Google|First paper to show text-to-text Transformer achieves SOTA results. Also shows span corruption works well.|
|[BART](https://chanys.github.io/bart/)|406M|Oct-2019|Meta|Similar to T5. But T5 predicts only the masked spans, whereas BART predicts the complete text|
|[mT5](https://chanys.github.io/mt5/)|13B|Oct-2020|Google|Multilingual version of the T5 model|
|[Switch](https://chanys.github.io/switch/)|3.8B|Jan-2021|Google|Based on T5, but the original dense FFN is replaced with a sparse Switch FFN layer.|
|[T0](https://chanys.github.io/t0/)|11B|Oct-2021|Hugging Face and others|Multitask fine-tuning on T5 improves zero-shot performance on unseen tasks. Performs better than GPT-3 (175B)|

### Sentence Transformers
* [SGPT - GPT Sentence Embeddings](https://chanys.github.io/sgpt/)
* [Sentence Embeddings Using Siamese Networks and all-mpnet-base-v2](https://chanys.github.io/sbert/)

### Large Scale Evaluation of Language Models
* [SuperGLUE](https://chanys.github.io/superglue/): (May-2019) A Stickier Benchmark for General-Purpose Language Understanding Systems
* [BEIR](https://chanys.github.io/beir-dataset/): (Apr-2021) An aggregation of 18 datasets for zero-shot evaluation of 10 IR models
* [MTEB](https://chanys.github.io/mteb-dataset/): (Oct-2022) Benchmark to evaluate sentence-embedding models
* [Holistic Evaluation of Language Models](https://chanys.github.io/holistic-evaluation/): (Nov-2022) A large scale evaluation of 30 language models over a set of 16 scenarios and 7 categories of metrics

### Improve Efficiency of Language Models
* [Knowledge Distillation and DistilBERT](https://chanys.github.io/knowledge-distillation/)
* [Model Quantization](https://chanys.github.io/model-quantization/)

### Strategies to Improve Language Models
* [Techniques to Enable Learning Deep Neural Networks](https://chanys.github.io/techniques-to-enable-deep-nn/): Skipped Connection and Layer Normalization
* [Chain-of-Thought (CoT) prompting](https://chanys.github.io/chain-of-thought/) to elicit reasoning in language models
* [Self-Consistency](https://chanys.github.io/self-consistency/) as in inference strategy
* [Big Bird Transformer for Longer Sequences](https://chanys.github.io/big-bird/)
* [REALM - Augment Language Models with a Knowledge Retriever](https://chanys.github.io/realm/)
* [Permutation Language Modeling](https://chanys.github.io/plm/)

### Search
* [Efficient KNN search using product quantization](https://chanys.github.io/knn/)
* [Information Retrieval Evaluation Metrics](https://chanys.github.io/ir-metrics/): MAP@K and NDCG
* [ColBERT - Passage Search via Contextualized Late Interaction over BERT](https://chanys.github.io/colbert/)
* [ColBERTv2 - Efficient Passage Retrieval](https://chanys.github.io/colbertv2/)


## Contact me
chanys.nlp at gmail.com
