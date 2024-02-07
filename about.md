---
layout: page
title: About
permalink: /about/
---

Hi! I am Dr. Yee Seng Chan and this is my blog where I write about Artificial Intelligence and Machine Learning, focusing on Natural Language Processing (NLP) and transformers such as BERT, GPT, LLaMA, etc. Throughout my career, I had been an IC, technical lead, manager, PI, and mentor, sometimes serving in multiple parallel capacities. I also have a history of internal and external collaborations (R&D organizations of DoD and National Intelligence, private organizations (e.g. MITRE), and universities).

## Short Bio
After my PhD in NLP with [Prof. Ng](https://www.comp.nus.edu.sg/~nght/), I did a postdoc with [Prof. Roth](https://www.linkedin.com/in/dan-roth-8667361/) at UIUC. Over the course of my PhD and postdoc, I published 12 main papers at premier NLP conferences (6 ACL, 3 EMNLP, 1 AAAI, 1 IJCAI, 1 COLING), 1 journal, and a few workshop papers. I am the first author in most of the conference papers. In collaboration with [David Chiang](https://www3.nd.edu/~dchiang/), our ACL paper showed for the first time that word sense information improves state-of-the-art machine translation performance, helping to settle a 10-year long debate within the academic field. My work in word sense disambiguation and MT evaluation metric had also won international benchmark competitions. 

I then worked at Raytheon BBN Technologies for 10 years. Raytheon BBN is a R&D company that was awarded the National Medal of Technology and Innovation in 2013. While at Raytheon BBN, I was serving in multiple parallel capacities for several multi-million NLP projects:
* Principal investigator in a project funded by DARPA and the Gates Foundation, where we collaborated with external private organizations and universities
* R&D lead in a challenging multilingual project where we collaborated with external universities
* The creator, architect, and main developer of a deep learning NLP R&D system called NLPLingo used in almost all of Raytheon BBN’s NLP projects, helping to bring in millions of revenue. NLPLingo leverages transformers to perform various NLP tasks such as information extraction, named entity recognition, etc.
* For 2 years running, I was also selected and served as an expert advisor and judge for Raytheon Innovation Challenge, on topics regarding “AI, ML, and Expert systems for National Security”.

To get out of my comfort zone, I left BBN to join Quattr, which is a startup focusing on applying NLP to tackle Search Engine Optimization (SEO) for corporate clients such as Coursera, Pinterest, Mcafee, etc. At Quattr, I managed the NLP/ML team, where we applied state-of-the-art NLP techniques, transformers, GPT, prompt engineering, etc. to create solutions for search intent discovery, web page topic discovery, automatic content generation, and internal linking among web pages. 

I then moved on to Elemental Cognition (EC). EC is founded by original members of the IBM Watson AI team which won the Jeopardy gameshow, and is funded by Bridgewater and other investors. At EC, I leveraged encoders, GPT, prompt engineering and various transformer models to devise NLP solutions for the biomedical and cyber security domains.

## Organization of Blog Posts
Check out [Transformer Architecture Explained](https://chanys.github.io/transformer-architecture/), then go on to the following.

### Transformer Basics
* [Subword Tokenization](https://chanys.github.io/subword-tokenization/): Byte pair encoding (BPE), WordPiece tokenization, and Unigram tokenization
* [Relative Position Embedding](https://chanys.github.io/relative-position-embedding/)
* [Rotary Position Embedding](https://chanys.github.io/rotary-position-embedding/)
* [Perplexity](https://chanys.github.io/perplexity/) is commonly used as an intrinsic evaluation of a language model
* [Likelihood based Generative Models](https://chanys.github.io/likelihood/)

### Deep Neural Network Basics
* [Unicode and UTF-8](https://chanys.github.io/unicode/)
* [Entropy and Cross-Entropy](https://chanys.github.io/entropy/)
* [Loss Functions](https://chanys.github.io/loss-functions/)
* [Activation Functions](https://chanys.github.io/activation-functions/)
* [Optimizers](https://chanys.github.io/optimizers/)
* [Linear Algebra](https://chanys.github.io/linear-algebra/)

### Encoder Language Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[BERT](https://chanys.github.io/bert/)|110M, 340M|Oct-2018|Google|Introduced masked language modeling (MLM)|
|[RoBERTa](https://chanys.github.io/roberta/)|125M, 355M|Jul-2019|UWash, Meta|Replication of BERT with more robust training
|[XLM](https://chanys.github.io/xlm/)|XLM-100 has 570M|Jan-2019|Meta|Cross-lingual model that uses translation language modeling (TLM) and MLM|
|[XLM-R](https://chanys.github.io/xlmr/)|270M, 550M|Nov-2019|Meta|Cross-lingual model that uses MLM with more robust training|
|[ELECTRA](https://chanys.github.io/electra/)|110M, 335M|Mar-2020|Stanford, Google|Introduced "replaced token detection" pretraining, that is more sample efficient than MLM|
|[DeBERTa](https://chanys.github.io/deberta/)|100M|Oct-2021|Microsoft|Keeps the content embeddings separate from the relative position embeddings|
|[DeBERTa-v3](https://chanys.github.io/deberta-v3/)|86M+98M|Mar-2023|Microsoft|Modified the replaced token detection (RTD) objective of ELECTRA, and combined it with the disentangled attention approach of DeBERTa|

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
|[U-PaLM](https://chanys.github.io/upalm/)|540B|Oct-2022|Google|Continue training PaLM with the UL2 mixture of denoising pretraining objective|
|[LLaMA-1](https://chanys.github.io/llama1/)|6.7B - 65B|Feb-2023|Meta|Trained on 1.4T tokens of publicly available texts|
|[LLaMA-2](https://chanys.github.io/llama2/)|7B - 70B|Jul-2023|Meta|Instruction fine-tuned and RLHF|
|[Mistral 7B](https://chanys.github.io/mistral/)|7B|Oct-2023|Mistral.ai|Outperforms Llama-2-7B and Llama-2-13B|
|[Zephyr-7B](https://chanys.github.io/zephyr/)|7B|Oct-2023|Hugging Face|Starts with Mistral-7B, then instruction fine-tuning, then DPO|

### Encoder-Decoder Language Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[T5](https://chanys.github.io/t5/)|11B|Oct-2019|Google|First paper to show text-to-text Transformer achieves SOTA results. Also shows span corruption works well.|
|[BART](https://chanys.github.io/bart/)|406M|Oct-2019|Meta|Similar to T5. But T5 predicts only the masked spans, whereas BART predicts the complete text|
|[mT5](https://chanys.github.io/mt5/)|13B|Oct-2020|Google|Multilingual version of the T5 model|
|[Switch](https://chanys.github.io/switch/)|3.8B|Jan-2021|Google|Based on T5, but the original dense FFN is replaced with a sparse Switch FFN layer.|
|[T0](https://chanys.github.io/t0/)|11B|Oct-2021|Hugging Face and others|Multitask fine-tuning on T5 improves zero-shot performance on unseen tasks. Performs better than GPT-3 (175B)|
|[UL2](https://chanys.github.io/ul2/)|20B|May-2022|Google|Uses a mixture of denoisers for pretraining|
|[Tk-INSTRUCT](https://chanys.github.io/tkinstruct/)|11B|Apr-2022|University of Washington and others|T5 fine-tuned on 1600+ NLP tasks with written instructions|

### Dialog Models

|Model|Size|Date|Organization|Description|
|:---|:---|:---|:---|:---|
|[TransferTransfo](https://chanys.github.io/huggingface-dialog-model/)|~117M|Jan-2019|Hugging Face|Open-source dialog model that takes on persona|
|[InstructGPT](https://chanys.github.io/chatgpt/)|175G|Sept-2020|OpenAI|Leveraged human preferences, reward modeling, and reinforcement learning to improve GPT-3 models. Predecessor of ChatGPT|
|[LaMDA](https://chanys.github.io/lamda/)|137B|Jan-2022|Google|Fine-tune decoder model for quality, safety, and groundedness|

### Sentence Transformers
* [SGPT - GPT Sentence Embeddings](https://chanys.github.io/sgpt/)
* [Sentence Embeddings Using Siamese Networks and all-mpnet-base-v2](https://chanys.github.io/sbert/)

### Large Scale Evaluation of Language Models
* [SuperGLUE](https://chanys.github.io/superglue/): (May-2019) A Stickier Benchmark for General-Purpose Language Understanding Systems
* [BEIR](https://chanys.github.io/beir-dataset/): (Apr-2021) An aggregation of 18 datasets for zero-shot evaluation of 10 IR models
* [MTEB](https://chanys.github.io/mteb-dataset/): (Oct-2022) Benchmark to evaluate sentence-embedding models
* [Holistic Evaluation of Language Models](https://chanys.github.io/holistic-evaluation/): (Nov-2022) A large scale evaluation of 30 language models over a set of 16 scenarios and 7 categories of metrics

### Improve Efficiency of Language Models
* Quantization
   * [Model Quantization](https://chanys.github.io/model-quantization/)
   * [Quantization (16-bit, 8-bit, 4-bit) and QLoRA](https://chanys.github.io/qlora/)
   * [GPTQ](https://chanys.github.io/gptq/)
* PEFT and LoRA
   * [Parameter Efficient Fine Tuning (PEFT)](https://chanys.github.io/peft/)
   * [LoRA](https://chanys.github.io/lora/)
   * [Code Example on Instruction Fine-tuning of llama2-7B using LoRA](https://chanys.github.io/flan-code/)
* Time Efficency
   * [Gradient Check-Pointing](https://chanys.github.io/gradient-checkpointing/)
   * [Key-Value Caching](https://chanys.github.io/kv-caching/)
   * [Multi-query and Grouped Multi-query Attention](https://chanys.github.io/multi-query-attention/)
* Others
   * [Knowledge Distillation and DistilBERT](https://chanys.github.io/knowledge-distillation/)

### Strategies to Improve Language Models
* [Techniques to Enable Learning Deep Neural Networks](https://chanys.github.io/techniques-to-enable-deep-nn/): Skipped Connection, Layer Normalization, RMSNorm
* [Chain-of-Thought (CoT) prompting](https://chanys.github.io/chain-of-thought/) to elicit reasoning in language models
* [Self-Consistency](https://chanys.github.io/self-consistency/) as in inference strategy
* [Big Bird Transformer for Longer Sequences](https://chanys.github.io/big-bird/)
* [Permutation Language Modeling](https://chanys.github.io/plm/)
* [Fine-Tuning Language Models for Factuality](https://chanys.github.io/llm-factuality/)
* [Direct Preference Optimization](https://chanys.github.io/dpo/)

### Search and Retrieval
* [Efficient KNN search using product quantization](https://chanys.github.io/knn/)
* [Information Retrieval Evaluation Metrics](https://chanys.github.io/ir-metrics/): MAP@K and NDCG
* [ColBERT - Passage Search via Contextualized Late Interaction over BERT](https://chanys.github.io/colbert/)
* [ColBERTv2 - Efficient Passage Retrieval](https://chanys.github.io/colbertv2/)
* [REALM - Augment Language Models with a Knowledge Retriever](https://chanys.github.io/realm/)
* RAG related
   * [Dense Passage Retrieval (DPR)](https://chanys.github.io/dpr/)
   * [Retrieval-Augmented Generation (RAG)](https://chanys.github.io/rag/)
   * [Retrieval-Augmented Dual Instruction Tuning](https://chanys.github.io/radit/)
   * [RAG-end2end for Domain Specific QA](https://chanys.github.io/rag-domain-qa/)
  
## Contact me
chanys.nlp at gmail.com
