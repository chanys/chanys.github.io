---
layout: post
title: BEIR Dataset for Zero-shot Evaluation of IR Models
---

The BEIR information retrievel (IR) dataset was introduced in the paper "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models", published in 2021. The paper puts together 18 publicly available datasets, to evaluate 10 IR models. The task is: given a query, retrieve the relevant passages/documents as a ranked list. Evaluate using nDCG@10. 

#### Datasets
The dataset domains and datasets are:
* **Bio-medical IR**: Given a biomedical scientific query, retrieve bio-medical documents as output.
 * TREC-COVID: An ad-hoc search challenge based on the CORD-19 dataset containing scientific articles related to the COVID-19 pandemic.
 * NFCorpus: Natural language queries from NutritionFacts, and medical documents from PubMed as target corpus.
   * BioASQ: Biomedical semantic QA. Articles from PubMed as target corpus.  
* **Open domain QA**:
   * Natural Questions [KPR+]: Given a Google search query, return relevant Wikipedia passages.
   * HotpotQA: Each question requires reasoning over multiple Wikipedia passages to get the answer.
   * FiQA-2018: Financial domain opinion based QA, mined from StackExchange posts under the Investment topic.
* **Tweet retrieval**: 
   * Signal-1M related tweets: Given a news article title, retrieve relevant tweets.
* **News retrieval**:
   * TREC-NEWS: Given a news headline, retrieve relevant news articles that provide important context or background information.
   * Robust04: TREC task focusing on poorly performing topics, where queries are single sentences.
* **Argument retrieval**:
   * ArguAna Counterargs corpus: Given an argument, retrieve best counter-argument. Scraped from online debate portal.
   * Touche-2020: A conversational argument retrieval task.
* **Duplicate question retrieval**: a given query is the input, and duplicate questions are the output.
   * CQADupStack: A query is a title + body. From StackExchange subforums. 
   * Quora: Duplicate questions detection from Quora.
* **Entity retrieval**: retrieve Wikipedia pages (title + abstract) to entities mentioned in the query.
    * DBPedia-Entity-v2: Given queries containing entities, retrieve entities from English DBpedia.
* **Citation prediction**:
   * SCIDOCS: Given a paper title, retrieve cited papers from a list of 5 cited and 25 uncited papers.
* **Fact checking**: sentence-level claim as input, and the relevant document passage verifying the claim as output.
   * FEVER [TVC+]: claims verified against introductory sections of Wikipedia pages.
   * Climate-FEVER: Climate claims verified against Wikipedia articles.
   * SciFact: Verifies scientific claims against scientific paper abstracts. 

#### Evaluation
The following IR models perform well:
* BM25 is a strong baseline.
* DocT5query: First train a T5 (base) sequence-to-sequence model (on MS MARCO dataset) to generate queries when given a document. Then concatenate these generated queries (up to 40) to each original document in the retrieval set. Then index these expanded documents using BM25. This improves **1.6%** over BM25.
* COLBERT (refer to the summary on COLBERT-v2)
