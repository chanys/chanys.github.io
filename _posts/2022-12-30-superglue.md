---
layout: post
title: SuperGLUE Benchmark Dataset
---

The authors noted that system performance on their previously introduced GLUE benchmark dataset, has surpassed the level of non-expert humans. Thus, they introduced SuperGLUE, a new benchmark dataset in the paper "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems", published in 2019. 

SuperGLUE consisting of the following 8 tasks:
* Boolean Questions (**BoolQ**): Given a short passage (multiple sentences) and a question, answer Yes/No.
	* Passage: Barq's ... is owned by the Barq family but bottled by the Coca-Cola Company ..
	  Question: is barq's root beer a pepsi product
	  Answer: `No`
* CommitmentBank (**CB**): Given a premise text (multiple sentences) and a hypothesis, predict three-class entailment.
	* Text: ... I hope to see employer based... do you think we are, setting a trend?
	  Hypothesis: they are setting a trend
	  Entailment: `Unknown`
* Choice of Plausible Alternatives (**COPA**): Give a premise and a question, decide whether alternative 1 or 2 is correct.
	* Premise: My body cast a shadow over the grass. 
	  Question: What's the CAUSE for this?
	  Alternative 1: The sun was rising.    Alternative 2: The grass was cut.
	  Correct alternative: `1`
* Multi-sentence Reading Comprehension (**MultiRC**): Given a context paragraph (multiple sentences), a question about that paragraph, answer True/False for each candidate answer.
	*  Paragraph: Susan wanted to have a birthday party. .. On the day of the party, all five friends showed up. ..
	   Question: Did Susan's sick friend recover?
	   Candidate answers: 
	   	* Yes, she recovered (`T`)
	   	* No (`F`)
	   	* Yes (`T`)
	   	* No, she didn't recover (`F`)
	   	* Yes, she was at Susan's party (`T`)
* Reading Comprehension with Commonsense Reasoning Dataset (**ReCoRD**): Each example consists of a news article (multiple sentences) and a question about the article in which one entity is masked out. The system needs to predict the masked out entity, from a given list of possible entities.
	* Paragraph: (CNN) Puerto Rico on Sunday overwhelmingly voted for statehood. But Congress, .. will ultimately decide whether the status of the US commonwealth changes...
	  Query: For one, they can truthfully say, "Don't blame me, I didn't vote for them," when discussing the `<placeholder>` presidency
	  Correct Entities: `US`
* Recognizing Textual Entailment (RTE): Merged data from RTE1, RTE2, RTE3, RTE5. All datasets are converted to two-class classification: *entailment* and *not_entailment*.
	* Text: Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.
	  Hypothesis: Christopher Reeve had an accident.
	  Entailment: `False`
* Word-in-Context (WiC): Given two text snippets and a polysemous word that appears in both sentences, determine whether the two is used with the same sense in both sentences.
	* Context 1: Room and *board*.
	   Context 2: He nailed *boards* across the windows.
	   Sense match: `False`
* Winograd Schema Challenge (WSC): The authors include a version of WSC recast as NLI (natural language inference), known as WNLI. Each example consists of a sentence with a marked pronoun and noun, and the task is to determine if the pronoun refers to the noun (coreference decision).
	* Text: Mark told Pete many lies about himself, which Pete included in his book. He should have been more truthful.
	  Coreference: `False`

	| Corpus | Train | Dev | Test | Task | Classification |
	|:------:|:-----:|:---:|:----:|:----:|:--------------:|
	|BoolQ   |9427   |3270 |3245  |QA    |binary |
	|CB      |250    |57   |250   |NLI   |3 class RTE|
	|COPA    |400    |100  |500   |QA    |2 class|
	|MultiRC |5100   |953  |1800  |QA    |binary for each candidate|
	|ReCoRD  |101K   |10K  |10K   |QA    |entity from context|
	|RTE     |2500   |278  |300   |NLI   |2 class entailment|
	|WiC     |6000   |638  |1400  |WSD   |binary|
	|WSC     |554    |104  |146   |coref |binary|
