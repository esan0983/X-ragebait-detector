# X Ragebait Detector & Generator (WIP)

A data science & machine learning project, with most of its inspiration taken from this [paper](https://arxiv.org/html/2609.02262v1). There are two goals:  
1. An end-to-end data pipeline, ending with a machine learning algorithm that can classify posts into ragebait, controversial (but not necessarily ragebait), or none of the two.
2. If ragebait/controversy detection is successful, I will develop an LLM that generates ragebait/controversial posts that will farm engagement, trained from a huge batch of posts that were classfied by the Phase 1 model as ragebait or controversial.

## Data Pipeline
### Collection
I used the twikit package. Since I don't want to get banned, I want to do this right with a robust architecture and proper API calling, so with a detailed enough prompt, I let Claude generate a data collection script. The script handles rate limits well with randomized wait times and exponential back-offs as a response to 429s.  

These are the following criteria for data collection:
* Posted on January to August 2026
* Likes + replies + reposts >= 50
* In between 50 and 280 characters  

A large table of keywords was constructed to increase the chances of collecting ragebait/controversial posts. There is a huge trade-off: a lot of the statistical analysis (especially the studies made by the Japanese paper) will be subject to selection bias. For example, we can't construct a word cloud anymore as the results will essentially just duplicate the keyword table. In return, we will have a lot of training data for the LLM model in a significantly shorter period of time, and our machine learning model will less likely struggle with class imbalance.

### Cleaning
* Removed post duplicates
* Removed posts that include very recent events since I don't want the LLM to be outdated

### Annotation
It's very tedious to annotate the data on a large scale. Hence, we will use the GPT 5.6 Luna model to perform batched annotation. There will be three categories:

#### 1. Ragebait
* **Definition:** A post whose primary discernible function is to provoke negative emotion (anger, outrage, contempt, indignation) in order to generate engagement (replies, reposts, quote-posts), regardless of whether the author sincerely holds the stated view.
* **Necessary Condition:** The text alone, without external context, must be sufficient to provoke a negative emotional reaction in a plausible reader.
* **Non-Necessary Conditions:** (May or may not be present; presence increases confidence but is not required) Offensive language, hate speech, slurs, toxic phrasing, insults.
* **Distinguishing Signal:** Statements that are extreme, absolute, or overgeneralized to a degree that suggests performative provocation rather than a genuine, considered opinion (e.g., sweeping negative claims about a demographic group stated as flat fact, with no qualification or support).
* **Labeling Rule:** Label as Ragebait if the post's content and phrasing appear engineered to provoke anger/outrage as their main function, independent of topic.

#### 2. Controversial Non-Ragebait
* **Definition:** A post that addresses a topic prone to disagreement (e.g., politics, religion, health, social issues) but is not primarily engineered to provoke negative emotion.
* **Distinguishing Signals:**
    * Tone is measured, polite, or open-minded rather than inflammatory.
    * Claims are qualified, explained, or supported (e.g., reasoning, evidence, acknowledgment of nuance/counterpoints) rather than stated as absolute, unqualified fact.
    * The post reads as a genuine expression of belief rather than a provocation engineered for reaction.
* **Labeling Rule:** Label as Controversial Non-Ragebait if the topic is contentious AND the tone/construction indicates sincere argumentation rather than provocation.

#### 3. None of the Above
* **Definition:** Applies when a post does not meet the criteria for Ragebait or Controversial Non-Ragebait. Includes two specific cases:
    * **Case A — Incomplete/context-dependent post:** The post's meaning or rhetorical force depends on an attached image, video, link, or other external context not present in the text itself, such that the text alone reads as incomplete or under-specified. Do not label such posts as Ragebait even if the attachment (which you cannot see) would plausibly make it provocative. Label based on the standalone text only.
    * **Case B — Positive-emotion clickbait:** The post is constructed to drive engagement but targets positive or neutral emotions (humor, excitement, curiosity, harmless jokes) rather than negative/anger-based emotions. This includes posts that could later cause outrage once external context becomes known, but do not do so based on the standalone text alone.
* **Labeling Rule:** Label as None of the Above if the post is either context-dependent for its provocative effect (Case A) or is engagement-oriented but not negative-emotion-oriented (Case B), or otherwise fails to meet the definitions of categories 1 and 2.

### Human Validation
Two humans and GPT 5.6 Luna will annotate 500 posts to check for agreement.

## Commit Notes (9/16)
* Fixed slow collection problem

## Sources
1. From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X. (n.d.). https://arxiv.org/html/2609.02262v1