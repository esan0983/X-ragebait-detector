# X Ragebait Detector & Generator (WIP)

A data science & machine learning project, with most of its inspiration taken from this [paper](https://arxiv.org/html/2609.02262v1). There are two goals:  
1. An end-to-end data pipeline, ending with a machine learning algorithm that can classify posts into ragebait, controversial (but not necessarily ragebait), or none of the two.
2. If ragebait/controversy detection is successful, I will develop an LLM that generates ragebait/controversial posts that will farm engagement, trained from a huge batch of posts that were classfied by the Phase 1 model as ragebait or controversial.

## Data Collection
I used the twikit package. Since I don't want to get banned, I want to do this right with a robust architecture and proper API calling, so with a detailed enough prompt, I let Claude generate a data collection script. The script handles rate limits well with randomized wait times and exponential back-offs as a response to 429s.  

These are the following criteria for data collection:
* Posted within August 2026
* Likes + replies + reposts >= 500
* In between 50 and 280 characters  

A large table of keywords was constructed to increase the chances of collecting ragebait/controversial posts. There is a huge trade-off: a lot of the statistical analysis (especially the studies made by the Japanese paper) will be subject to selection bias. For example, we can't construct a word cloud anymore as the results will essentially just duplicate the keyword table. In return, we will have a lot of training data for the LLM model in a significantly shorter period of time, and our machine learning model will less likely struggle with class imbalance.

## Sources
1. From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X. (n.d.). https://arxiv.org/html/2609.02262v1