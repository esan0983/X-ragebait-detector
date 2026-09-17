# Large-scale X Ragebait Classifier

A data science & machine learning project, with most of its inspiration taken from this [paper](https://arxiv.org/html/2609.02262v1). It is an end-to-end data pipeline, ending with a machine learning algorithm that can score posts by their ragebait and controversial factor. Supplemental applications of this project will also be included in this repository.

## Data Pipeline
### Collection
I used the twikit package. Since I don't want to get banned, I want to do this right with a robust architecture and proper API calling, so with a detailed enough prompt, I let Claude generate a data collection script. The script handles rate limits well with randomized wait times and exponential back-offs as a response to 429s.  

These are the following criteria for data collection:
* Posted on January to August 2026
* Likes + replies + reposts >= 50
* In between 50 and 280 characters  

A large table of keywords was constructed to increase the chances of collecting ragebait/controversial posts. There is a huge trade-off: a lot of the statistical analysis (especially the studies made by the Japanese paper) will be subject to selection bias. For example, we can't construct a word cloud anymore as the results will essentially just duplicate the keyword table. In return, we will have a lot of training data for the LLM model in a significantly shorter period of time.

### Cleaning
* Removed post duplicates
* Stripped white space

### Annotation
In a timely fashion, we will use TypeSafe AI's latest Jev model to perform classification. As a test, I randomly sampled 298 X posts from my dataset. Here is the question JSON structure:

```
{
  "ragebait": {
    "type": "noul",
    "instructions": "Classify this post as ragebait only if its primary purpose appears to be provoking anger or outrage rather than informing, expressing a personal experience, or stating an opinion. Do not classify a post as ragebait merely because its subject matter is disturbing, controversial, political, or emotionally charged.",
    "criteria": {
      "true": "The post uses inflammatory editorializing, mocking or insulting language, slurs, deliberately exaggerated framing, unsupported sweeping claims, or other rhetorical techniques whose apparent purpose is to provoke anger or outrage. The inflammatory framing must be directed toward provoking a reaction, not merely describing an event",
      "false": "The post reports an event, summarizes a claim or news story, describes a personal experience, or expresses an opinion—even when the subject is controversial, offensive, disturbing, or emotionally charged—provided. It does not add clear inflammatory editorializing, unsupported generalizations, or deliberate outrage-baiting framing."
    }
  },
  "controversial": {
    "type": "noul",
    "instructions": "Does this post address a polarizing, sensitive, or high-stakes topic in a way that naturally invites split opinions or intense debate, regardless of whether the author is intentionally provoking anger?",
    "criteria": {
      "true": "The content taps into a debate-heavy, polarizing, or sensitive topic likely to divide readers and generate high engagement through opposing viewpoints, including both earnest debate and intentional ragebait.",
      "false": "The post expresses a widely accepted view, a neutral factual observation, or a non-contentious personal statement that is unlikely to trigger significant disagreement or divided commentary."
    }
  }
}
```

The results were amazing:
* Each request was around 200-300ms by inspection (2/3 of the latency is most likely due to geographical distance to the West Coast)
* Total cost of 0.7 cents
* By inspecting the output, the accuracy is excellent. Even with challenging posts that are controversial in nature but not necessarily ragebait, Jev was able to implement the criteria well and give a low ragebait score and a high controversial score.

### Human Validation
Two humans and the latest Jev model will annotate 300 posts to check for agreement.

## Commit Notes (9/17)
* Performed Jev test. Very promising!

## Sources
1. From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X. (n.d.). https://arxiv.org/html/2609.02262v1