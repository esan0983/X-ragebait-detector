# Large-scale X Ragebait Classifier

A data science & machine learning project, with most of its inspiration taken from this [paper](https://arxiv.org/html/2609.02262v1). It is an end-to-end data pipeline, ending with a machine learning algorithm that can score posts by their ragebait and controversial factor. Supplemental applications of this project will also be included in this repository.

## Data Pipeline
### Collection
I used the twikit package. Since I don't want to get banned, I want to do this right with a robust architecture and proper API calling, so with a detailed enough prompt, I let Claude generate a data collection script. The script handles rate limits well with randomized wait times and exponential back-offs as a response to 429s.  

These are the following criteria for data collection:
* Posted on January to August 2026
* Likes + replies + reposts >= 50
* In between 50 and 280 characters  

A large table of keywords was constructed to increase the chances of collecting ragebait/controversial posts. There is a huge trade-off: a lot of the statistical analysis (especially the studies made by the Japanese paper) will be subject to selection bias. For example, we can't construct a word cloud anymore as the results will essentially just duplicate the keyword table. In return, we will have a lot of training data in a significantly shorter period of time.

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

For the initial collection, I got 50010 tweets, which were filtered down to 49812. I added extra criteria for my Jev input to aid with statistical analysis:

<details>
  <summary>Full Criteria</summary>

```
{
  "ragebait": {
    "type": "noul",
    "instructions": "Classify this post as ragebait only if its primary purpose appears to be provoking anger or outrage rather than informing, expressing a personal experience, or stating an opinion. Do not classify a post as ragebait merely because its subject matter is disturbing, controversial, political, or emotionally charged.",
    "criteria": {
      "true": "The post uses inflammatory editorializing, mocking or insulting language, slurs, deliberately exaggerated framing, unsupported sweeping claims, or other rhetorical techniques whose apparent purpose is to provoke anger or outrage. The inflammatory framing must be directed toward provoking a reaction, not merely describing an event",
      "false": "The post reports an event, summarizes a claim or news story, describes a personal experience, or expresses an opinion—even when the subject is controversial, offensive, disturbing, or emotionally charged—provided it does not add clear inflammatory editorializing, unsupported generalizations, or deliberate outrage-baiting framing."
    }
  },
  "controversial": {
    "type": "noul",
    "instructions": "Does this post address a polarizing, sensitive, or high-stakes topic in a way that naturally invites split opinions or intense debate, regardless of whether the author is intentionally provoking anger?",
    "criteria": {
      "true": "The content taps into a debate-heavy, polarizing, or sensitive topic likely to divide readers and generate high engagement through opposing viewpoints, including both earnest debate and intentional ragebait.",
      "false": "The post expresses a widely accepted view, a neutral factual observation, or a non-contentious personal statement that is unlikely to trigger significant disagreement or divided commentary."
    }
  },
  "time_sensitivity": {
    "type": "score",
    "instructions": "How dependent is this post's meaning or interpretability on the time in which it was posted? Judge whether a reader encountering the post without its original temporal context could still understand what it means, not whether the topic itself will remain relevant or interesting.",
    "criteria": [
      "Timeless: The post's meaning is essentially independent of when it was posted. A reader years later could understand it without knowing the original date or current events.",
      "Low: The post contains minor references to current events, trends, or circumstances, but its meaning remains mostly understandable without knowing when it was posted.",
      "Moderate: Understanding the post reasonably well requires some awareness of the period in which it was written, such as a contemporary event, trend, ongoing situation, or cultural moment.",
      "High: The post depends heavily on its original temporal context. Without knowing what was happening around the time it was posted, its meaning, implication, or reference would be difficult to understand.",
      "Ephemeral: The post is strongly tied to a moment-specific event, live situation, breaking news, reaction, trend, or rapidly changing circumstance and may become substantially unintelligible without the original context."
    ]
  },
  "nicheness": {
    "type": "score",
    "instructions": "How much specialized knowledge, cultural familiarity, community membership, or domain-specific context is required to understand the post's meaning or intended reference? Judge the accessibility of the post itself, not whether the underlying topic is popular.",
    "criteria": [
      "Broadly accessible: Most people can understand the post's meaning without specialized knowledge or familiarity with a particular community, hobby, profession, or subculture.",
      "Somewhat specialized: Most people can understand the general meaning, but familiarity with a particular interest, cultural reference, or common online context would improve comprehension.",
      "Niche: The post is primarily understandable to people familiar with a particular community, hobby, fandom, profession, game, platform culture, or specialized topic.",
      "Highly niche: Understanding the intended meaning or reference requires substantial specialized knowledge or familiarity with a specific community, subculture, technical field, or ongoing inside discussion.",
      "Extremely niche: The post is effectively directed at a very small or specialized audience and would be difficult to interpret correctly without insider knowledge or highly specific context."
    ]
  },
  "emotional_intensity": {
    "type": "score",
    "instructions": "How emotionally charged is the language of the post, regardless of whether the author is intentionally trying to provoke the reader?",
    "criteria": [
      "Neutral: Little or no emotionally charged language.",
      "Mild: Contains limited emotional language, but the overall tone remains restrained.",
      "Moderate: Noticeably emotional, with clear expressions of enthusiasm, frustration, sadness, anger, disgust, excitement, or similar emotion.",
      "High: Strong emotional language, emphatic wording, insults, profanity, or dramatic expression is prominent throughout the post.",
      "Extreme: The post is overwhelmingly emotionally charged, highly aggressive, explosive, or intensely expressive."
    ]
  },
  "content_type": {
    "type": "choice",
    "instructions": "What is the primary communicative function of the post?",
    "criteria": {
      "factual_report": "Primarily reports or describes an event, observation, or information.",
      "claim": "Primarily makes a factual or causal claim that could in principle be evaluated as true or false.",
      "opinion": "Primarily expresses a subjective judgment, preference, belief, or evaluation.",
      "personal_experience": "Primarily describes the author's own experience, feelings, or circumstances.",
      "question": "Primarily asks a question or solicits information rather than making a substantive claim.",
      "humor_or_entertainment": "Primarily attempts to entertain, joke, parody, or amuse rather than communicate a serious claim or opinion.",
      "other": "Does not fit the other categories clearly."
    }
  },
  "context_dependence": {
    "type": "score",
    "instructions": "How much external context is required to correctly understand the meaning or intended implication of the post?",
    "criteria": [
      "Self-contained: The post can be understood accurately on its own.",
      "Minor context: Some additional context would improve understanding, but the main meaning is clear.",
      "Moderate context: External information is needed to fully understand the reference, implication, or situation being discussed.",
      "High context: The post is difficult to interpret correctly without substantial knowledge of an external event, conversation, media, or preceding content.",
      "Extreme context: The post is largely unintelligible or highly ambiguous without very specific external context."
    ]
  },
  "rhetorical_target": {
    "type": "choice",
    "instructions": "Who or what is the post primarily directing its criticism, praise, mockery, or other rhetoric toward?",
    "criteria": {
      "none": "The post does not have a clear rhetorical target.",
      "individual": "A specific identifiable person is the primary target.",
      "group": "A demographic, social, political, professional, cultural, or other group is the primary target.",
      "organization": "A company, institution, government, media outlet, team, community, or other organization is the primary target.",
      "product_or_work": "A product, service, piece of media, game, artwork, or other specific work is the primary target.",
      "abstract_issue": "An idea, policy, behavior, event, phenomenon, or abstract concept is the primary target.",
      "multiple": "The post targets multiple distinct entities without one clearly being primary."
    }
  },
  "generalization": {
    "type": "score",
    "instructions": "How broadly does the post generalize beyond the specific person, event, or situation being discussed?",
    "criteria": [
      "Specific: The statement is limited to a particular person, event, situation, or clearly defined example.",
      "Limited generalization: The post makes a modest generalization but retains meaningful qualifications or limitations.",
      "Broad: The post generalizes across a substantial category of people, events, or situations.",
      "Sweeping: The post makes a broad, categorical, or absolute claim about a large group or class with little qualification.",
      "Extreme: The post makes an exceptionally broad or absolute generalization that reduces a complex subject to a simplistic categorical claim."
    ]
  },
  "sarcasm_irony": {
    "type": "score",
    "instructions": "To what extent does the post rely on sarcasm, irony, parody, or deliberately non-literal language to communicate its intended meaning?",
    "criteria": [
      "None: The post is substantially literal and does not rely on sarcasm or irony.",
      "Possible: The wording contains mild or ambiguous ironic elements, but the intended meaning remains substantially literal.",
      "Present: Sarcasm or irony is clearly used as part of the post's communication.",
      "Strong: Understanding the intended meaning requires recognizing substantial sarcasm, irony, parody, or deliberate reversal of the literal wording.",
      "Essential: The post's intended meaning is primarily dependent on interpreting its language non-literally."
    ]
  }
}
```

</details>  

\
Even with an incredibly detailed instructions and criteria, the results were still good: a total cost of **$4.71** and a total runtime of **2 hours and 48 minutes**. Note that I haven't implemented concurrency: this can be even faster.

The first batch of annotated data will then be trained via a RoBERTa-base model (refer to Machine Learning for both phases of training)


### Human Validation
Two humans and the latest Jev model will annotate 300 posts to check for agreement.

## Commit Notes (9/21)
* Project is reorganized
* Finished training initial classifier model
* Set up some EDA for first batch
* Prepared src files to perform dataframe splitting and inference

## Post-Commit Plans:
* Make a browser extension that detects ragebait potential
* Flesh out and make the data pipeline section of the README easier to read. Right now, the audience has no idea on what the data pipeline looks like.

## Sources
1. From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X. (n.d.). https://arxiv.org/html/2609.02262v1