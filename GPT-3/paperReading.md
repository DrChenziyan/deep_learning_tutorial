## GPT-3: Language Models are Few-Shot Learners
1. [Paper](https://arxiv.org/pdf/2005.14165.pdf)
2. Introduced by OpenAI
3. Reading with Yannic Kicher

## Paper Reading 
### Abstract
1. For humans, they can perform a good language task from only small examples or intrustions, for machine, however, to achieve the same level performance, large datasets for pre-training is necessary.
2. In this paper, authors scaled up the model architectures(175 billion parameters) with little and easy fine-tuning.
3. Even though a model is task-agnostic, but it has to fine-tuning for specific tasks such as translation, question answering, reading comprehension.
4. The GPT-3 means to be trained more robust by larger datasets so as to little fine-tuning for specific tasks
```
Q1: how to build the model and fine-tuning?
Q2: to train the model, what kind of datasets did they use? (may have conflicts with human learning experience)
```
### Contents
1. **Background**
    - NLP tasks are too complicited to gain good performance in each field by one model
    - pre-training and fine-tuning conflicits beacuse of wider distributions of datasets used for pre-training but narrower task distributions in fine-tuning.
    - task switching 

2. **Datasets**

| Datasets | Quantity(tokens) | Weight in training mix |
| :--: | :--: | :--: |
| Common Crawl | 410 billion | 60% |
| WebText2 | 19 billion | 22% |
| Books1 | 12 billion | 8% |
| Books2 | 55 billion | 8% |
| Wikipedia | 3 billion | 3% |

3. **Model Architecture** 
    - The usage of the model is tasks fine-tuning
```
zero-shot: you give a task description and prompt to the model and no gradient updates are performed, just rely on the model seeing this kind of e.g in training.
one-shot: a description + a example + a prompt
few-shot: a description + few examples + a prompt
```
   - Based on Attention

   - The large parameters stored in the model, in other words, the large infomation stored in the model, leading it to fit different task, but we can see that the preformance of GPT-3 few-shot just won a little to one-shot, suggesting maybe one example enough for model ectracting infomation. 


4. **Results**
    - validation loss decreased followed with scaled-up models and more computing time.
    - Question answering: larger model with few-shot, higher accuracy
    - Translation: the model prefromed well when translating such as French, Romaniam, German to English
    - Winogrande: the model could get similiar performance as the specific task designed models.
    - Common sence resoning:
    - Reading comprehension: not so well to other models (these tasks are not language model but reasoning which resulted in poor accuracy)
    - Natural language inference:
      - it concerns the ability to understand the relationship between 2 sentences.
      - the model classifies whether the second sentence logically follows from the first, contradicts the first, or is possibly true.
      - GPT-3 perfomed *perticularly poor*
    - Arhtimetic: as the model parameters go larger, it can do some logical compution(+ - * )
      - especially, 2 or 3 digit addition or subtraction showed pretty well accuracy but poor at  more numbers addution or subtraction and multiplication  
    - Wordscramble(cycle letters, random insertion, reversed words): not so good 
    - SAT Analogies: 
      - Analogies are a style of multiple choice question that constituted a section of the SAT college entrance exam(近义词)
      - not so good, even few-shot only reached about 60% accuracy with largerest parameters(understanding).
    - News Article generation:
      - as the model size increased, the abilities of its article generation made human harder to realize, relatively. And the accuracy of correcting differentiating is even only 12%. 
      - Just give the model: title and subtitle, it can generate the article itself which humans had the greatest difficulty distinguishing from a human written one.
    - Make up words: 
      - gramma modify 


5. **Summary**
   1. The model is too large to achieve personally for its huge-huge-huge size and huge-huge-huge datasets
   2. Honestly, the model perfomed pretty well in different specific language tasks such Q&A, translation, even artile generation, with little fine-tuning.
   3. By training such size model, in many task, its performance only gets general levels, how to improve it? still scaling up until new efficient architecture comes up or something else?
   4. GPT-3 is a general model which could fit most tasks but not every task could realize good results, in present, should build models which for more distributed tasks or which for specific task?  I think the latter would be more meaningful in these years until the hardware updated to next generation or more novel architecture appearance.


