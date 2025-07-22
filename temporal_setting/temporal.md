## Temporal Data

The exact data used in the paper is unfortunately **not** available due to corporate reasons. However, the following are attached:

### Scripts to Generate the Data

* A developer Guardian API key is required, which can be found [here](https://open-platform.theguardian.com/access/). Export this key using `export GUARDIAN_NEWS_API_KEY=...`.
* To generate unanswerable questions, use `python create_temporal_questions_unanswerable.py --guardian_key $GUARDIAN_NEWS_API_KEY`, and for answerable questions, use `python create_temporal_questions_answerable.py --guardian_key $GUARDIAN_NEWS_API_KEY`. Note that these two scripts are almost identical except in how they draw the articles. Originally, these were intended to be expanded to other question types as well before AllenAI's Coconot paper.
* After creating your answerable or unanswerable questions, you will want to confirm that these questions were created in the desired manner. To do so, run `python filter_questions.py --input $INPUT_FILE` to get labels on whether LLaMA-3-70B considers the questions answerable or unanswerable. Then, you may want to filter out those that do not match your criteria. For example, if the questions are supposed to be answerable, you would filter out the unanswerable ones. This can be done in about four lines of code using pandas.
* To generate refusal messages, use `python create_refusal_messages.py`. This script constructs four types of responses: a response, a generic refusal, a short refusal, and a long refusal message. Initially, we found that for SFT, this had little impact on refusal rates and thus used short refusals. It is worth mentioning that other papers, like [this one](https://openreview.net/forum?id=6Mxhg9PtDE), offer a more nuanced study on this matter.

### Data Included

In the `temporal_data_example` folder is an example of what the data pipeline might yield. This data can also be used for experiments.

### Evaluation

For evaluation, please see the files found in the `eval` folder and read through the respective flags to understand the best configuration for your use case.

