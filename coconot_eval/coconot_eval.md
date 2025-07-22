## CoCoNot Evaluation

The code in this folder contains scripts for generating completions from the models trained with refusal tokens and for evaluating the models under various decoding strategies.

* For a single refusal token, please use the `generate_for_eval_threshold.py` script.
* For multiple refusal tokens, please use the `generate_for_eval_threshold_cat_tokens.py` script.
* For evaluating the generations, please use `eval_responses_coconot.py`.