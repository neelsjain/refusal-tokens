## Codebase for *Refusal Tokens: A Simple Way to Calibrate Refusals in Large Language Models*

This is the official repo for [*Refusal Tokens: A Simple Way to Calibrate Refusals in Large Language Models*](https://openreview.net/forum?id=Pbs4i3FgbD&noteId=FiT1vPztJQ), published at the Conference on Language Modeling (COLM) 2025.

**MODELS**: The models can be found in the [Hugging Face collection](https://huggingface.co/collections/tomg-group-umd/refusal-token-paper-models-687f8d2898cea31c3b17f490). The Zephyr recipe and codebase were used to train these models and can be found [here](https://github.com/huggingface/alignment-handbook).

**EVALUATION**: The evaluation code used in the paper is from the CoCoNoT evaluation, described in [this paper](https://arxiv.org/abs/2407.12043). The Temporal Setting materials can be found in their respective folders.

**Temporal Setting**: Instructions for generating training data and evaluating the model can be found [here](temporal_setting/temporal.md).

**CoCoNoT Evaluation**: Details for the CoCoNoT evaluation can be found [here](coconot_eval/coconot_eval.md).

<!-- 
TODOs:
- [] Add more information on running the scripts.
- [] Add the exact config used in the alignment-handbook code base
-->

Please send any inquiries to `njain17` at umd dot edu.