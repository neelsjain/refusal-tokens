def get_model_tokenizer(model_name_or_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="left", use_fast=False
    )
    return model, tokenizer


def get_eval_dataset_as_hf(data_path="refusal_eval_data/refusal_eval_v1.csv"):
    import datasets
    import pandas as pd

    data_path = data_path
    data = pd.read_csv(data_path, index_col=None)
    data = datasets.Dataset.from_pandas(data)
    return data
