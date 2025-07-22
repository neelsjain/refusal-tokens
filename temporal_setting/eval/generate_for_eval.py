from utils import get_model_tokenizer, get_eval_dataset_as_hf
from transformers import pipeline
import torch
import numpy as np
import os


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def generate_solutions(example, model, tokenizer):
    with torch.no_grad():
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=tokenizer.model_max_length,
        )
        chat = [
            {"role": "user", "content": f"{example['question']}"},
        ]
        example["model_response"] = pipe(chat)[0]["generated_text"][-1]["content"]
        example["model"] = model.config._name_or_path
    return example


def generate_solutions_batch(batch, model, tokenizer, system_prompt=None, seed=None):
    if seed is not None:
        set_seed(seed)

    with torch.no_grad():
        model.generation_config.stop_strings = [
            tokenizer.eos_token,
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
        ]
        generate_kwargs = {
            "do_sample": True,
            "temperature": 0.6,
        }
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, **generate_kwargs
        )
        print("model eos token id: ", model.config.eos_token_id)
        print("tokenizer eos token id: ", tokenizer.eos_token_id)
        pipe.tokenizer.pad_token_id = tokenizer.eos_token_id

        if system_prompt is None:
            chat = [
                [{"role": "user", "content": f"{example}"}]
                for example in batch["question"]
            ]
        else:
            chat = [
                [
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": f"{example}"},
                ]
                for example in batch["question"]
            ]
        print(tokenizer.apply_chat_template(chat[0], tokenize=True))
        print(
            tokenizer.apply_chat_template(
                chat[0], add_generation_prompt=True, tokenize=False
            )
        )
        outputs = pipe(
            chat,
            tokenizer=tokenizer,
            batch_size=len(chat),
            max_new_tokens=128,
            truncation=True,
        )

        answers = [example[0]["generated_text"][-1]["content"] for example in outputs]

        print(answers)
        model_name_for_batch = [model.config._name_or_path] * len(answers)
        batch["model_response"] = answers
        batch["model"] = model_name_for_batch
    return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="This script evaluates the resonses for refusal messages"
    )
    parser.add_argument("-o", "--output_file", type=str, default=None)
    parser.add_argument("-m", "--model_path", type=str, default=None, required=True)
    parser.add_argument(
        "-d",
        "--eval_data",
        type=str,
        default="refusal_eval_data/refusal_eval_temporal.csv",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("-sp", "--system_prompt", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=None)

    args = parser.parse_args()
    print("Args: ", args)
    #### CHECK THAT GENERATION DON'T ALREADY EXIST
    if args.output_file is not None and os.path.exists(args.output_file):
        #### CHECK MODEL RESPONSE COLUMN IS FULL
        import pandas as pd

        output_data = pd.read_json(args.output_file, lines=True)
        if (
            "model_response" in output_data.columns
            and output_data["model_response"].notnull().all()
        ):
            print(
                f"Output file {args.output_file} already exists with all model responses and is complete. Skipping generation."
            )
            exit(0)

    model, tokenizer = get_model_tokenizer(args.model_path)
    print("model eos token id: ", model.config.eos_token_id)
    print("tokenizer eos token id: ", tokenizer.eos_token_id)
    data = get_eval_dataset_as_hf(data_path=args.eval_data)
    # change prompt column to question column if no question column
    if "question" not in data.column_names and "prompt" in data.column_names:
        data = data.rename_column("prompt", "question")
        print(f"Renamed prompt to question")
    data_output = data.map(
        generate_solutions_batch,
        fn_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": args.system_prompt,
            "seed": args.seed,
        },
        batched=True,
        batch_size=args.batch_size,
    )
    if "json" in args.output_file:
        data_output.to_json(args.output_file)
    elif "csv" in args.output_annotation_loc:
        data_output.to_csv(args.output_file)
    else:
        raise NotImplementedError("Only handles `csv` and `json` output formats")
