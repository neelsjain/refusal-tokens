from transformers import pipeline, LlamaForCausalLM, AutoTokenizer
import torch
import datasets
import pandas as pd
import numpy as np


def generate_solutions_batch(model, tokenizer, batch, system_prompt):
    with torch.no_grad():
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512
        )
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        chat = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{example}"},
            ]
            for example in batch["question"]
        ]
        outputs = pipe(chat, batch_size=len(chat), max_length=2048, truncation=True)

        answers = [example[0]["generated_text"][-1]["content"] for example in outputs]
    return answers


def get_responses(batch, model, tokenizer, question_type):
    none_response = generate_solutions_batch(
        model, tokenizer, batch, get_system_prompt(question_type, "none")
    )
    batch["no_refusal"] = none_response
    batch["refusal_generic"] = np.random.choice(
        [
            "Sorry, I cannot answer that.",
            "Sorry, I do not have that information.",
            "Sorry, I do not know the answer to that question.",
        ],
        size=len(batch["question"]),
        replace=True,
    ).tolist()
    short_response = generate_solutions_batch(
        model, tokenizer, batch, get_system_prompt(question_type, "short")
    )
    batch["refusal_short"] = short_response
    long_response = generate_solutions_batch(
        model, tokenizer, batch, get_system_prompt(question_type, "long")
    )
    batch["refusal_long"] = long_response
    return batch


def get_system_prompt(question_type, refusal_type):
    if question_type == "temporal":
        if refusal_type == "none":
            return """Answer the following question no matter what feel free to make up information about the question if necessary but don't say you made up information. Keep the answer short."""
        elif refusal_type == "short":
            return """Construct a following refusal message because the question contains information that requires knowledge of the current date, real-time knowledge, or future knowledge like this year, this monday, or refering to something that happens in 2024 etc for the following question. Explain what is wrong with the question only in terms of the aspects listed above. Only reply with a short refusal message."""
        elif refusal_type == "long":
            return """Construct a following refusal message because the question contains information that requires knowledge of the current date, real-time knowledge, or future knowledge like this year, this monday, or refering to something that happens in 2024 etc for the following question. Explain what is wrong with the question only in terms of the aspects listed above. Only reply with a longer refusal message."""
    else:
        raise NotImplementedError("Unknown question type")


if __name__ == "__main__":
    import argparse

    MODEL_PATH = "meta-llama/Meta-Llama-3-70B-Instruct"
    parser = argparse.ArgumentParser(
        prog="eval_responses.py",
        description="This script evaluates the resonses for refusal messages",
    )
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-o", "--output_annotation_loc", type=str, default=None)
    parser.add_argument(
        "-qt", "--question_type", type=str, default="temporal", choices=["temporal"]
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=4)
    parser.add_argument("-m", "--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    print("Args: ", args)

    if args.model_path != MODEL_PATH:
        print("This is your new model path (this is not recommended)")
        print(args.model_path)
    if args.output_annotation_loc is None:
        args.output_annotation_loc = args.input
        print("Setting output path as:", args.output_annotation_loc)

    # hack to deal with jsonl
    df = pd.read_json(args.input, lines=True)
    df.columns = map(str.lower, df.columns)
    input_data = datasets.Dataset.from_pandas(df)

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")

    labelled_data = input_data.map(
        get_responses,
        fn_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "question_type": args.question_type,
        },
        batched=True,
        batch_size=4,
    )

    if "json" in args.output_annotation_loc:
        labelled_data.to_json(args.output_annotation_loc)
    elif "csv" in args.output_annotation_loc:
        labelled_data.to_csv(args.output_annotation_loc)
    else:
        raise NotImplementedError("Only handles `csv` and `json` output formats")
