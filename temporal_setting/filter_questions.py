# Use a pipeline as a high-level helper
import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline

import datasets

# Replace with your custom model of choice


def get_label_helper(model, tokenizer, original_question, system_prompt):
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{original_question}"},
    ]
    generate_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
    }
    with torch.no_grad():
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **generate_kwargs,
            max_new_tokens=128,
        )
        all_text = pipe(message)[0]["generated_text"]

    # outputed_text = all_text.split(split_on)[-1]
    outputed_text = all_text[-1]["content"]
    print(
        f"Outputed Text for question {original_question}. The model thinks the response {outputed_text}"
    )
    if ("**Unanswerable**".lower() in outputed_text.lower()) or (
        "Unanswerable".lower() in outputed_text.lower()
    ):
        return "Unanswerable"
    elif (
        "**Answerable**".lower() in outputed_text.lower()
        or "Answerable".lower() in outputed_text.lower()
    ):
        return "Answerable"
    else:
        return "Unknown: " + outputed_text


def get_labels(entry, model, tokenizer, system_prompt):
    label = get_label_helper(model, tokenizer, entry["question"], system_prompt)
    entry["label"] = label
    if label == "Unanswerable":
        entry["label_int"] = 1
    elif label == "Answerable":
        entry["label_int"] = 0
    else:
        entry["label_int"] = -1
    print("--- label and label int ---")
    print(label)
    print(entry["label_int"])
    return entry


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
        "-qt",
        "--question_type",
        type=str,
        default="temporal",
        choices=["temporal", "illdefined"],
    )
    parser.add_argument("-m", "--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    print("Args: ", args)

    if args.question_type == "temporal":
        args.system_prompt = """The following question may require knowledge of the current date, real-time knowledge, or future knowledge like this year, this monday, or refering to something that happens in 2024 etc. If the question has temporal charactistics. Reply **Unanswerable**. If you think that there exists an answer that does not reply on temporal behvaior like which year is it, reply **Answerable**."""
    else:
        raise NotImplementedError("Only `temporal` implemented")

    print("New System Prompt", args.system_prompt)

    if args.model_path != MODEL_PATH:
        print("This is your new model path (this is not recommended)")
        print(args.model_path)
    if args.output_annotation_loc is None:
        args.output_annotation_loc = args.input
        print("Setting output path as:", args.output_annotation_loc)

    # hack to deal with jsonl
    df = pd.read_json(args.input, lines=True)
    df.columns = map(str.lower, df.columns)
    if "label" in df.columns:
        df = df.drop(columns="label")
    if "label_int" in df.columns:
        df = df.drop(columns="label_int")
    input_data = datasets.Dataset.from_pandas(df)

    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # if 'labels' not in df.columns:
    labelled_data = input_data.map(
        get_labels,
        fn_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "system_prompt": args.system_prompt,
        },
    )

    if "json" in args.output_annotation_loc:
        labelled_data.to_json(args.output_annotation_loc)
    elif "csv" in args.output_annotation_loc:
        labelled_data.to_csv(args.output_annotation_loc)
    else:
        raise NotImplementedError("Only handles `csv` and `json` output formats")
