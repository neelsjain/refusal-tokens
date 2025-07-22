from utils import get_eval_dataset_as_hf, get_model_tokenizer
from transformers import pipeline
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def generate_solutions_batch(
    batch,
    model,
    tokenizer,
    threshold=0.5,
    system_prompt=None,
    seed=None,
    softmax_temperature=1.0,
    args=None,
):
    if seed is not None:
        set_seed(seed)

    with torch.no_grad():
        model.generation_config.stop_strings = [tokenizer.eos_token, "<|eot_id|>"]
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        pipe.tokenizer.pad_token_id = model.config.eos_token_id
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

        refuse_token = tokenizer.encode(" [refuse]")[-1]
        respond_token = tokenizer.encode(" [respond]")[-1]
        messages = tokenizer.apply_chat_template(
            chat,
            padding="longest",
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        messages = messages.to(device)
        # tokenized_messages = tokenizer.tokenize(messages,padding="left", return_tensors='pt')
        outputs = model(**messages)
        predictions = torch.nn.functional.softmax(
            outputs.logits / softmax_temperature, dim=-1
        )
        refuse_token_prob = predictions[:, -1, refuse_token]

        if softmax_temperature > 1.0:
            print(
                "Before: ",
                torch.nn.functional.softmax(outputs.logits, dim=-1)[
                    :, -1, refuse_token
                ],
            )
            print("After: ", refuse_token_prob)

        if threshold == -1:
            respond_token_prob = predictions[:, -1, respond_token]
            add_refuse_token_prob = refuse_token_prob >= respond_token_prob
            tokens_to_add = torch.where(
                add_refuse_token_prob, refuse_token, respond_token
            ).reshape(len(batch["question"]), 1)
        else:
            add_refuse_token_prob = refuse_token_prob >= threshold
            tokens_to_add = torch.where(
                add_refuse_token_prob, refuse_token, respond_token
            ).reshape(len(batch["question"]), 1)
        print(
            "Original and then New:\n",
            outputs.logits.argmax(dim=-1)[:, -1].cpu().tolist(),
            "\n",
            tokens_to_add.flatten().tolist(),
        )
        ones_for_attention = torch.ones_like(tokens_to_add)
        messages["input_ids"] = torch.cat((messages.input_ids, tokens_to_add), dim=-1)
        messages["attention_mask"] = torch.cat(
            (messages.attention_mask, ones_for_attention), dim=-1
        )
        assert messages["attention_mask"].shape == messages["input_ids"].shape
        gen_outputs = model.generate(
            **messages, max_new_tokens=128, tokenizer=tokenizer
        )

        answers = [
            tokenizer.decode(
                gen_outputs[i][len(messages.input_ids[i]) - 1 :],
                skip_special_tokens=True,
            )
            for i in range(len(gen_outputs))
        ]
        model_name_for_batch = [model.config._name_or_path] * len(answers)
        batch["model_response"] = answers
        batch["model"] = model_name_for_batch
    return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="eval_responses.py",
        description="This script evaluates the resonses for refusal messages",
    )
    parser.add_argument("-o", "--output_file", type=str, default=None)
    parser.add_argument("-m", "--model_path", type=str, default=None, required=True)
    parser.add_argument("-d", "--eval_data", type=str, default="coconot_eval.csv")
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument(
        "-sp", "--system_prompt", type=str, default="You are a helpful assistant."
    )
    parser.add_argument("-t", "--threshold", type=float, default=-1)
    parser.add_argument("-temp", "--temperature", type=float, default=None)
    parser.add_argument("-st", "--softmax_temperature", type=float, default=1.0)
    parser.add_argument("-s", "--seed", type=int, default=None)

    args = parser.parse_args()
    print("Args: ", args)
    if "zephyr" in args.model_path:
        args.system_prompt = None
    model, tokenizer = get_model_tokenizer(args.model_path)
    data = get_eval_dataset_as_hf(data_path=args.eval_data)
    if "question" not in data.column_names and "prompt" in data.column_names:
        data = data.rename_column("prompt", "question")
        print("Renamed prompt to question")

    data_output = data.map(
        generate_solutions_batch,
        fn_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "threshold": args.threshold,
            "system_prompt": args.system_prompt,
            "seed": args.seed,
            "softmax_temperature": args.softmax_temperature,
            "args": args,
        },
        batched=True,
        batch_size=args.batch_size,
    )

    if "json" in args.output_file:
        data_output.to_json(args.output_file)
    elif "csv" in args.output_file:
        data_output.to_csv(args.output_file)
    else:
        raise NotImplementedError("Only handles `csv` and `json` output formats")
