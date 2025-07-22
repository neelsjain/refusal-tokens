from utils import get_eval_dataset_as_hf, get_model_tokenizer
from transformers import pipeline
import torch
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    tokens_to_suppress=None,
    system_prompt=None,
    generate_only_first_token=False,
    args=None,
):
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

        refuse_cat_to_add = [
            " [Humanizing requests]",
            " [Incomplete requests]",
            " [Indeterminate requests]",
            " [Requests with safety concerns]",
            " [Unsupported requests]",
        ]
        if tokens_to_suppress is None:
            raise ValueError("Only implemented for Coconot Categories")
        else:
            all_refuse_tokens = []
            refuse_tokens_suppress = []
            for token_ in refuse_cat_to_add:
                token_ = " " + token_.strip()
                all_refuse_tokens.append(tokenizer.encode(token_)[-1])
                if token_ in tokens_to_suppress:
                    refuse_tokens_suppress.append(tokenizer.encode(token_)[-1])
            respond_token = tokenizer.encode(" [respond]")[-1]

        if len(refuse_tokens_suppress) == 0:
            raise ValueError("Check passed the categories")

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
        for softmax_temp in [1, 1.5, 2]:
            args.softmax_temp = softmax_temp
            predictions = torch.nn.functional.softmax(
                outputs.logits / args.softmax_temp, dim=-1
            )
            refuse_tokens_prob = predictions[:, -1, all_refuse_tokens]
            token_refuse_class_idx = torch.argmax(refuse_tokens_prob, dim=-1)
            #### get probabilities of the

        #### GET ALL CLASS TOKENS PROBABILITIES ####
        original_predicition = torch.argmax(predictions[:, -1, :], dim=-1)
        refuse_tokens_prob = predictions[:, -1, all_refuse_tokens]
        token_refuse_class_idx = torch.argmax(refuse_tokens_prob, dim=-1)
        probabilites = torch.max(refuse_tokens_prob, dim=-1)
        token_refuse_class = torch.take(
            torch.Tensor(all_refuse_tokens).cuda(), token_refuse_class_idx
        ).to(int)

        in_token_to_supress = (
            token_refuse_class.clone()
            .cpu()
            .apply_(lambda x: x in refuse_tokens_suppress)
            .bool()
            .cuda()
        )

        thresholding_swap = probabilites.values >= threshold
        tokens_if_respond = torch.where(
            in_token_to_supress, respond_token, original_predicition
        )
        tokens_if_refuse = torch.where(
            in_token_to_supress, token_refuse_class, original_predicition
        )

        condition_to_refuse_selected_tokens = torch.logical_and(
            thresholding_swap, in_token_to_supress
        )
        tokens_to_add = torch.where(
            condition_to_refuse_selected_tokens, tokens_if_refuse, tokens_if_respond
        ).reshape(len(batch["question"]), 1)

        for element_1, element_2 in zip(
            outputs.logits.argmax(dim=-1)[:, -1].cpu().tolist(),
            tokens_to_add.flatten().tolist(),
        ):
            if element_1 != element_2:
                print(
                    f"Switch from {tokenizer.decode(element_1)} to {tokenizer.decode(element_2)}"
                )
        ones_for_attention = torch.ones_like(tokens_to_add)
        messages["input_ids"] = torch.cat((messages.input_ids, tokens_to_add), dim=-1)
        messages["attention_mask"] = torch.cat(
            (messages.attention_mask, ones_for_attention), dim=-1
        )
        assert messages["attention_mask"].shape == messages["input_ids"].shape
        if generate_only_first_token:
            answers = [
                tokenizer.decode(
                    messages.input_ids[i][len(messages.input_ids[i]) - 1 :],
                    skip_special_tokens=True,
                )
                for i in range(len(messages.input_ids))
            ]
        else:
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
        if generate_only_first_token:
            first_token = [
                tokenizer.decode(
                    messages.input_ids[i][len(messages.input_ids[i]) - 1 :],
                    skip_special_tokens=True,
                )
                for i in range(len(messages.input_ids))
            ]
            first_token_id = [
                messages.input_ids[i][len(messages.input_ids[i]) - 1]
                for i in range(len(messages.input_ids))
            ]
            batch["first_token"] = first_token
            batch["label"] = [
                "respond"
                if respond_token in first_token_id
                else "refuse"
                if first_token_id in all_refuse_tokens
                else "unknown"
                for first_token_id in first_token_id
            ]

    return batch


if __name__ == "__main__":
    import argparse

    MODEL_PATH = "../llm-c-model-registry/meta-llama_meta-llama-3-8b-instruct/"
    parser = argparse.ArgumentParser(
        prog="eval_responses.py",
        description="This script evaluates the resonses for refusal messages",
    )
    parser.add_argument("-o", "--output_file", type=str, default=None)
    parser.add_argument("-m", "--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("-d", "--eval_data", type=str, default="coconot_eval.csv")
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument(
        "-sp", "--system_prompt", type=str, default="You are a helpful assistant."
    )
    parser.add_argument("-t", "--threshold", type=float, default=None)
    parser.add_argument("-ts", "--tokens_supress", type=str, default=None)
    parser.add_argument("-gof", "--generate_only_first_token", type=bool, default=False)
    parser.add_argument("-st", "--softmax_temp", type=float, default=1.0)
    args = parser.parse_args()
    print("Args: ", args)
    if "zephyr" in args.model_path:
        args.system_prompt = None
    model, tokenizer = get_model_tokenizer(args.model_path)
    data = get_eval_dataset_as_hf(data_path=args.eval_data)

    if "question" not in data.column_names and "prompt" in data.column_names:
        data = data.rename_column("prompt", "question")
        print("Renamed prompt to question")
    fn_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "threshold": args.threshold,
        "system_prompt": args.system_prompt,
        "tokens_to_suppress": ast.literal_eval(args.tokens_supress),
        "generate_only_first_token": args.generate_only_first_token,
        "do_cheap_sweep_thres": args.cheap_sweap,
        "args": args,
    }
    data_output = data.map(
        generate_solutions_batch,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=args.batch_size,
    )
    if "json" in args.output_file:
        data_output.to_json(args.output_file)
    elif "csv" in args.output_annotation_loc:
        data_output.to_csv(args.output_file)
    else:
        raise NotImplementedError("Only handles `csv` and `json` output formats")
